"""
Download hourly ERA5 fields for a selected domain into chunked, compressed Zarr stores.

Source: Google's public ARCO-ERA5 raw bucket, one global NetCDF3 file per day,
single and pressure levels. Only the selected domain rows are fetched.

Each variable is written to its own Zarr store. The time axis is split into blocks
equal to the Zarr time chunk, so blocks are downloaded and written in parallel
without two writers touching the same chunk. Finished blocks are recorded in a
manifest and skipped when the command is run again.
"""

import argparse
import json
import math
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import aiohttp
import dask.array as dsa
import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from zarr.codecs import BloscCodec

import utils

RETRIES = 3


def parse_args(argv=None) -> utils.Settings:
    """Read run settings supplied by the Slurm script or a direct invocation."""
    parser = argparse.ArgumentParser(description="Download ERA5 fields into Zarr stores.")
    parser.add_argument("--source-url", default="https://storage.googleapis.com/gcp-public-data-arco-era5/raw")
    parser.add_argument("--start", required=True, type=date.fromisoformat, help="first day, YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=date.fromisoformat, help="last day inclusive, YYYY-MM-DD")
    parser.add_argument("--bbox", required=True, nargs=4, type=float,
                        metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    parser.add_argument("--vars", required=True, action="append", nargs="+", metavar="VAR",
                        help="NAME LEVEL_TYPE SOURCE_NAME [LEVEL ...]; repeat for each variable")
    parser.add_argument("--chunks", nargs=3, type=int, default=[720, 35, 79], metavar=("TIME", "LAT", "LON"))
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = parser.parse_args(argv)

    # Each download block covers whole days, one source file per day.
    if args.chunks[0] % 24:
        parser.error("the time chunk must be a multiple of 24 hours")
    variables = [utils.Variable(name, level_type, source_name, [int(level) for level in levels])
                 for name, level_type, source_name, *levels in args.vars]
    return utils.Settings(
        source_url=args.source_url.rstrip("/"), start=args.start, end=args.end,
        bbox=dict(zip(("lat_min", "lat_max", "lon_min", "lon_max"), args.bbox)),
        vars=variables, chunks=dict(zip(("time", "lat", "lon"), args.chunks)),
        max_workers=args.max_workers, data_dir=args.data_dir, log_dir=args.log_dir,
    )


@dataclass
class Grid:
    """Bounding-box selection on the source grid and the matching output coordinates."""

    rows: slice  # source rows inside the bbox (source latitude is descending)
    cols: slice  # source columns inside the bbox
    lat: np.ndarray  # output latitude, ascending
    lon: np.ndarray  # output longitude, -180..180
    attrs: dict  # units and names of the source variable


def source_url(settings: utils.Settings, var: utils.Variable, level: int | None, day) -> str:
    """URL of the daily source file for one variable (and pressure level)."""
    leaf = "surface.nc" if level is None else f"{level}.nc"
    return f"{settings.source_url}/date-variable-{var.level_type}/{day:%Y/%m/%d}/{var.source_name}/{leaf}"


def read_layouts(fs, urls: list[str]) -> list[utils.Nc3Layout]:
    """Fetch and parse the header of each file in one concurrent batch."""
    n = len(urls)
    headers = fs.cat_ranges(urls, [0] * n, [utils.HEADER_BYTES] * n, on_error="raise")
    return [utils.read_nc3_layout(h) for h in headers]


def source_grid(fs, url: str, bbox: dict) -> Grid:
    """Read one source header and work out the bbox rows, columns and output coordinates."""
    (layout,) = read_layouts(fs, [url])
    rows, cols = utils.bbox_rows_cols(layout.lat, layout.lon, bbox)
    return Grid(
        rows=rows,
        cols=cols,
        lat=layout.lat[rows][::-1].astype("float64"),
        lon=((layout.lon[cols] + 180) % 360 - 180).astype("float64"),
        attrs=layout.attrs,
    )


def fetch_days(fs, urls: list[str], days: pd.DatetimeIndex, rows: slice, cols: slice) -> np.ndarray:
    """Read the subregion data of several daily files with concurrent range requests.

    Return: a float32 array (hours, rows, cols) in source row order (latitude descending).
    """
    layouts = read_layouts(fs, urls)
    for url, day, layout in zip(urls, days, layouts):
        if not layout.time.equals(pd.date_range(day, periods=24, freq="h")):
            raise ValueError(f"{url}: time values are not the 24 hours of {day:%Y-%m-%d}")

    paths, starts, ends = [], [], []
    for url, layout in zip(urls, layouts):
        file_starts, file_ends = utils.bbox_byte_ranges(layout, rows)
        paths += [url] * len(file_starts)
        starts += file_starts
        ends += file_ends
    blobs = fs.cat_ranges(paths, starts, ends, on_error="raise")

    parts, i = [], 0
    for layout in layouts:
        n = layout.shape[0]
        parts.append(utils.decode_rows(blobs[i:i + n], layout, rows, cols))
        i += n
    return np.concatenate(parts)


def zarr_encoding(name: str, chunks: tuple) -> dict:
    """Chunking and compression for every variable."""
    return {name: {
        "chunks": chunks,
        "compressors": [BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")],
        "_FillValue": np.nan,
    }}


def create_store(settings: utils.Settings, var: utils.Variable, grid: Grid) -> None:
    """Write the Zarr metadata and coordinates for the full time axis, without data chunks.
    Chunks that are never written read back as NaN.
    """
    times = settings.times
    c = settings.chunks
    coords = {"time": times, "lat": grid.lat, "lon": grid.lon}
    if var.level_type == "single_level":
        dims = ("time", "lat", "lon")
        chunks = (c["time"], c["lat"], c["lon"])
    else:
        dims = ("time", "level", "lat", "lon")
        chunks = (c["time"], 1, c["lat"], c["lon"])
        coords = {"time": times, "level": var.levels, "lat": grid.lat, "lon": grid.lon}
    shape = tuple(len(coords[d]) for d in dims)
    empty = dsa.full(shape, np.nan, dtype="float32", chunks=chunks)

    ds = xr.Dataset({var.name: (dims, empty, grid.attrs)}, coords=coords)
    ds["lat"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
    ds["lon"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
    if "level" in ds.coords:
        ds["level"].attrs = {"units": "hPa", "long_name": "pressure level"}
    ds.attrs = {
        "title": f"ERA5 hourly {var.source_name}",
        "source": "ERA5 reanalysis, ARCO-ERA5 raw files (gs://gcp-public-data-arco-era5/raw)",
        "geospatial_lat_min": settings.bbox["lat_min"],
        "geospatial_lat_max": settings.bbox["lat_max"],
        "geospatial_lon_min": settings.bbox["lon_min"],
        "geospatial_lon_max": settings.bbox["lon_max"],
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # Consolidated metadata lets readers open the store with a single metadata read. It stays
    # valid because the block writes below only add data chunks. Zarr warns that consolidated
    # metadata is not yet part of the v3 specification; that notice is expected here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Consolidated metadata", category=UserWarning)
        ds.to_zarr(settings.store_path(var), mode="w-", compute=False, consolidated=True,
                   encoding=zarr_encoding(var.name, chunks))


def write_block(settings: utils.Settings, var: utils.Variable, level: int | None, t0: int, t1: int, data: np.ndarray) -> None:
    """Write hours t0..t1 (latitude ascending) into their region of the variable's store."""
    if level is None:
        ds = xr.Dataset({var.name: (("time", "lat", "lon"), data)})
        region = {"time": slice(t0, t1)}
    else:
        k = var.levels.index(level)
        ds = xr.Dataset({var.name: (("time", "level", "lat", "lon"), data[:, np.newaxis])})
        region = {"time": slice(t0, t1), "level": slice(k, k + 1)}
    ds.to_zarr(settings.store_path(var), region=region, consolidated=False)


def finished_blocks(settings: utils.Settings) -> set[tuple[str, int]]:
    """(series, block) pairs already recorded in the manifest."""
    if not settings.manifest_path.exists():
        return set()
    with open(settings.manifest_path) as f:
        return {(rec["series"], rec["block"]) for rec in map(json.loads, f)}


def run_block(fs, settings, var, level, series, block, grid, logger, lock) -> None:
    """Download one block, write it, and record completion in the manifest."""
    times = settings.times
    t0 = block * settings.chunks["time"]
    t1 = min(t0 + settings.chunks["time"], len(times))
    days = pd.date_range(times[t0], periods=(t1 - t0) // 24, freq="D")
    urls = [source_url(settings, var, level, day) for day in days]

    for attempt in range(1, RETRIES + 1):
        try:
            data = fetch_days(fs, urls, days, grid.rows, grid.cols)
            break
        except (OSError, aiohttp.ClientError) as err:
            if attempt == RETRIES:
                raise
            logger.warning("%s block %d attempt %d failed (%s), retrying", series, block, attempt, err)
            time.sleep(2 ** attempt)
    write_block(settings, var, level, t0, t1, data[:, ::-1, :])

    record = {"series": series, "block": block, "start": str(times[t0]), "end": str(times[t1 - 1]),
              "files": len(urls)}
    with lock, open(settings.manifest_path, "a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("%s block %2d (%s to %s): %d files written",
                series, block, days[0].date(), days[-1].date(), len(urls))


def run(settings: utils.Settings) -> bool:
    """Download all missing blocks. Returns True if every block succeeded."""
    logger = utils.setup_logging(settings.log_dir, "download")
    fs, _ = fsspec.core.url_to_fs(settings.source_url)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    done = finished_blocks(settings)
    n_blocks = math.ceil(len(settings.times) / settings.chunks["time"])
    tasks = []
    for var in settings.vars:
        first_level = var.levels[0] if var.levels else None
        grid = source_grid(fs, source_url(settings, var, first_level, settings.start), settings.bbox)
        if not settings.store_path(var).exists():
            create_store(settings, var, grid)
            logger.info("Created %s with grid %d x %d", settings.store_path(var), len(grid.lat), len(grid.lon))
        for series, level in var.series():
            tasks += [(var, level, series, b, grid) for b in range(n_blocks) if (series, b) not in done]
    logger.info("%s to %s: %d blocks to download, %d already done",
                settings.start, settings.end, len(tasks), len(done))

    lock = threading.Lock()
    failed = []
    with ThreadPoolExecutor(settings.max_workers) as pool:
        futures = {pool.submit(run_block, fs, settings, *task, logger, lock): task for task in tasks}
        for future in as_completed(futures):
            _, _, series, block, _ = futures[future]
            try:
                future.result()
            except Exception as err:
                failed.append(f"{series} block {block}")
                logger.error("%s block %d failed: %s", series, block, err)
    if failed:
        logger.error("%d blocks failed (%s). Run the same command again to retry them.",
                     len(failed), ", ".join(sorted(failed)))
        return False

    logger.info("All blocks written to %s", settings.data_dir)
    return True


def main():
    ok = run(parse_args())
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
