"""Shared helpers for download.py
    1. Settings: variables and run settings
    2. Logging: console/file logging
    3. NetCDF3 byte-range reading: locate and decode a bounding box inside a remote file
"""

import logging
import struct
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# 1. Settings
# =============================================================================

@dataclass
class Variable:
    """One ERA5 variable to download.

    name:        variable name in the output Zarr store, e.g. "t2m"
    level_type:  "single_level" or "pressure_level"
    source_name: variable folder name in the ARCO-ERA5 raw 
    levels:      pressure levels in hPa (for pressure_level fields only)
    """

    name: str
    level_type: str
    source_name: str
    levels: list[int] = field(default_factory=list)

    def series(self) -> list[tuple[str, int | None]]:
        """Return data pairs (series name, level)"""
        if self.level_type == "single_level":
            return [(self.name, None)]
        return [(f"{self.name}{level}", level) for level in self.levels]


@dataclass
class Settings:
    """Run settings for one download."""

    source_url: str
    start: date
    end: date  # inclusive
    bbox: dict[str, float]  # lat_min, lat_max, lon_min, lon_max; longitude in -180..180
    vars: list[Variable]
    chunks: dict[str, int]  # Zarr chunk sizes for time, lat, lon
    max_workers: int
    data_dir: Path
    log_dir: Path

    @property
    def times(self) -> pd.DatetimeIndex:
        """Hourly time axis from start 00 UTC to end 23 UTC."""
        last = pd.Timestamp(self.end) + pd.Timedelta(hours=23)
        return pd.date_range(pd.Timestamp(self.start), last, freq="h")

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.jsonl"

    def store_path(self, var: Variable) -> Path:
        return self.data_dir / var.level_type / f"{var.name}.zarr"


# =============================================================================
# 2. Logging
# =============================================================================

def setup_logging(log_dir: Path, name: str) -> logging.Logger:
    """Log INFO messages to the console and to log_dir/<name>_<timestamp>.log."""
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    for handler in (logging.StreamHandler(), logging.FileHandler(log_dir / f"{name}_{stamp}.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# =============================================================================
# 3. NetCDF3 byte-range reading
#
# The ARCO-ERA5 raw files are NetCDF3 (64-bit offset), uncompressed, with one
# (time, latitude, longitude) variable stored contiguously. Parsing the header
# gives the byte offset of that variable, so any block of rows can be fetched
# with an HTTP range request instead of downloading the whole file.
# Format ref: https://docs.unidata.ucar.edu/nug/current/file_format_specifications.html
# =============================================================================

HEADER_BYTES = 65536  # first read per file; holds the header and the coordinate values

_NC_DTYPES = {1: ">i1", 2: "S1", 3: ">i2", 4: ">i4", 5: ">f4", 6: ">f8"}


@dataclass
class Nc3Layout:
    """Location and packing of the (time, latitude, longitude) variable in a NetCDF3 file."""

    var_name: str
    offset: int  # byte offset of the first data value
    shape: tuple[int, int, int]  # (time, latitude, longitude)
    dtype: np.dtype  # on-disk dtype, big-endian
    scale_factor: float
    add_offset: float
    fill_value: float | None
    attrs: dict  # attributes of the data variable (units, long_name, ...)
    time: pd.DatetimeIndex
    lat: np.ndarray
    lon: np.ndarray


class _HeaderReader:
    """Sequential reader over NetCDF3 header bytes."""

    def __init__(self, buf: bytes):
        self.buf = buf
        self.version = buf[3]  # 1 = classic, 2 = 64-bit offset
        self.pos = 4

    def int32(self) -> int:
        value = struct.unpack_from(">i", self.buf, self.pos)[0]
        self.pos += 4
        return value

    def offset(self) -> int:
        fmt, size = (">q", 8) if self.version == 2 else (">i", 4)
        value = struct.unpack_from(fmt, self.buf, self.pos)[0]
        self.pos += size
        return value

    def values(self, nc_type: int, count: int):
        dtype = np.dtype(_NC_DTYPES[nc_type])
        nbytes = count * dtype.itemsize
        raw = self.buf[self.pos:self.pos + nbytes]
        self.pos += -(-nbytes // 4) * 4  # entries are padded to 4 bytes
        if nc_type == 2:
            return raw.decode("utf-8", errors="replace")
        return np.frombuffer(raw, dtype)

    def name(self) -> str:
        return self.values(2, self.int32())

    def attributes(self) -> dict:
        self.int32()  # NC_ATTRIBUTE tag, or 0 when absent
        attrs = {}
        for _ in range(self.int32()):
            key = self.name()
            nc_type = self.int32()
            attrs[key] = self.values(nc_type, self.int32())
        return attrs


def read_nc3_layout(buf: bytes) -> Nc3Layout:
    """Parse a NetCDF3 header and locate its (time, latitude, longitude) variable.

    buf holds the first HEADER_BYTES of the file. The coordinate values are read
    from buf too; in the ARCO-ERA5 files the header and coordinates take about 10 KB.
    """
    if buf[:3] != b"CDF" or buf[3] not in (1, 2):
        raise ValueError("not a NetCDF3 classic or 64-bit-offset file")
    reader = _HeaderReader(buf)
    reader.int32()  # number of records
    reader.int32()  # NC_DIMENSION tag
    dims = []
    for _ in range(reader.int32()):
        dims.append((reader.name(), reader.int32()))
    reader.attributes()  # global attributes
    reader.int32()  # NC_VARIABLE tag
    variables = {}
    for _ in range(reader.int32()):
        name = reader.name()
        dim_ids = [reader.int32() for _ in range(reader.int32())]
        attrs = reader.attributes()
        dtype = np.dtype(_NC_DTYPES[reader.int32()])
        reader.int32()  # vsize
        variables[name] = {
            "dims": [dims[i] for i in dim_ids],
            "dtype": dtype,
            "offset": reader.offset(),
            "attrs": attrs,
        }

    def coord(name):
        var = variables[name]
        values = np.frombuffer(buf, var["dtype"], count=var["dims"][0][1], offset=var["offset"])
        return values.astype(var["dtype"].newbyteorder("="))  # native byte order for pandas/xarray

    var_name = next((n for n, v in variables.items()
                     if [d[0] for d in v["dims"]] == ["time", "latitude", "longitude"]), None)
    if var_name is None:
        raise ValueError("no variable with dimensions (time, latitude, longitude)")
    var = variables[var_name]
    shape = tuple(length for _, length in var["dims"])
    # A record (unlimited) dimension has length 0 here and interleaves the data,
    # so the offset arithmetic below would be wrong.
    if 0 in shape:
        raise ValueError(f"{var_name} uses a record dimension, which is not supported")

    attrs = var["attrs"]
    fill = attrs.get("_FillValue", attrs.get("missing_value"))
    time_var = xr.Dataset({"time": ("time", coord("time"), {"units": variables["time"]["attrs"]["units"]})})
    
    return Nc3Layout(
        var_name=var_name,
        offset=var["offset"],
        shape=shape,
        dtype=var["dtype"],
        scale_factor=float(attrs["scale_factor"][0]) if "scale_factor" in attrs else 1.0,
        add_offset=float(attrs["add_offset"][0]) if "add_offset" in attrs else 0.0,
        fill_value=None if fill is None else float(fill[0]),
        attrs={k: v for k, v in attrs.items() if isinstance(v, str)},
        time=pd.DatetimeIndex(xr.decode_cf(time_var)["time"].values),
        lat=coord("latitude"),
        lon=coord("longitude"),
    )


def bbox_rows_cols(lat: np.ndarray, lon: np.ndarray, bbox: dict) -> tuple[slice, slice]:
    """Row and column slices of the source grid that fall inside the bounding box.

    The source grid has latitude descending and longitude in 0..360, while bbox
    longitudes are in -180..180. The box must not cross the 0 degree meridian.
    """
    rows = np.flatnonzero((lat >= bbox["lat_min"]) & (lat <= bbox["lat_max"]))
    lon180 = (lon + 180) % 360 - 180
    cols = np.flatnonzero((lon180 >= bbox["lon_min"]) & (lon180 <= bbox["lon_max"]))
    if not len(rows) or not len(cols):
        raise ValueError("the bounding box contains no source grid points")
    if not np.all(np.diff(cols) == 1):
        raise ValueError("the bounding box must not cross the 0 degree meridian")
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(cols[0]), int(cols[-1]) + 1)


def bbox_byte_ranges(layout: Nc3Layout, rows: slice) -> tuple[list[int], list[int]]:
    """Byte ranges (starts, ends) that hold the selected rows, all longitudes, for every hour.

    The variable is stored hour by hour, so for each hour the selected latitude
    rows form one contiguous run of bytes.
    """
    n_time, n_lat, n_lon = layout.shape
    row_bytes = n_lon * layout.dtype.itemsize
    starts = [layout.offset + (t * n_lat + rows.start) * row_bytes for t in range(n_time)]
    ends = [start + (rows.stop - rows.start) * row_bytes for start in starts]
    return starts, ends


def decode_rows(blobs: list[bytes], layout: Nc3Layout, rows: slice, cols: slice) -> np.ndarray:
    """Unpack fetched row blocks into a float32 array (time, rows, cols); fill values become NaN."""
    n_rows = rows.stop - rows.start
    n_lon = layout.shape[2]
    packed = np.stack([np.frombuffer(b, layout.dtype).reshape(n_rows, n_lon)[:, cols] for b in blobs])
    values = (packed * layout.scale_factor + layout.add_offset).astype(np.float32)
    if layout.fill_value is not None:
        values[packed == layout.fill_value] = np.nan
    return values
