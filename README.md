# Overview

A simple workflow for downloading hourly ERA5 reanalysis from the public ARCO-ERA5 raw archive and storing data in compressed Zarr. The example selects 2 m temperature and temperature at 500/850 hPa over CONUS.

## Efficient Data Access and Storage

- **Selective reads:** Download only the required spatial subset to reduce data transfer.
- **Parallel downloads:** Process multiple data blocks concurrently to improve download efficiency. 
- **Chunked storage:** Enable efficient data access and reduce storage size by Zarr compression.

## Performance

30 days of `t2m` over CONUS (January 1–30, 2020; 720 hours, 105 × 237 grid):

| Approach | Download (s) | Write (s) | Stored (MB) |
|---|---:|---:|---:|
| daily NetCDF | 36.07 | 0.82 | 72.0 |
| Zarr | 2.40 | 0.19 | 42.1 |


## Quick start

From the repository root:

```bash
conda env create -f environment.yml
conda activate data-era5

# Edit the settings in slurm/download.sh as needed, then submit:
sbatch slurm/download.sh
```

## Output

Each variable has a Zarr store under `data/example/`. Missing values remain NaN.

