"""Explore Zarr Dataset

Cannot read the dataset directly into xarray, because xarray assumes
CF-compliant metadata and expects variables to be at the top level of the Zarr
group.

Our data is nested one level deeper, under a group like `radar/rtcor/c`.

The code below is a workaround to read the Zarr group as an xarray.Dataset,
written by chatGPT. Could still be improved, but it works for now.

"""

import xarray as xr
import zarr
from pathlib import Path
from typing import Optional


def read_zarr_group_as_xr(path: str, group: Optional[str] = None) -> xr.Dataset:
    """
    Load a Zarr group as an xarray.Dataset without assuming aligned dimensions.

    Parameters:
    - path: Path to Zarr root store.
    - group: Optional subgroup path inside the Zarr store.

    Returns:
    - xarray.Dataset with variables from the group.
    """
    zarr_path = Path(path)
    group_path = zarr_path / group if group else zarr_path
    store = zarr.open_group(str(group_path), mode="r")

    dataset = {}
    for name, array in store.arrays():
        # Use unique dim names per variable to avoid conflicts
        dims = tuple(f"{name}_dim_{i}" for i in range(array.ndim))
        data = xr.DataArray(array[:], dims=dims, name=name)
        dataset[name] = data

    return xr.Dataset(dataset)


if __name__ == "__main__":
    zarr_path = "/home/peter/weathergenerator/data/nowcasting/dataset.zarr"
    group = "radar"  # Specify the group you want to read, or None
    ds = read_zarr_group_as_xr(zarr_path, group)

    import IPython

    IPython.embed()  # Start an interactive session to explore the dataset
