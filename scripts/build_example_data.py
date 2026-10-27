"""Rebuild the quickstart's example-data release asset from JAXScape's v0.0.6 data.zip.

The asset holds Mountain Pygmy-possum (Burramys parvus) pairwise Fst between 12
sampling units, kindly provided by Cesar Australia, and ESA WorldCover 2021 land
cover (CC BY 4.0). Run once with the examples extra, check the output by hand,
and attach it to the ``example-data-v1`` pre-release:

    uv run --extra examples python scripts/build_example_data.py

The archive contains, under ``mountain-pygmy-possum/``:

- ``landcover.tif``: uint8 ESA WorldCover classes in EPSG:7855 at the native
  8.36 m resolution, deflate-compressed;
- ``sites.csv``: site name, then x and y in the raster's CRS;
- ``fst.csv``: the site-labeled 12x12 pairwise Fst matrix;
- ``README.md``: provenance and attribution.
"""

import argparse
import sqlite3
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray
from pyproj import Transformer

SOURCE_URL = "https://github.com/vboussange/jaxscape/releases/download/v0.0.6/data.zip"
ROOT = Path(__file__).resolve().parents[1]
NAME = "mountain-pygmy-possum"
WORLDCOVER_CLASSES = {10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100}
README = """\
# Mountain Pygmy-possum example data

Pairwise Fst between 12 Mountain Pygmy-possum (*Burramys parvus*) sampling units
in the Victorian Alps, kindly provided by [Cesar Australia](https://cesaraustralia.com),
with ESA WorldCover 2021 land cover.

- `landcover.tif`: ESA WorldCover classes (uint8; 0 is no data) in EPSG:7855
  (GDA2020 / MGA zone 55) at 8.36 m resolution. © ESA WorldCover project 2021 /
  Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover
  consortium. Licensed under CC BY 4.0: https://creativecommons.org/licenses/by/4.0/
- `sites.csv`: sampling-unit names and their x and y coordinates in EPSG:7855.
- `fst.csv`: pairwise Fst between the sampling units, labeled by site name.
  Genetic data courtesy of Cesar Australia.

Built by `scripts/build_example_data.py` in https://github.com/vboussange/ilg-toolkit
from https://github.com/vboussange/jaxscape/releases/download/v0.0.6/data.zip.
"""


def read_sites(path: Path) -> pd.DataFrame:
    """Site names with longitude and latitude, in the GeoPackage's site order."""
    with sqlite3.connect(path) as connection:
        ((srs,),) = connection.execute(
            "SELECT srs_id FROM gpkg_geometry_columns WHERE table_name = 'site_metadata'"
        ).fetchall()
        if srs != 4326:
            raise ValueError(f"Expected WGS 84 site coordinates, found SRS {srs}")
        return pd.read_sql_query(
            "SELECT site_id, site_name, longitude, latitude FROM site_metadata ORDER BY site_id",
            connection,
        )


def build(source_dir: Path, output: Path) -> None:
    """Write the release asset from the extracted JAXScape ``data/cesar`` directory."""
    landcover = rioxarray.open_rasterio(source_dir / "landcover_7855.tif")
    if landcover.rio.crs.to_epsg() != 7855:
        raise ValueError(f"Expected EPSG:7855 land cover, found {landcover.rio.crs}")
    classes = set(np.unique(landcover.values).astype(int).tolist())
    if not classes <= WORLDCOVER_CLASSES | {0}:
        raise ValueError(f"Unexpected land-cover classes {sorted(classes - WORLDCOVER_CLASSES)}")
    sites = read_sites(source_dir / "site_metadata.gpkg")
    fst = np.load(source_dir / "target_distances.npy")
    if fst.shape != (len(sites), len(sites)) or not np.allclose(fst, fst.T):
        raise ValueError("Fst must be a symmetric matrix aligned with the sites")
    if (np.diag(fst) != 0).any() or (fst < 0).any() or (fst >= 1).any():
        raise ValueError("Fst must lie in [0, 1) with a zero diagonal")
    x, y = Transformer.from_crs(4326, 7855, always_xy=True).transform(
        sites["longitude"].to_numpy(), sites["latitude"].to_numpy()
    )
    left, bottom, right, top = landcover.rio.bounds()
    if not ((left < x) & (x < right) & (bottom < y) & (y < top)).all():
        raise ValueError("Every site must fall inside the land-cover raster")
    names = sites["site_name"].tolist()

    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary) / NAME
        staging.mkdir()
        landcover = landcover.astype(np.uint8).rio.write_nodata(0, encoded=False)
        landcover.attrs = {
            key: landcover.attrs[key]
            for key in ("copyright", "license", "reference", "title", "product_version")
        }
        landcover.rio.to_raster(staging / "landcover.tif", compress="deflate", tiled=True)
        pd.DataFrame({"site": names, "x": x, "y": y}).to_csv(
            staging / "sites.csv", index=False, float_format="%.3f"
        )
        pd.DataFrame(fst, index=pd.Index(names, name="site"), columns=names).to_csv(
            staging / "fst.csv"
        )
        (staging / "README.md").write_text(README, encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.iterdir()):
                archive.write(path, f"{NAME}/{path.name}")
    print(f"Wrote {output} ({output.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    """Download the JAXScape source archive and build the release asset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build" / f"{NAME}.zip")
    options = parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "data.zip"
        urllib.request.urlretrieve(SOURCE_URL, archive)
        with zipfile.ZipFile(archive) as source:
            source.extractall(temporary)
        build(Path(temporary) / "data" / "cesar", options.output)


if __name__ == "__main__":
    main()
