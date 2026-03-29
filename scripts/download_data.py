#!/usr/bin/env python3
"""Download all datasets required for TSNN 3-stage training.

Usage:
    python scripts/download_data.py --all              # Download everything
    python scripts/download_data.py --stage-a          # MDD + MISATO
    python scripts/download_data.py --stage-b          # DD-13M
    python scripts/download_data.py --stage-c          # BindingDB + KOFFI
    python scripts/download_data.py --mdd              # MDD only (24.5 GB, fastest start)
    python scripts/download_data.py --misato           # MISATO only (193 GB)
    python scripts/download_data.py --dd13m            # DD-13M only (204 GB)
    python scripts/download_data.py --bindingdb        # BindingDB koff (~525 MB)
    python scripts/download_data.py --koffi            # KOFFI (~small, API)
    python scripts/download_data.py --data-dir /path   # Custom data directory
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve, Request, urlopen

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "raw"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _download_file(url: str, dest: Path, desc: str = ""):
    """Download a file with progress reporting."""
    if dest.exists():
        logger.info(f"Already exists: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    logger.info(f"Downloading {desc or dest.name} ...")
    logger.info(f"  URL: {url}")
    logger.info(f"  Dest: {dest}")

    try:
        # Try wget first (better for large files, shows progress)
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(tmp), url],
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Fall back to urllib
        def _progress(block, block_size, total):
            done = block * block_size
            if total > 0:
                pct = min(100, done * 100 // total)
                mb = done / 1e6
                print(f"\r  {pct}% ({mb:.1f} MB)", end="", flush=True)

        urlretrieve(url, str(tmp), reporthook=_progress)
        print()

    tmp.rename(dest)
    logger.info(f"  Done: {dest} ({dest.stat().st_size / 1e9:.2f} GB)")


def _extract_tar(archive: Path, dest_dir: Path):
    """Extract a tar.gz archive."""
    if dest_dir.exists() and any(dest_dir.iterdir()):
        logger.info(f"Already extracted: {dest_dir}")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {archive.name} → {dest_dir} ...")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=dest_dir)
    logger.info("  Extraction complete.")


def _extract_zip(archive: Path, dest_dir: Path):
    """Extract a zip archive."""
    if dest_dir.exists() and any(dest_dir.iterdir()):
        logger.info(f"Already extracted: {dest_dir}")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Extracting {archive.name} → {dest_dir} ...")
    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(path=dest_dir)
    logger.info("  Extraction complete.")


# ─────────────────────────────────────────────────────────────
# Dataset downloaders
# ─────────────────────────────────────────────────────────────

def download_mdd(data_dir: Path):
    """Download MDD: 862 complexes, 200ns MD each (24.5 GB).

    Source: Zenodo 11172815
    Format: GROMACS (XTC trajectories, GRO structures, TPR topology)
    """
    mdd_dir = data_dir / "mdd"
    archive = mdd_dir / "MD_dataset.tar.gz"

    _download_file(
        url="https://zenodo.org/records/11172815/files/MD_dataset.tar.gz?download=1",
        dest=archive,
        desc="MDD dataset (24.5 GB)",
    )
    _extract_tar(archive, mdd_dir / "extracted")

    logger.info("MDD download complete.")
    logger.info(f"  Location: {mdd_dir}")
    logger.info("  Next: run `python scripts/preprocess_data.py --mdd`")


def download_misato(data_dir: Path):
    """Download MISATO: ~17K complexes, 10ns MD each.

    Source: Zenodo 7711953
    Format: HDF5 (ML-ready)
    Files: MD.hdf5 (133 GB), train/val/test splits
    """
    misato_dir = data_dir / "misato"
    misato_dir.mkdir(parents=True, exist_ok=True)

    base = "https://zenodo.org/records/7711953/files"

    # MD trajectories (main file, 133 GB)
    _download_file(
        url=f"{base}/MD.hdf5?download=1",
        dest=misato_dir / "MD.hdf5",
        desc="MISATO MD trajectories (133 GB)",
    )

    # QM properties (343 MB)
    _download_file(
        url=f"{base}/QM.hdf5?download=1",
        dest=misato_dir / "QM.hdf5",
        desc="MISATO QM properties (343 MB)",
    )

    # Train/val/test splits
    for split in ["train_MD.txt", "val_MD.txt", "test_MD.txt"]:
        _download_file(
            url=f"{base}/{split}?download=1",
            dest=misato_dir / split,
            desc=f"MISATO {split}",
        )

    logger.info("MISATO download complete.")
    logger.info(f"  Location: {misato_dir}")
    logger.info("  Next: run `python scripts/preprocess_data.py --misato`")


def download_dd13m(data_dir: Path):
    """Download DD-13M: 26,612 dissociation trajectories, 565 complexes.

    Source: HuggingFace SZBL-IDEA/MD
    Format: H5MD (HDF5-based) + PDB
    Size: ~204 GB
    """
    dd13m_dir = data_dir / "dd13m"
    dd13m_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DD-13M download: 204 GB from HuggingFace")
    logger.info("  Using huggingface-cli if available, else git lfs clone")

    # Try huggingface-cli first
    try:
        subprocess.run(["huggingface-cli", "--version"], capture_output=True, check=True)
        logger.info("  Using huggingface-cli download ...")
        subprocess.run(
            [
                "huggingface-cli", "download",
                "SZBL-IDEA/MD",
                "--repo-type", "dataset",
                "--local-dir", str(dd13m_dir),
            ],
            check=True,
        )
        logger.info("DD-13M download complete via huggingface-cli.")
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Try pip installing huggingface_hub, then use Python API
    try:
        from huggingface_hub import snapshot_download
        logger.info("  Using huggingface_hub.snapshot_download ...")
        snapshot_download(
            repo_id="SZBL-IDEA/MD",
            repo_type="dataset",
            local_dir=str(dd13m_dir),
        )
        logger.info("DD-13M download complete via huggingface_hub.")
        return
    except ImportError:
        pass

    # Fall back to git lfs clone
    logger.info("  Falling back to git lfs clone ...")
    subprocess.run(
        [
            "git", "lfs", "clone",
            "https://huggingface.co/datasets/SZBL-IDEA/MD",
            str(dd13m_dir),
        ],
        check=True,
    )
    logger.info("DD-13M download complete.")
    logger.info(f"  Location: {dd13m_dir}")
    logger.info("  Next: run `python scripts/preprocess_data.py --dd13m`")


def download_bindingdb(data_dir: Path):
    """Download BindingDB full TSV dump and extract koff records.

    Source: bindingdb.org
    Format: TSV (zipped, ~525 MB)
    """
    bdb_dir = data_dir / "bindingdb"
    bdb_dir.mkdir(parents=True, exist_ok=True)

    # The URL pattern uses YYYYMM format; we use a recent one
    # If this specific month isn't available, try the download page
    archive = bdb_dir / "BindingDB_All.tsv.zip"

    _download_file(
        url="https://www.bindingdb.org/bind/downloads/BindingDB_All_2025m1_tsv.zip",
        dest=archive,
        desc="BindingDB full TSV dump (~525 MB)",
    )
    _extract_zip(archive, bdb_dir)

    logger.info("BindingDB download complete.")
    logger.info(f"  Location: {bdb_dir}")
    logger.info("  Next: run `python scripts/preprocess_data.py --bindingdb`")


def download_koffi(data_dir: Path):
    """Download KOFFI kinetics database via REST API.

    Source: koffidb.org
    Format: JSON → CSV (1,705+ interactions)
    Rate limit: 1 req / 3 sec, 100 req / day
    """
    koffi_dir = data_dir / "koffi"
    koffi_dir.mkdir(parents=True, exist_ok=True)
    output_file = koffi_dir / "koffi_all.json"

    if output_file.exists():
        logger.info(f"Already exists: {output_file}")
        return

    logger.info("Downloading KOFFI database via API ...")
    logger.info("  Rate limited: 1 req / 3 sec, 100 req / day")

    all_records = []
    page = 1
    page_size = 50  # Max allowed

    while True:
        url = f"http://koffidb.org/api/interactions/?page={page}&page_size={page_size}"
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())

            results = data.get("results", [])
            if not results:
                break

            all_records.extend(results)
            total = data.get("count", "?")
            logger.info(f"  Page {page}: got {len(results)} records "
                       f"(total so far: {len(all_records)}/{total})")

            if not data.get("next"):
                break

            page += 1
            time.sleep(3.5)  # Respect rate limit

        except Exception as e:
            logger.warning(f"  API error on page {page}: {e}")
            if page == 1:
                logger.error("  Cannot reach KOFFI API. Skipping.")
                return
            break

    with open(output_file, "w") as f:
        json.dump(all_records, f, indent=2)

    logger.info(f"KOFFI download complete: {len(all_records)} records")
    logger.info(f"  Location: {output_file}")
    logger.info("  Next: run `python scripts/preprocess_data.py --koffi`")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for TSNN 3-stage training"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Root directory for raw data")
    parser.add_argument("--all", action="store_true",
                        help="Download all datasets")
    parser.add_argument("--stage-a", action="store_true",
                        help="Download Stage A datasets (MDD + MISATO)")
    parser.add_argument("--stage-b", action="store_true",
                        help="Download Stage B datasets (DD-13M)")
    parser.add_argument("--stage-c", action="store_true",
                        help="Download Stage C datasets (BindingDB + KOFFI)")
    parser.add_argument("--mdd", action="store_true")
    parser.add_argument("--misato", action="store_true")
    parser.add_argument("--dd13m", action="store_true")
    parser.add_argument("--bindingdb", action="store_true")
    parser.add_argument("--koffi", action="store_true")

    args = parser.parse_args()
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # If no specific flag, show help
    any_selected = (args.all or args.stage_a or args.stage_b or args.stage_c
                    or args.mdd or args.misato or args.dd13m
                    or args.bindingdb or args.koffi)
    if not any_selected:
        parser.print_help()
        print("\n\nRecommended start:")
        print("  python scripts/download_data.py --mdd          # 24.5 GB, fastest")
        print("  python scripts/download_data.py --stage-c      # koff labels")
        print("  python scripts/download_data.py --dd13m        # 204 GB, Stage B")
        print("  python scripts/download_data.py --misato       # 193 GB, Stage A scale")
        return

    logger.info(f"Data directory: {data_dir}")

    # Stage A
    if args.all or args.stage_a or args.mdd:
        download_mdd(data_dir)
    if args.all or args.stage_a or args.misato:
        download_misato(data_dir)

    # Stage B
    if args.all or args.stage_b or args.dd13m:
        download_dd13m(data_dir)

    # Stage C
    if args.all or args.stage_c or args.bindingdb:
        download_bindingdb(data_dir)
    if args.all or args.stage_c or args.koffi:
        download_koffi(data_dir)

    logger.info("=" * 60)
    logger.info("Download complete. Next step:")
    logger.info("  python scripts/preprocess_data.py --help")


if __name__ == "__main__":
    main()
