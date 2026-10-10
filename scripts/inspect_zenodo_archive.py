#!/usr/bin/env python3
"""Inspect Zenodo's giant ZIP central directory without downloading trajectories.

Requires: pip install requests remotezip
Writes the archive member inventory, not the member contents.
"""
import argparse
import csv
import json
from pathlib import Path

import requests
from remotezip import RemoteZip

ARCHIVE_URL = "https://zenodo.org/records/21919704/files/alchemical_tfep.zip?download=1"


def require_range_support(url):
    # Important: a normal GET would attempt to transfer ~49 GB.
    # stream=True and closing before reading the body prevents that fallback.
    with requests.get(
        url, headers={"Range": "bytes=-32"}, stream=True, timeout=90
    ) as response:
        if response.status_code != 206 or "Content-Range" not in response.headers:
            raise RuntimeError(
                "Zenodo did not honor a byte-range request; refusing full ZIP download. "
                f"HTTP {response.status_code}; Content-Range="
                f"{response.headers.get('Content-Range')!r}"
            )
        print("Range supported:", response.headers["Content-Range"])
        print("Final archive URL:", response.url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=ARCHIVE_URL)
    parser.add_argument("--output-dir", default="archive-metadata")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    require_range_support(args.url)

    # RemoteZip reads the tail/central directory via HTTP byte ranges.
    # No trajectory/energy-file payloads are requested.
    with RemoteZip(args.url, initial_buffer_size=512 * 1024, timeout=120) as archive:
        members = archive.infolist()

    manifest = output / "zenodo-members.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["name", "bytes_uncompressed", "bytes_compressed", "crc32"])
        for member in members:
            writer.writerow(
                [member.filename, member.file_size, member.compress_size,
                 f"{member.CRC:08x}"]
            )

    # Select exactly the target ligand, without pulling all other ligands' large
    # preprocess.npz files into the candidate list.
    selected = [m for m in members if "p38a_2g" in m.filename.lower()]
    target_by_suffix = {}
    for member in selected:
        suffix = Path(member.filename).suffix.lower() or "(directory)"
        target_by_suffix[suffix] = target_by_suffix.get(suffix, 0) + 1
    summary = {
        "source": args.url,
        "dataset_doi": "10.5281/zenodo.21919704",
        "record_zip_md5": "ee24b58a3a5134f8c33411a59ad73da1",
        "archive_members": len(members),
        "total_uncompressed_bytes": sum(m.file_size for m in members),
        "selected_matching_members": len(selected),
        "target_suffix_counts": target_by_suffix,
        "selected_compressed_bytes": sum(m.compress_size for m in selected),
        "selected_members": [
            {"name": m.filename, "bytes": m.file_size, "compressed_bytes": m.compress_size}
            for m in selected
        ],
    }
    (output / "zenodo-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Indexed {len(members)} archive entries; {len(selected)} p38a_2g matches.")
    print("p38a_2g file extensions:", target_by_suffix)
    print(f"Inventory: {manifest}")
    print("p38a_2g candidate members (first 120):")
    for member in selected[:120]:
        print(f"  {member.compress_size:>12} B compressed  {member.filename}")


if __name__ == "__main__":
    main()
