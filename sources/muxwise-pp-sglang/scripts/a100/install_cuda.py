#!/usr/bin/env python3
"""Install a pinned CUDA 12.8 compiler subset into a user-owned directory.

No system packages, drivers, global symlinks, or Python dependencies are changed.
NVIDIA archives are verified against pinned SHA-256 values and their manifest.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time
import urllib.request


RELEASE = "12.8.1"
BASE_URL = "https://developer.download.nvidia.com/compute/cuda/redist/"
MANIFEST_URL = BASE_URL + "redistrib_" + RELEASE + ".json"
COMPONENTS = {
    "cuda_nvcc": "9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3",
    "cuda_cudart": "8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68",
    "cuda_cccl": "0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url, path, expected_size, expected_hash):
    if path.is_file() and path.stat().st_size == expected_size:
        if sha256(path) == expected_hash:
            return
    partial = path.with_name(path.name + ".partial")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                with partial.open("wb") as destination:
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        destination.write(block)
            if partial.stat().st_size != expected_size:
                raise RuntimeError("Unexpected archive size: " + str(partial))
            if sha256(partial) != expected_hash:
                raise RuntimeError("Archive SHA-256 mismatch: " + str(partial))
            partial.replace(path)
            return
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def extract_archive(archive, prefix):
    # Each NVIDIA component has one enclosing directory. Validate every member
    # and link before extracting its contents directly into the local prefix.
    with tarfile.open(archive, "r:xz") as source:
        for member in source.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts:
                raise RuntimeError("Unsafe archive member: " + member.name)
            stripped = Path(*path.parts[1:])
            if member.issym():
                target = (prefix / stripped.parent / member.linkname).resolve()
                if not target.is_relative_to(prefix.resolve()):
                    raise RuntimeError("Unsafe archive symlink: " + member.name)
            elif member.islnk():
                link = PurePosixPath(member.linkname)
                if link.is_absolute() or ".." in link.parts:
                    raise RuntimeError("Unsafe archive hardlink: " + member.name)
            elif not (member.isfile() or member.isdir()):
                raise RuntimeError("Unsupported archive member: " + member.name)
    subprocess.run(
        ["tar", "-xJf", str(archive), "--strip-components=1", "-C", str(prefix)],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path,
                        default=Path.home() / ".local" / "cuda-12.8")
    parser.add_argument("--provenance", type=Path,
                        default=Path(__file__).with_name("toolkit-provenance.json"))
    args = parser.parse_args()
    prefix = args.prefix.expanduser().resolve()
    if str(prefix).startswith(("/usr/", "/opt/", "/etc/")):
        raise RuntimeError("Choose a user-owned prefix outside system directories")
    prefix.mkdir(parents=True, exist_ok=True)
    archives = prefix / ".archives"
    archives.mkdir(exist_ok=True)
    with urllib.request.urlopen(MANIFEST_URL, timeout=60) as response:
        manifest_bytes = response.read()
    manifest = json.loads(manifest_bytes)
    if manifest["release_label"] != RELEASE:
        raise RuntimeError("Unexpected CUDA release in manifest")
    records = []
    for component, expected_hash in COMPONENTS.items():
        metadata = manifest[component]
        platform = metadata["linux-x86_64"]
        if platform["sha256"] != expected_hash:
            raise RuntimeError("Manifest disagrees with pinned hash: " + component)
        url = BASE_URL + platform["relative_path"]
        archive = archives / Path(platform["relative_path"]).name
        print("Downloading and verifying " + component, flush=True)
        download(url, archive, int(platform["size"]), expected_hash)
        extract_archive(archive, prefix)
        records.append({"component": component, "version": metadata["version"],
                        "url": url, "bytes": archive.stat().st_size,
                        "sha256": expected_hash})
    # Redistribution archives use lib; FlashInfer's Ninja generator uses lib64.
    if not (prefix / "lib64").exists() and (prefix / "lib").is_dir():
        (prefix / "lib64").symlink_to("lib", target_is_directory=True)
    required = ["bin/nvcc", "include/cuda_runtime.h", "include/cub/cub.cuh",
                "nvvm/libdevice/libdevice.10.bc", "lib64/libcudart.so"]
    for relative in required:
        if not (prefix / relative).exists():
            raise RuntimeError("Missing toolkit component: " + relative)
    nvcc_version = subprocess.check_output(
        [str(prefix / "bin/nvcc"), "--version"], text=True
    )
    if "release 12.8," not in nvcc_version:
        raise RuntimeError("Installed nvcc is not CUDA 12.8")
    provenance = {
        "release": RELEASE,
        "prefix": str(prefix),
        "manifest_url": MANIFEST_URL,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "installed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "components": records,
        "nvcc_version": nvcc_version,
        "system_cuda_modified": False,
        "python_dependencies_modified": False,
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(nvcc_version, end="")
    print("Installed at " + str(prefix), flush=True)
    print("Provenance: " + str(args.provenance), flush=True)


if __name__ == "__main__":
    main()
