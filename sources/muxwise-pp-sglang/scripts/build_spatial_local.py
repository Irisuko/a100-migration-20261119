#!/usr/bin/env python3
"""Add this fork's spatial extension to sgl-kernel in the active virtualenv.

Run with the environment's Python, for example:
    .venv/bin/python scripts/build_spatial_local.py --verbose

The spatial source only calls the CUDA driver API, so host C++ can compile it.
This avoids rebuilding the other sgl-kernel kernels or requiring nvcc to match
the CUDA version used by the installed PyTorch wheel. No GPU work is launched.
"""

import argparse
import importlib.util
import os
from pathlib import Path
import re
import shutil
import sys
import sysconfig


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda-root",
        type=Path,
        default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")),
        help="CUDA toolkit with headers for green contexts (CUDA >= 12.4)",
    )
    parser.add_argument(
        "--build-dir", type=Path, default=repo / "run" / "spatial-build"
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if sys.prefix == sys.base_prefix:
        parser.error("Run this script with the target virtualenv's Python.")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    python_header = Path(sysconfig.get_path("include")) / "Python.h"
    if not python_header.is_file():
        parser.error(f"Missing Python development header: {python_header}")

    cuda_include = args.cuda_root.resolve() / "include"
    for header in ("cuda.h", "cuda_runtime.h"):
        if not (cuda_include / header).is_file():
            parser.error(f"Missing CUDA header: {cuda_include / header}")
    version = re.search(
        r"^#define\s+CUDA_VERSION\s+(\d+)",
        (cuda_include / "cuda.h").read_text(),
        re.MULTILINE,
    )
    if version is None or int(version.group(1)) < 12040:
        parser.error("CUDA headers >= 12.4 are required for green contexts.")

    kernel_spec = importlib.util.find_spec("sgl_kernel")
    if kernel_spec is None or not kernel_spec.submodule_search_locations:
        parser.error("Install sgl-kernel in the target virtualenv first.")
    package_dir = Path(next(iter(kernel_spec.submodule_search_locations))).resolve()
    if not package_dir.is_relative_to(Path(sys.prefix).resolve()):
        parser.error(f"Refusing to change sgl-kernel outside this venv: {package_dir}")
    if importlib.util.find_spec("torch") is None:
        parser.error("Install PyTorch in the target virtualenv first.")

    import torch
    from torch.utils.cpp_extension import load

    sources = repo / "sgl-kernel"
    original = sources / "csrc" / "spatial" / "greenctx_stream.cu"
    binding = sources / "csrc" / "spatial_extension.cc"
    wrapper = sources / "python" / "sgl_kernel" / "spatial.py"
    for source in (original, binding, wrapper):
        if not source.is_file():
            parser.error(f"Missing repository source: {source}")

    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    host_source = build_dir / "greenctx_stream.cpp"
    shutil.copy2(original, host_source)
    os.environ["MAX_JOBS"] = str(args.jobs)
    # Invoking .venv/bin/python does not itself add .venv/bin to PATH.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    print(f"Python: {sys.executable}", flush=True)
    print(f"PyTorch: {torch.__version__}; CUDA headers: {cuda_include}", flush=True)
    print(f"Install destination: {package_dir}", flush=True)
    extension = load(
        name="spatial_ops",
        sources=[str(host_source), str(binding)],
        extra_include_paths=[
            str(sources / "include"),
            str(sources / "csrc" / "spatial"),
            str(cuda_include),
        ],
        extra_cflags=["-O2"],
        extra_ldflags=["-lcuda"],
        with_cuda=False,
        build_directory=str(build_dir),
        verbose=args.verbose,
    )
    if not hasattr(torch.ops.sgl_kernel, "create_greenctx_stream_by_value"):
        raise RuntimeError("The compiled spatial operator was not registered.")

    # The interpreter-specific suffix takes priority over any older abi3 binary.
    destination = package_dir / f"spatial_ops{sysconfig.get_config_var('EXT_SUFFIX')}"
    shutil.copy2(extension.__file__, destination)
    shutil.copy2(wrapper, package_dir / "spatial.py")
    print(f"Installed {destination}")
    print(f"Installed {package_dir / 'spatial.py'}")
    print("Build and operator import succeeded. GPU execution has not been tested.")


if __name__ == "__main__":
    main()
