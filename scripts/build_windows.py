"""Build the Windows x64 release ZIP from the verified Python distributions."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import urllib.request
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.13.17"
# Windows embeddable package (64-bit) at https://www.python.org/downloads/release/python-31317/
PYTHON_SHA256 = "9e35319bffedab8b36ad44d8f377cb4f1733bcec257a0b6d04d94f01486ddb38"
GITHUB_ASSET_LIMIT = 2 * 1024**3


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_windows_zip(wheel: Path, output: Path) -> None:
    if sysconfig.get_platform() != "win-amd64" or sys.version_info[:2] != (3, 13):
        raise SystemExit("Building this release requires 64-bit Windows and Python 3.13.")

    system_root = Path(os.environ["SystemRoot"])
    compiler = system_root / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    if not compiler.is_file():
        raise SystemExit("The .NET Framework x64 C# compiler is required to build LocalMind.exe.")

    with tempfile.TemporaryDirectory(prefix="lm-") as directory:
        staging = Path(directory)
        package = staging / "LocalMind"
        runtime = package / "runtime"
        archive = staging / "python.zip"
        url = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
        print(f"Downloading the CPython {PYTHON_VERSION} Windows runtime", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if sha256(archive) != PYTHON_SHA256:
            raise SystemExit("CPython download failed SHA-256 verification.")
        with ZipFile(archive) as python_zip:
            python_zip.extractall(runtime)

        site_packages = runtime / "Lib/site-packages"
        (runtime / "python313._pth").write_text(
            "python313.zip\n.\nLib\nLib/site-packages\nimport site\n", encoding="ascii"
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-compile", "--target", str(site_packages), str(wheel)],
            check=True,
        )
        shutil.copy2(ROOT / "LICENSE", package / "LICENSE")
        shutil.copy2(ROOT / "README.md", package / "README.md")
        shutil.copy2(ROOT / "scripts/smoke_windows.py", runtime / "smoke_windows.py")

        launcher = package / "LocalMind.exe"
        subprocess.run(
            [
                str(compiler), "/nologo", "/utf8output", "/target:winexe", "/platform:x64",
                "/reference:System.Windows.Forms.dll", f"/out:{launcher}",
                str(ROOT / "scripts/LocalMindLauncher.cs"),
            ],
            check=True,
        )
        smoke_env = os.environ.copy()
        for name in ("CONDA_PREFIX", "PYTHONHOME", "PYTHONPATH", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
            smoke_env.pop(name, None)
        smoke_env.update(
            PATH=os.pathsep.join([str(system_root / "System32"), str(system_root)]),
            LOCALAPPDATA=str(staging / "appdata"),
            QT_QPA_PLATFORM="offscreen",
            HF_HUB_OFFLINE="1",
            PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True",
        )
        print("Testing the packaged launcher and its runtime", flush=True)
        subprocess.run([str(launcher), "--smoke-test"], cwd=staging, env=smoke_env, check=True, timeout=180)

        # Use the destination filesystem so the completed ZIP can be moved atomically.
        with tempfile.TemporaryDirectory(prefix="lm-zip-", dir=output.parent) as zip_directory:
            temporary_zip = Path(zip_directory) / output.name
            with ZipFile(temporary_zip, "w", compression=ZIP_DEFLATED, compresslevel=3) as release_zip:
                for path in sorted(package.rglob("*")):
                    if not path.is_file():
                        continue
                    if path.is_relative_to(site_packages):
                        relative = path.relative_to(site_packages)
                        # pip generates console scripts and installation URLs for the build machine.
                        if relative.parts[0] in {"bin", "Scripts"} or path.name == "direct_url.json":
                            continue
                    release_zip.write(path, path.relative_to(staging))
            if temporary_zip.stat().st_size >= GITHUB_ASSET_LIMIT:
                raise SystemExit("The Windows ZIP exceeds the GitHub Release asset size limit.")
            temporary_zip.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", type=Path, help="Directory containing the wheel and source distribution")
    args = parser.parse_args()
    dist = args.dist.resolve()
    with (ROOT / "pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    wheel = dist / f"localmind_demo-{version}-py3-none-any.whl"
    source = dist / f"localmind_demo-{version}.tar.gz"
    output = dist / f"LocalMind-{version}-windows-x64.zip"
    if not wheel.is_file() or not source.is_file():
        parser.error(f"Run python -m build --outdir {dist} before packaging Windows.")

    build_windows_zip(wheel, output)
    (dist / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in (output, source)),
        encoding="ascii",
        newline="\n",
    )
    print(output)


if __name__ == "__main__":
    main()
