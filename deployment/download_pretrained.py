import hashlib
from pathlib import Path, PurePosixPath
import tempfile
import tomllib
from zipfile import ZipFile

import gdown


def download_policy(policy, repository, temporary_dir):
    destination = repository / policy["destination"]
    if destination.exists():
        raise FileExistsError(f"Policy destination already exists: {destination}")

    archive_path = temporary_dir / policy["archive"]
    print(f"Downloading {policy['name']} to {archive_path}", flush=True)
    gdown.download(url=policy["url"], output=str(archive_path), quiet=False)
    with archive_path.open("rb") as file:
        checksum = hashlib.file_digest(file, "sha256").hexdigest()
    if checksum != policy["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {archive_path}: received {checksum}")

    print(f"Extracting {policy['name']}", flush=True)
    with tempfile.TemporaryDirectory(dir=temporary_dir) as staging:
        staging_root = Path(staging)
        with ZipFile(archive_path) as archive:
            # Archives retain repository-relative paths; restrict extraction to this run.
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if ".." in path.parts or not path.is_relative_to(PurePosixPath(policy["destination"])):
                    raise ValueError(f"Unexpected archive path: {member.filename}")
            archive.extractall(staging_root)

        extracted = staging_root / policy["destination"]
        for name in policy["required_files"]:
            if (extracted / name).stat().st_size == 0:
                raise ValueError(f"Empty policy file: {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        extracted.rename(destination)
    print(f"Installed {policy['name']}: {destination}", flush=True)


def main():
    config_path = Path(__file__).with_suffix(".toml")
    with config_path.open("rb") as file:
        cfg = tomllib.load(file)
    repository = Path(__file__).resolve().parents[1]
    temporary_dir = repository / cfg["temporary_dir"]
    temporary_dir.mkdir(parents=True, exist_ok=True)

    for policy in cfg["policies"]:
        download_policy(policy, repository, temporary_dir)


if __name__ == "__main__":
    main()
