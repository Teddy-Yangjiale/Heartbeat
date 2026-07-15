from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path


CIRCOR_ROOT = "https://physionet.org/files/circor-heart-sound/1.0.1"
PHYSIONET_2016_ROOT = "https://physionet.org/files/challenge-2016/1.0.0"


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"reuse {destination}")
        return destination
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.part")
    print(f"download {url}")
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Heartbeat-Dataset-Downloader/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            temporary.replace(destination)
            return destination
        except Exception:
            if temporary.exists():
                try:
                    temporary.unlink()
                except PermissionError:
                    pass
            if attempt == 3:
                raise
            print(f"retry {attempt}/3 {url}")
            time.sleep(attempt)
    raise RuntimeError(f"Unable to download {url}")


def download_circor(root: Path, subject_limit: int) -> dict:
    destination = root / "circor-heart-sound-1.0.1"
    records_path = download(f"{CIRCOR_ROOT}/RECORDS", destination / "RECORDS")
    download(f"{CIRCOR_ROOT}/LICENSE.txt", destination / "LICENSE.txt")
    download(f"{CIRCOR_ROOT}/training_data.csv", destination / "training_data.csv")

    records = [line.strip() for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected_subjects: list[str] = []
    for record in records:
        subject = Path(record).name.split("_", maxsplit=1)[0]
        if subject not in selected_subjects:
            selected_subjects.append(subject)
        if len(selected_subjects) >= max(1, subject_limit):
            break
    selected = [record for record in records if Path(record).name.split("_", maxsplit=1)[0] in selected_subjects]

    downloaded: list[str] = []
    for subject in selected_subjects:
        relative = Path("training_data") / f"{subject}.txt"
        download(f"{CIRCOR_ROOT}/{relative.as_posix()}", destination / relative)
        downloaded.append(relative.as_posix())
    for record in selected:
        for suffix in (".wav", ".hea", ".tsv"):
            relative = Path(record + suffix)
            download(f"{CIRCOR_ROOT}/{relative.as_posix()}", destination / relative)
            downloaded.append(relative.as_posix())

    manifest = {
        "dataset": "CirCor DigiScope Phonocardiogram Dataset",
        "version": "1.0.1",
        "license": "Open Data Commons Attribution License 1.0",
        "source": f"{CIRCOR_ROOT}/",
        "subjects": selected_subjects,
        "records": selected,
        "files": downloaded,
    }
    (destination / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def download_physionet2016_validation(root: Path) -> dict:
    destination = root / "physionet-challenge-2016-1.0.0"
    archive = download(f"{PHYSIONET_2016_ROOT}/validation.zip", destination / "validation.zip")
    extracted = destination / "validation"
    if not extracted.exists():
        print(f"extract {archive}")
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(destination)
    wav_files = sorted(path.relative_to(destination).as_posix() for path in extracted.rglob("*.wav"))
    manifest = {
        "dataset": "PhysioNet/CinC Challenge 2016 validation",
        "version": "1.0.0",
        "license": "Open Data Commons Attribution License 1.0",
        "source": f"{PHYSIONET_2016_ROOT}/validation.zip",
        "wav_count": len(wav_files),
        "files": wav_files,
    }
    (destination / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download reproducible heartbeat denoising datasets.")
    parser.add_argument(
        "dataset",
        choices=("starter", "circor", "physionet2016-validation"),
        help="Dataset tier to download",
    )
    parser.add_argument("--root", type=Path, default=Path("data/external"), help="Local data root")
    parser.add_argument(
        "--circor-subjects",
        type=int,
        default=5,
        help="Number of CirCor subjects; all available auscultation locations are retained",
    )
    args = parser.parse_args()

    manifests = []
    if args.dataset in ("starter", "circor"):
        manifests.append(download_circor(args.root, args.circor_subjects))
    if args.dataset in ("starter", "physionet2016-validation"):
        manifests.append(download_physionet2016_validation(args.root))
    print(json.dumps(manifests, indent=2))


if __name__ == "__main__":
    main()
