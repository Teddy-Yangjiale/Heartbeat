from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from pathlib import Path

from scipy.io import wavfile


def wav_metadata(path: Path) -> dict:
    sample_rate, audio = wavfile.read(path)
    channels = 1 if audio.ndim == 1 else int(audio.shape[1])
    frames = int(audio.shape[0])
    return {
        "sample_rate": int(sample_rate),
        "channels": channels,
        "frames": frames,
        "duration_seconds": float(frames / sample_rate),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def valid_case_id(case_id: str) -> bool:
    return re.fullmatch(r"[A-Za-z0-9_-]+", case_id) is not None


def validate_case(case_id: str, reference: Path, v1: Path, candidate: Path) -> dict:
    paths = {"reference": reference, "v1": v1, "candidate": candidate}
    for label, path in paths.items():
        if path.suffix.lower() != ".wav" or not path.is_file():
            raise ValueError(f"{case_id}: {label} must be an existing WAV: {path}")
    metadata = {label: wav_metadata(path) for label, path in paths.items()}
    signatures = {
        (item["sample_rate"], item["channels"], item["frames"])
        for item in metadata.values()
    }
    if len(signatures) != 1:
        raise ValueError(f"{case_id}: reference, v1 and candidate must have identical WAV geometry")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a reproducible, identity-hidden v1 versus candidate WAV pack.")
    parser.add_argument(
        "--case",
        action="append",
        nargs=4,
        metavar=("ID", "REFERENCE", "V1", "CANDIDATE"),
        required=True,
        help="Repeat for each listening case",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--answer-key",
        type=Path,
        help="Private answer-key path; defaults to a sibling of the listening-pack directory",
    )
    parser.add_argument("--seed", type=int, required=True, help="Recorded random seed for reproducibility")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    answer_key_path = args.answer_key or args.out.parent / f"{args.out.name}_answer_key.json"
    output_root = args.out.resolve()
    resolved_answer_key = answer_key_path.resolve()
    if resolved_answer_key == output_root or output_root in resolved_answer_key.parents:
        raise ValueError("answer key must be outside the listening-pack directory")
    answer_key_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    answer_key = {"seed": args.seed, "cases": []}
    public_manifest = {"seed": args.seed, "cases": []}
    score_rows = []

    for raw_case in args.case:
        case_id, reference_text, v1_text, candidate_text = raw_case
        if not valid_case_id(case_id):
            raise ValueError(f"case ID may contain only letters, numbers, hyphens and underscores: {case_id}")
        reference = Path(reference_text).resolve()
        v1 = Path(v1_text).resolve()
        candidate = Path(candidate_text).resolve()
        metadata = validate_case(case_id, reference, v1, candidate)

        candidate_is_a = bool(rng.getrandbits(1))
        a_source, b_source = (candidate, v1) if candidate_is_a else (v1, candidate)
        a_label, b_label = ("candidate", "v1") if candidate_is_a else ("v1", "candidate")
        destinations = {
            "reference": args.out / f"{case_id}_reference.wav",
            "A": args.out / f"{case_id}_A.wav",
            "B": args.out / f"{case_id}_B.wav",
        }
        shutil.copy2(reference, destinations["reference"])
        shutil.copy2(a_source, destinations["A"])
        shutil.copy2(b_source, destinations["B"])

        answer_key["cases"].append({"case_id": case_id, "A": a_label, "B": b_label})
        public_manifest["cases"].append(
            {
                "case_id": case_id,
                "duration_seconds": metadata["reference"]["duration_seconds"],
                "sample_rate": metadata["reference"]["sample_rate"],
                "channels": metadata["reference"]["channels"],
                "files": {
                    label: {"name": path.name, "sha256": wav_metadata(path)["sha256"]}
                    for label, path in destinations.items()
                },
            }
        )
        for variant in ("A", "B"):
            score_rows.append(
                {
                    "case_id": case_id,
                    "variant": variant,
                    "gap_noise_cleanliness_1_to_5": "",
                    "friction_suppression_1_to_5": "",
                    "s1s2_fullness_1_to_5": "",
                    "attack_integrity_1_to_5": "",
                    "rhythm_integrity_1_to_5": "",
                    "artifacts_1_to_5": "",
                    "notes": "",
                }
            )

    (args.out / "pack_manifest.json").write_text(
        json.dumps(public_manifest, indent=2), encoding="utf-8"
    )
    answer_key_path.write_text(json.dumps(answer_key, indent=2), encoding="utf-8")
    with (args.out / "blind_scores.csv").open("w", newline="", encoding="utf-8-sig") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)

    instructions = f"""# 心跳降噪盲听 A/B

1. 先听 `*_reference.wav`，再以相同播放音量比较 `*_A.wav` 和 `*_B.wav`。
2. 在 `blind_scores.csv` 中填写所有 1–5 分字段；5 分始终代表更好，`artifacts` 的 5 分表示没有可闻伪影。
3. 优先判断 S1/S2 饱满度、起音完整性和节奏完整性，不要单纯选择背景最安静的版本。
4. 如果任一版本生成、截断或明显削薄了心音，即使背景更安静，也请写进 `notes`。
5. 全部样本的 A/B 都评分并保存后，才能打开试听目录外的答案表 `{answer_key_path.name}`。
"""
    (args.out / "README.md").write_text(instructions, encoding="utf-8")
    print(f"created {len(args.case)} blind cases in {args.out}; private key: {answer_key_path}")


if __name__ == "__main__":
    main()
