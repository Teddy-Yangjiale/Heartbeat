from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


METRICS = (
    "gap_noise_cleanliness_1_to_5",
    "friction_suppression_1_to_5",
    "s1s2_fullness_1_to_5",
    "attack_integrity_1_to_5",
    "rhythm_integrity_1_to_5",
    "artifacts_1_to_5",
)
NOISE_METRICS = METRICS[:2]
SAFETY_METRICS = METRICS[2:]


def read_answer_key(path: Path) -> dict[tuple[str, str], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[tuple[str, str], str] = {}
    for case in payload.get("cases", []):
        case_id = str(case["case_id"])
        for variant in ("A", "B"):
            identity = case.get(variant)
            if identity not in ("v1", "candidate"):
                raise ValueError(f"invalid identity for {case_id}/{variant}: {identity}")
            mapping[(case_id, variant)] = identity
    if not mapping:
        raise ValueError("answer key contains no cases")
    return mapping


def read_scores(path: Path, answers: dict[tuple[str, str], str]) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    expected = set(answers)
    observed: set[tuple[str, str]] = set()
    parsed: list[dict] = []
    for line_number, row in enumerate(rows, start=2):
        key = (str(row.get("case_id", "")).strip(), str(row.get("variant", "")).strip().upper())
        if key not in expected:
            raise ValueError(f"line {line_number}: unknown case/variant {key}")
        if key in observed:
            raise ValueError(f"line {line_number}: duplicate case/variant {key}")
        observed.add(key)
        values = {}
        for metric in METRICS:
            text = str(row.get(metric, "")).strip()
            try:
                score = float(text)
            except ValueError as exc:
                raise ValueError(f"line {line_number}: {metric} must be scored from 1 to 5") from exc
            if not 1.0 <= score <= 5.0:
                raise ValueError(f"line {line_number}: {metric} must be scored from 1 to 5")
            values[metric] = score
        parsed.append(
            {
                "case_id": key[0],
                "variant": key[1],
                "identity": answers[key],
                "scores": values,
                "notes": str(row.get("notes", "")).strip(),
            }
        )
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"scores are incomplete; missing {missing}")
    return parsed


def analyze(rows: list[dict]) -> dict:
    values: dict[str, dict[str, list[float]]] = {
        identity: defaultdict(list) for identity in ("v1", "candidate")
    }
    per_case: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    notes = []
    for row in rows:
        identity = row["identity"]
        per_case[row["case_id"]][identity] = row["scores"]
        for metric, score in row["scores"].items():
            values[identity][metric].append(float(score))
        if row["notes"]:
            notes.append(
                {"case_id": row["case_id"], "identity": identity, "notes": row["notes"]}
            )

    means = {
        identity: {
            metric: sum(metric_values) / len(metric_values)
            for metric, metric_values in identity_values.items()
        }
        for identity, identity_values in values.items()
    }
    deltas = {
        metric: means["candidate"][metric] - means["v1"][metric]
        for metric in METRICS
    }
    case_totals = []
    for case_id, identities in sorted(per_case.items()):
        if set(identities) != {"v1", "candidate"}:
            raise ValueError(f"case {case_id} does not contain both identities")
        case_totals.append(
            {
                "case_id": case_id,
                "candidate_minus_v1_total": sum(
                    identities["candidate"][metric] - identities["v1"][metric]
                    for metric in METRICS
                ),
                "candidate_minus_v1_safety": sum(
                    identities["candidate"][metric] - identities["v1"][metric]
                    for metric in SAFETY_METRICS
                ),
                "candidate_minus_v1_noise": sum(
                    identities["candidate"][metric] - identities["v1"][metric]
                    for metric in NOISE_METRICS
                ),
            }
        )

    safety_not_worse = all(deltas[metric] >= 0.0 for metric in SAFETY_METRICS)
    noise_not_worse = all(deltas[metric] >= 0.0 for metric in NOISE_METRICS)
    noise_improved = any(deltas[metric] > 0.0 for metric in NOISE_METRICS)
    return {
        "case_count": len(per_case),
        "means": means,
        "candidate_minus_v1": deltas,
        "per_case": case_totals,
        "notes": notes,
        "decision_gate": {
            "safety_metrics_not_worse": safety_not_worse,
            "noise_metrics_not_worse": noise_not_worse,
            "at_least_one_noise_metric_improved": noise_improved,
            "candidate_passes": safety_not_worse and noise_not_worse and noise_improved,
            "policy": "S1/S2, attack, rhythm and artifacts must not regress; noise or friction must improve.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reveal and summarize a completed heartbeat blind A/B score sheet.")
    parser.add_argument("scores", type=Path)
    parser.add_argument("answer_key", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/blind_ab_analysis.json"))
    args = parser.parse_args()

    answers = read_answer_key(args.answer_key)
    rows = read_scores(args.scores, answers)
    report = analyze(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
