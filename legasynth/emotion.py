"""Feature A: map a heartbeat recording to an emotional state and a visual style.

This is deliberately a transparent, rule-based mapping rather than a black-box
model. It reads the tempo / inter-beat-interval (IBI) statistics that the Stage-1
preprocessor already computed, derives a small set of heart-rate-variability (HRV)
features, places the recording on the classic valence / arousal circumplex, and
turns that point into a concrete "visual style profile" that Feature B consumes to
grade and cut the music video.

IMPORTANT: none of this is a medical diagnosis. The valence/arousal estimate is a
heuristic affective mapping over heart-rate and heart-rate-variability, following
the general direction reported in the affective-computing literature (higher heart
rate -> higher arousal; healthier/steadier HRV -> more positive/relaxed valence).
It is intended to drive an artistic effect, nothing more.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


# --- affective mapping constants (documented, tunable) -----------------------

# Heart-rate range used to normalise arousal. 55 bpm reads as calm/rested, 110 bpm
# as highly activated. Values outside are clamped.
HR_CALM_BPM = 55.0
HR_ACTIVATED_BPM = 110.0

# RMSSD (ms) range used as a proxy for parasympathetic "calm/positive" tone.
# Low short-term variability (a very metronomic beat) reads as tense/flat; a
# moderate amount of natural variability reads as relaxed. Extremely high values
# usually mean noise or irregularity, so valence is rolled back past the top.
RMSSD_LOW_MS = 10.0
RMSSD_HEALTHY_MS = 45.0
RMSSD_EXCESS_MS = 120.0


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(max(low, min(high, value)))


def _lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * _clamp(t))


def compute_hrv_features(heartbeat_summary: dict[str, Any]) -> dict[str, Any]:
    """Derive HRV features from the Stage-1 tempo summary.

    Returns mean heart rate plus SDNN, RMSSD and pNN50 style statistics in
    milliseconds, computed from the inter-beat intervals.
    """
    tempo = heartbeat_summary.get("tempo", {}) if heartbeat_summary else {}
    ibi_seconds = np.asarray(tempo.get("ibi_seconds", []) or [], dtype=np.float64)
    # Guard against absurd intervals (dropped/spurious beats) before computing HRV.
    if ibi_seconds.size:
        ibi_seconds = ibi_seconds[(ibi_seconds > 0.25) & (ibi_seconds < 2.0)]

    estimated_bpm = float(tempo.get("estimated_bpm") or 0.0)
    if ibi_seconds.size >= 1:
        mean_ibi = float(np.mean(ibi_seconds))
        mean_hr = float(60.0 / mean_ibi) if mean_ibi > 0 else estimated_bpm
    else:
        mean_hr = estimated_bpm or 75.0

    ibi_ms = ibi_seconds * 1000.0
    sdnn = float(np.std(ibi_ms)) if ibi_ms.size >= 2 else 0.0
    if ibi_ms.size >= 2:
        successive = np.diff(ibi_ms)
        rmssd = float(np.sqrt(np.mean(successive ** 2)))
        pnn50 = float(np.mean(np.abs(successive) > 50.0))
    else:
        rmssd = 0.0
        pnn50 = 0.0
    mean_ibi_ms = float(np.mean(ibi_ms)) if ibi_ms.size else (60000.0 / mean_hr if mean_hr else 0.0)
    ibi_cv = float(sdnn / mean_ibi_ms) if mean_ibi_ms > 0 else 0.0

    recording_quality = heartbeat_summary.get("recording_quality", {}) if heartbeat_summary else {}
    envelope_contrast = float(recording_quality.get("metrics", {}).get("envelope_contrast", 0.0) or 0.0)

    return {
        "mean_heart_rate_bpm": round(mean_hr, 2),
        "beat_count_used": int(ibi_seconds.size + 1) if ibi_seconds.size else 0,
        "sdnn_ms": round(sdnn, 2),
        "rmssd_ms": round(rmssd, 2),
        "pnn50": round(pnn50, 4),
        "ibi_cv": round(ibi_cv, 4),
        "envelope_contrast": round(envelope_contrast, 4),
    }


def estimate_valence_arousal(features: dict[str, Any]) -> dict[str, float]:
    """Place the recording on a [0,1] valence / [0,1] arousal plane.

    Arousal rises with heart rate. Valence rises with a healthy amount of HRV and
    falls when the beat is either flat/metronomic or highly irregular.
    """
    hr = float(features["mean_heart_rate_bpm"])
    rmssd = float(features["rmssd_ms"])

    arousal = _clamp((hr - HR_CALM_BPM) / (HR_ACTIVATED_BPM - HR_CALM_BPM))

    # Valence: triangular response peaking around a healthy RMSSD.
    if rmssd <= RMSSD_HEALTHY_MS:
        valence = _lerp(0.28, 0.85, (rmssd - RMSSD_LOW_MS) / (RMSSD_HEALTHY_MS - RMSSD_LOW_MS))
    else:
        valence = _lerp(0.85, 0.4, (rmssd - RMSSD_HEALTHY_MS) / (RMSSD_EXCESS_MS - RMSSD_HEALTHY_MS))
    # A very high resting-to-active heart rate nudges valence down (stress reading).
    valence = _clamp(valence - 0.15 * max(0.0, arousal - 0.6))

    return {"valence": round(valence, 3), "arousal": round(arousal, 3)}


def label_quadrant(valence: float, arousal: float) -> dict[str, str]:
    """Name the affective quadrant (English + Chinese) for reports and overlays."""
    high_arousal = arousal >= 0.5
    high_valence = valence >= 0.5
    if high_arousal and high_valence:
        return {"mood": "excited", "mood_zh": "欢快激昂", "quadrant": "high-arousal positive"}
    if high_arousal and not high_valence:
        return {"mood": "tense", "mood_zh": "紧张不安", "quadrant": "high-arousal negative"}
    if not high_arousal and high_valence:
        return {"mood": "serene", "mood_zh": "宁静安详", "quadrant": "low-arousal positive"}
    return {"mood": "melancholic", "mood_zh": "沉静伤感", "quadrant": "low-arousal negative"}


def build_style_profile(valence: float, arousal: float, mean_hr: float) -> dict[str, Any]:
    """Turn a valence/arousal point into concrete render controls for Feature B.

    All fields are plain numbers so the renderer stays decoupled from this module:

    - warmth:          -1 cool .. +1 warm colour push (valence)
    - saturation:      multiplier on chroma (valence + arousal)
    - brightness:      multiplier on luma (valence)
    - contrast:        multiplier on contrast around mid-grey (arousal)
    - vignette:        0..1 darkened-edge strength (low valence)
    - pulse_intensity: 0..1 per-beat zoom/brightness pulse depth (arousal)
    - flash_strength:  0..1 white flash depth on strong beats (arousal)
    - beats_per_cut:   integer beats between hard section cuts (inverse arousal)
    - grade_name:      short human label for the look
    """
    warmth = round(_lerp(-0.6, 0.6, valence), 3)
    saturation = round(_lerp(0.7, 1.35, 0.5 * valence + 0.5 * arousal), 3)
    brightness = round(_lerp(0.9, 1.12, valence), 3)
    contrast = round(_lerp(0.95, 1.28, arousal), 3)
    vignette = round(_clamp(0.55 * (1.0 - valence) + 0.15 * (1.0 - arousal)), 3)
    pulse_intensity = round(_clamp(0.35 + 0.6 * arousal), 3)
    flash_strength = round(_clamp((arousal - 0.45) / 0.55) * 0.7, 3)

    # Cut faster when aroused: ~8 beats/cut when calm, ~2 beats/cut when energetic.
    beats_per_cut = int(round(_lerp(8.0, 2.0, arousal)))
    beats_per_cut = max(2, beats_per_cut)

    quadrant = label_quadrant(valence, arousal)
    grade_name = {
        "excited": "warm-vibrant",
        "tense": "cool-hardlight",
        "serene": "soft-warm",
        "melancholic": "cool-muted",
    }[quadrant["mood"]]

    return {
        "warmth": warmth,
        "saturation": saturation,
        "brightness": brightness,
        "contrast": contrast,
        "vignette": vignette,
        "pulse_intensity": pulse_intensity,
        "flash_strength": flash_strength,
        "beats_per_cut": beats_per_cut,
        "grade_name": grade_name,
        **quadrant,
    }


def analyze_emotion(heartbeat_summary: dict[str, Any]) -> dict[str, Any]:
    """Full Feature-A pipeline: summary -> HRV features -> affect -> style profile."""
    features = compute_hrv_features(heartbeat_summary)
    affect = estimate_valence_arousal(features)
    style = build_style_profile(affect["valence"], affect["arousal"], features["mean_heart_rate_bpm"])
    return {
        "disclaimer": (
            "Heuristic affective mapping over heart rate and HRV for artistic video "
            "styling only. Not a medical or diagnostic assessment."
        ),
        "features": features,
        "affect": affect,
        "mood": style["mood"],
        "mood_zh": style["mood_zh"],
        "quadrant": style["quadrant"],
        "style_profile": style,
    }


def save_emotion_report(report: dict[str, Any], out_dir: str | os.PathLike[str]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "emotion_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
