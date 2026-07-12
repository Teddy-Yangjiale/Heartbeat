from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.io import wavfile


SAMPLE_RATE = 44100
PPQ = 480

KEY_OFFSETS = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}

SCALES = {
    "Major": [0, 2, 4, 5, 7, 9, 11],
    "Natural minor": [0, 2, 3, 5, 7, 8, 10],
    "Dorian": [0, 2, 3, 5, 7, 9, 10],
    "Mixolydian": [0, 2, 4, 5, 7, 9, 10],
}


@dataclass(frozen=True)
class PhraseRequest:
    key: str = "C"
    scale: str = "Major"
    bars: int = 4
    tempo_bpm: float | None = None
    anchor_degrees: tuple[int, ...] = (4, 2, 7)
    intention: str = "Reflective"
    candidate_count: int = 4
    heartbeat_influence: float = 0.65


@dataclass(frozen=True)
class NoteEvent:
    start_beats: float
    duration_beats: float
    midi_note: int
    velocity: int
    role: str


STRATEGIES = (
    ("Pulse motif", "Keeps the heartbeat timing contour most clearly."),
    ("Call and response", "Repeats the patient motif, then answers it with a small variation."),
    ("Expanded arc", "Stretches the motif into a longer rising and returning phrase."),
    ("Spacious variation", "Uses more rests and leaves room for live improvisation."),
)


def parse_anchor_degrees(value: str) -> tuple[int, ...]:
    tokens = value.replace("-", ",").replace(" ", ",").split(",")
    degrees = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        degree = int(token)
        if degree < 1 or degree > 7:
            raise ValueError("Patient-selected scale degrees must be between 1 and 7.")
        degrees.append(degree)
    if not degrees:
        raise ValueError("Provide at least one patient-selected scale degree, for example 4, 2, 7.")
    return tuple(degrees)


def generate_phrase_candidates(
    heartbeat_summary: dict[str, Any],
    beat_times: np.ndarray | list[float],
    request: PhraseRequest,
) -> dict[str, Any]:
    _validate_request(request)
    source_bpm = float(heartbeat_summary.get("tempo", {}).get("estimated_bpm") or 72.0)
    tempo = float(request.tempo_bpm or np.clip(source_bpm, 50.0, 110.0))
    heartbeat_pattern = _heartbeat_rhythm_pattern(beat_times, source_bpm)
    seed = _stable_seed(heartbeat_pattern, request)

    candidates = []
    artifacts: dict[str, bytes] = {}
    for index in range(request.candidate_count):
        strategy_name, rationale = STRATEGIES[index % len(STRATEGIES)]
        rng = np.random.default_rng(seed + index * 104729)
        notes = _compose_candidate(
            request=request,
            heartbeat_pattern=heartbeat_pattern,
            strategy_index=index % len(STRATEGIES),
            rng=rng,
        )
        candidate_id = f"candidate_{index + 1:02d}"
        wav_data = render_phrase_wav(notes, tempo)
        midi_data = render_phrase_midi(notes, tempo, title=strategy_name)
        csv_data = _notes_csv(notes).encode("utf-8")
        metadata = {
            "id": candidate_id,
            "name": strategy_name,
            "rationale": rationale,
            "tempo_bpm": tempo,
            "source_heartbeat_bpm": source_bpm,
            "heartbeat_influence": request.heartbeat_influence,
            "patient_anchor_degrees": list(request.anchor_degrees),
            "note_count": len(notes),
            "duration_seconds": request.bars * 4.0 * 60.0 / tempo,
            "files": {
                "audio_preview": f"{candidate_id}.wav",
                "editable_midi": f"{candidate_id}.mid",
                "note_events": f"{candidate_id}_notes.csv",
            },
        }
        artifacts[metadata["files"]["audio_preview"]] = wav_data
        artifacts[metadata["files"]["editable_midi"]] = midi_data
        artifacts[metadata["files"]["note_events"]] = csv_data
        candidates.append({**metadata, "wav_bytes": wav_data, "midi_bytes": midi_data, "notes": notes})

    manifest = {
        "purpose": "Short musical idea candidates for therapist-patient co-composition",
        "scope_note": (
            "These are editable prompts, not finished compositions or clinical recommendations. "
            "The therapist and patient retain selection and authorship."
        ),
        "request": asdict(request),
        "heartbeat_rhythm_pattern_beats": heartbeat_pattern,
        "candidates": [{k: v for k, v in item.items() if k not in {"wav_bytes", "midi_bytes", "notes"}} for item in candidates],
    }
    artifacts["candidate_manifest.json"] = json.dumps(manifest, indent=2).encode("utf-8")
    return {
        "tempo_bpm": tempo,
        "source_heartbeat_bpm": source_bpm,
        "heartbeat_pattern": heartbeat_pattern,
        "candidates": candidates,
        "manifest": manifest,
        "zip_bytes": _zip_artifacts(artifacts),
    }


def _validate_request(request: PhraseRequest) -> None:
    if request.key not in KEY_OFFSETS:
        raise ValueError(f"Unsupported key: {request.key}")
    if request.scale not in SCALES:
        raise ValueError(f"Unsupported scale: {request.scale}")
    if request.bars not in {2, 4, 8}:
        raise ValueError("Phrase length must be 2, 4, or 8 bars.")
    if not 1 <= request.candidate_count <= len(STRATEGIES):
        raise ValueError(f"Candidate count must be between 1 and {len(STRATEGIES)}.")
    if not 0.0 <= request.heartbeat_influence <= 1.0:
        raise ValueError("Heartbeat influence must be between 0 and 1.")
    if request.tempo_bpm is not None and not 40.0 <= request.tempo_bpm <= 160.0:
        raise ValueError("Tempo must be between 40 and 160 BPM.")
    if not request.anchor_degrees or any(degree < 1 or degree > 7 for degree in request.anchor_degrees):
        raise ValueError("Anchor degrees must contain values from 1 to 7.")


def _heartbeat_rhythm_pattern(beat_times: np.ndarray | list[float], source_bpm: float) -> list[float]:
    times = np.asarray(beat_times, dtype=np.float64)
    ibis = np.diff(times)
    ibis = ibis[np.isfinite(ibis) & (ibis > 0.25) & (ibis < 2.0)]
    if len(ibis) < 2:
        return [1.0, 1.0, 1.0, 1.0]
    reference = 60.0 / max(source_bpm, 1.0)
    ratios = np.clip(ibis / reference, 0.55, 1.75)
    quantized = np.round(ratios * 4.0) / 4.0
    pattern = quantized[: min(8, len(quantized))].astype(float).tolist()
    return pattern or [1.0, 1.0, 1.0, 1.0]


def _stable_seed(heartbeat_pattern: list[float], request: PhraseRequest) -> int:
    payload = json.dumps({"heartbeat": heartbeat_pattern, "request": asdict(request)}, sort_keys=True).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _compose_candidate(
    request: PhraseRequest,
    heartbeat_pattern: list[float],
    strategy_index: int,
    rng: np.random.Generator,
) -> list[NoteEvent]:
    total_beats = request.bars * 4.0
    durations = _strategy_durations(heartbeat_pattern, strategy_index, request.heartbeat_influence, rng)
    pitches = _strategy_degrees(request.anchor_degrees, strategy_index, rng)
    root_midi = 60 + KEY_OFFSETS[request.key]
    scale = SCALES[request.scale]
    register_shift = {"Grounded": -12, "Reflective": 0, "Hopeful": 0, "Spacious": 12}.get(request.intention, 0)

    melody: list[NoteEvent] = []
    cursor = 0.0
    note_index = 0
    while cursor < total_beats - 0.01:
        duration = durations[note_index % len(durations)]
        if strategy_index == 3 and note_index % 3 == 2:
            cursor += min(duration, total_beats - cursor)
            note_index += 1
            continue
        duration = min(duration, total_beats - cursor)
        degree = pitches[note_index % len(pitches)]
        octave, scale_index = divmod(degree - 1, len(scale))
        midi_note = root_midi + scale[scale_index] + octave * 12 + register_shift
        velocity = int(np.clip(78 + rng.integers(-8, 9), 55, 100))
        melody.append(NoteEvent(cursor, max(0.25, duration * 0.88), midi_note, velocity, "melody"))
        cursor += duration
        note_index += 1

    harmony = _make_harmony(root_midi + register_shift, scale, request.bars)
    return sorted(melody + harmony, key=lambda note: (note.start_beats, note.role, note.midi_note))


def _strategy_durations(
    heartbeat_pattern: list[float], strategy_index: int, influence: float, rng: np.random.Generator
) -> list[float]:
    base = np.asarray(heartbeat_pattern, dtype=np.float64)
    regular = np.ones_like(base)
    blended = regular * (1.0 - influence) + base * influence
    if strategy_index == 1:
        transformed = np.concatenate([blended[:4], blended[:4][::-1]])
    elif strategy_index == 2:
        transformed = blended * 1.5
    elif strategy_index == 3:
        transformed = blended * 1.75
    else:
        transformed = blended
    jitter = rng.choice([-0.25, 0.0, 0.25], size=len(transformed), p=[0.12, 0.76, 0.12])
    return np.clip(np.round((transformed + jitter) * 4.0) / 4.0, 0.5, 2.5).astype(float).tolist()


def _strategy_degrees(anchor_degrees: tuple[int, ...], strategy_index: int, rng: np.random.Generator) -> list[int]:
    anchors = list(anchor_degrees)
    if strategy_index == 0:
        return anchors
    if strategy_index == 1:
        answer = [int(np.clip(degree + rng.choice([-1, 1]), 1, 7)) for degree in anchors]
        return anchors + answer
    if strategy_index == 2:
        peak = min(8, max(anchors) + 2)
        return anchors + list(range(max(1, anchors[-1]), peak + 1)) + anchors[::-1]
    return anchors + [anchors[-1], anchors[0]]


def _make_harmony(root_midi: int, scale: list[int], bars: int) -> list[NoteEvent]:
    events = []
    progression = [0, 3, 4, 0]
    for bar in range(bars):
        degree_index = progression[bar % len(progression)] % len(scale)
        bass = root_midi - 12 + scale[degree_index]
        chord_indices = [degree_index, (degree_index + 2) % len(scale), (degree_index + 4) % len(scale)]
        for chord_index in chord_indices:
            octave = 12 if chord_index < degree_index else 0
            events.append(NoteEvent(bar * 4.0, 3.8, root_midi + scale[chord_index] + octave, 42, "harmony"))
        events.append(NoteEvent(bar * 4.0, 3.8, bass, 46, "bass"))
    return events


def render_phrase_wav(notes: list[NoteEvent], tempo_bpm: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    beat_seconds = 60.0 / tempo_bpm
    end_beats = max((note.start_beats + note.duration_beats for note in notes), default=1.0)
    audio = np.zeros((int((end_beats * beat_seconds + 0.4) * sample_rate), 2), dtype=np.float64)
    for note in notes:
        start = int(note.start_beats * beat_seconds * sample_rate)
        duration = note.duration_beats * beat_seconds
        length = max(1, int(duration * sample_rate))
        t = np.arange(length, dtype=np.float64) / sample_rate
        frequency = 440.0 * 2.0 ** ((note.midi_note - 69) / 12.0)
        if note.role == "melody":
            tone = np.sin(2 * np.pi * frequency * t) + 0.35 * np.sin(4 * np.pi * frequency * t)
            envelope = _adsr(length, sample_rate, 0.015, 0.12, 0.65, min(0.18, duration * 0.35))
            pan = 0.15
        else:
            tone = np.sin(2 * np.pi * frequency * t) + 0.12 * np.sin(2 * np.pi * frequency * 2 * t)
            envelope = _adsr(length, sample_rate, 0.08, 0.2, 0.42, min(0.3, duration * 0.35))
            pan = -0.12 if note.role == "bass" else 0.0
        gain = (note.velocity / 127.0) * (0.26 if note.role == "melody" else 0.10)
        mono = tone * envelope * gain
        end = min(len(audio), start + length)
        stereo = np.column_stack((mono * (1.0 - pan), mono * (1.0 + pan)))
        audio[start:end] += stereo[: end - start]
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio *= 0.92 / peak
    pcm = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, pcm)
    return buffer.getvalue()


def _adsr(length: int, sample_rate: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    attack_n = min(length, int(attack * sample_rate))
    decay_n = min(length - attack_n, int(decay * sample_rate))
    release_n = min(length - attack_n - decay_n, int(release * sample_rate))
    sustain_n = max(0, length - attack_n - decay_n - release_n)
    parts = [
        np.linspace(0.0, 1.0, attack_n, endpoint=False),
        np.linspace(1.0, sustain, decay_n, endpoint=False),
        np.full(sustain_n, sustain),
        np.linspace(sustain, 0.0, release_n, endpoint=True),
    ]
    envelope = np.concatenate([part for part in parts if len(part)])
    return np.pad(envelope, (0, max(0, length - len(envelope))))[:length]


def render_phrase_midi(notes: list[NoteEvent], tempo_bpm: float, title: str) -> bytes:
    events: list[tuple[int, int, bytes]] = []
    for note in notes:
        channel = 0 if note.role == "melody" else 1
        start_tick = int(round(note.start_beats * PPQ))
        end_tick = int(round((note.start_beats + note.duration_beats) * PPQ))
        events.append((start_tick, 1, bytes([0x90 | channel, note.midi_note, note.velocity])))
        events.append((end_tick, 0, bytes([0x80 | channel, note.midi_note, 0])))
    events.sort(key=lambda event: (event[0], event[1]))

    track = bytearray()
    name = title.encode("ascii", errors="replace")[:60]
    track.extend(_varlen(0) + bytes([0xFF, 0x03, len(name)]) + name)
    micros = int(round(60_000_000 / tempo_bpm))
    track.extend(_varlen(0) + bytes([0xFF, 0x51, 0x03]) + micros.to_bytes(3, "big"))
    track.extend(_varlen(0) + bytes([0xC0, 0]))
    track.extend(_varlen(0) + bytes([0xC1, 89]))
    previous_tick = 0
    for tick, _, payload in events:
        track.extend(_varlen(tick - previous_tick) + payload)
        previous_tick = tick
    track.extend(_varlen(0) + bytes([0xFF, 0x2F, 0x00]))

    header = b"MThd" + (6).to_bytes(4, "big") + (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + PPQ.to_bytes(2, "big")
    return header + b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)


def _varlen(value: int) -> bytes:
    buffer = value & 0x7F
    output = bytearray([buffer])
    value >>= 7
    while value:
        buffer = (value & 0x7F) | 0x80
        output.insert(0, buffer)
        value >>= 7
    return bytes(output)


def _notes_csv(notes: list[NoteEvent]) -> str:
    lines = ["start_beats,duration_beats,midi_note,velocity,role"]
    lines.extend(
        f"{note.start_beats:.3f},{note.duration_beats:.3f},{note.midi_note},{note.velocity},{note.role}"
        for note in notes
    )
    return "\n".join(lines) + "\n"


def _zip_artifacts(artifacts: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in artifacts.items():
            archive.writestr(name, data)
    return buffer.getvalue()
