from __future__ import annotations

from datetime import datetime
from pathlib import Path

import streamlit as st

from heartbeat_preprocessor.core import ProcessingParams
from legasynth.remix_pipeline import process_song_remix
from legasynth.stems import StemSeparationUnavailable, demucs_is_available


st.set_page_config(page_title="Heartbeat Song Remixer", layout="wide")


def main() -> None:
    st.title("Heartbeat Song Remixer")
    st.write(
        "Stage 2: analyze a song WAV/MP3, fit the processed heartbeat best loop to the song beat grid, "
        "and export a separate heartbeat layer plus the final mix."
    )

    left, right = st.columns(2)
    heartbeat = left.file_uploader("Heartbeat recording", type=["wav", "mp3"])
    song = right.file_uploader("Target song", type=["wav", "mp3"])

    controls = st.columns(4)
    beats_per_bar = controls[0].selectbox("Beats per bar", options=[2, 3, 4, 5, 6, 7], index=2)
    loop_beats = controls[1].number_input("Heartbeat intervals per loop", 1, 16, int(beats_per_bar), 1)
    heartbeat_gain = controls[2].slider("Heartbeat gain (dB)", -36.0, 0.0, -15.0, 0.5)
    auto_bpm = controls[3].checkbox("Auto-detect BPM", value=True)
    bpm_override = st.number_input("Song BPM override", 20.0, 300.0, 120.0, 0.1, disabled=auto_bpm)

    auto_phase = st.checkbox("Auto-detect first beat / phase", value=True)
    first_beat_override = st.number_input(
        "First beat in seconds",
        0.0,
        120.0,
        0.0,
        0.01,
        disabled=auto_phase,
        help="Listen to the result and override this when the heartbeat starts between song beats.",
    )
    separate_stems = st.checkbox(
        "Separate vocals and accompaniment with Demucs",
        value=False,
        help="Optional and substantially slower. Demucs must be installed separately.",
    )
    extract_melody = st.checkbox(
        "Extract vocal melody with pYIN",
        value=False,
        disabled=not separate_stems,
    )
    if separate_stems and not demucs_is_available():
        st.warning("Demucs is not installed in the heartbeat environment. See STAGE2_REMIX.md.")

    if st.button("Create heartbeat remix", type="primary"):
        if heartbeat is None or song is None:
            st.error("Upload both a heartbeat recording and a song first.")
            return
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_dir = Path("work") / "remix_uploads" / run_id
        input_dir.mkdir(parents=True, exist_ok=True)
        heartbeat_path = input_dir / f"heartbeat_{heartbeat.name}"
        song_path = input_dir / f"song_{song.name}"
        heartbeat_path.write_bytes(heartbeat.getvalue())
        song_path.write_bytes(song.getvalue())
        try:
            with st.spinner("Analyzing and aligning audio..."):
                report = process_song_remix(
                    heartbeat_path=heartbeat_path,
                    song_path=song_path,
                    out_root=Path("outputs") / "remix" / run_id,
                    params=ProcessingParams(target_loop_beats=int(loop_beats)),
                    heartbeat_gain_db=float(heartbeat_gain),
                    song_bpm_override=None if auto_bpm else float(bpm_override),
                    first_beat_override=None if auto_phase else float(first_beat_override),
                    beats_per_bar=int(beats_per_bar),
                    separate_stems=separate_stems,
                    extract_melody=extract_melody,
                )
            st.session_state["remix_report"] = report
        except StemSeparationUnavailable as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Remix failed: {exc}")

    report = st.session_state.get("remix_report")
    if not report:
        st.info("Automatic BPM and first-beat estimates should be checked by listening before final export.")
        return

    song_analysis = report["song_analysis"]
    metrics = st.columns(4)
    metrics[0].metric("Song BPM", f"{song_analysis['estimated_bpm']:.2f}")
    metrics[1].metric("First beat", f"{song_analysis['first_beat_seconds']:.3f}s")
    metrics[2].metric("Meter", f"{song_analysis['beats_per_bar']}/4")
    metrics[3].metric("Beat confidence", f"{song_analysis['beat_tracking_confidence']:.0%}")
    st.caption(song_analysis["alignment_note"])

    audio = st.columns(3)
    audio[0].write("Processed heartbeat best loop")
    audio[0].audio(report["outputs"]["heartbeat_best_loop_wav"])
    audio[1].write("Aligned heartbeat layer")
    audio[1].audio(report["outputs"]["heartbeat_layer_wav"])
    audio[2].write("Final mix")
    audio[2].audio(report["outputs"]["final_mix_wav"])

    zip_path = Path(report["outputs"]["all_outputs_zip"])
    st.download_button(
        "Download all Stage 2 outputs",
        zip_path.read_bytes(),
        file_name=zip_path.name,
        mime="application/zip",
    )
    with st.expander("View run report"):
        st.json(report)


if __name__ == "__main__":
    main()
