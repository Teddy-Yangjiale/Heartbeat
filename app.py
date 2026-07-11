from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from heartbeat_preprocessor.core import (
    ProcessingParams,
    make_batch_zip,
    process_audio_bytes,
    save_result_to_dir,
)


st.set_page_config(page_title="LegaSynth Heartbeat Audio Analysis", layout="wide")


def sidebar_params() -> ProcessingParams:
    st.sidebar.header("Preprocessing Parameters")
    low = st.sidebar.slider("Band-pass low cutoff (Hz)", 5.0, 80.0, 25.0, 1.0)
    high = st.sidebar.slider("Band-pass high cutoff (Hz)", 80.0, 500.0, 160.0, 5.0)
    env_lp = st.sidebar.slider("Envelope low-pass (Hz)", 2.0, 20.0, 6.0, 0.5)
    min_bpm = st.sidebar.slider("Minimum plausible BPM", 30.0, 80.0, 40.0, 1.0)
    max_bpm = st.sidebar.slider("Maximum plausible BPM", 90.0, 180.0, 140.0, 1.0)
    prominence = st.sidebar.slider("Peak prominence", 0.01, 0.60, 0.12, 0.01)
    height_pct = st.sidebar.slider("Peak height percentile", 40.0, 90.0, 65.0, 1.0)
    double_peak_suppression = st.sidebar.slider("Double-peak suppression", 0.40, 0.90, 0.65, 0.01)
    loop_beats = st.sidebar.slider("Target loop length (beats)", 2, 12, 4, 1)
    crossfade_ms = st.sidebar.slider("Loop edge fade (ms)", 0.0, 80.0, 12.0, 1.0)

    st.sidebar.subheader("Speech and noise suppression")
    suppression_enabled = st.sidebar.checkbox("Enable speech/noise suppression", value=True)
    suppression_profile = st.sidebar.select_slider(
        "Suppression strength",
        options=["Mild", "Balanced", "Strong"],
        value="Balanced",
        disabled=not suppression_enabled,
        help="Strong mode removes more talking between heartbeats, but can make weak S1/S2 sounds quieter.",
    )
    profiles = {
        "Mild": {"hpss_margin": 1.4, "spectral_reduction_strength": 0.85, "between_beat_attenuation_db": -18.0, "beat_gate_post_ms": 260.0},
        "Balanced": {"hpss_margin": 2.0, "spectral_reduction_strength": 1.15, "between_beat_attenuation_db": -28.0, "beat_gate_post_ms": 300.0},
        "Strong": {"hpss_margin": 2.8, "spectral_reduction_strength": 1.45, "between_beat_attenuation_db": -36.0, "beat_gate_post_ms": 340.0},
    }
    speech_suppression_params = profiles[suppression_profile]
    st.sidebar.subheader("Heartbeat template confirmation")
    template_enabled = st.sidebar.checkbox("Confirm beats with heartbeat template", value=True)
    template_threshold = st.sidebar.slider(
        "Template correlation threshold", 0.10, 0.90, 0.35, 0.05, disabled=not template_enabled
    )
    return ProcessingParams(
        bandpass_low_hz=low,
        bandpass_high_hz=high,
        envelope_lowpass_hz=env_lp,
        min_bpm=min_bpm,
        max_bpm=max_bpm,
        peak_prominence=prominence,
        peak_height_percentile=height_pct,
        double_peak_suppression=double_peak_suppression,
        target_loop_beats=loop_beats,
        crossfade_ms=crossfade_ms,
        enable_speech_suppression=suppression_enabled,
        enable_template_confirmation=template_enabled,
        template_correlation_threshold=template_threshold,
        **speech_suppression_params,
    )


def render_result(result: dict, save_outputs: bool) -> dict | None:
    summary = result["summary"]
    tempo = summary["tempo"]
    loop = summary["best_loop"]
    quality = summary["quality"]
    recording_quality = summary["recording_quality"]

    st.subheader(result["name"])
    cols = st.columns(7)
    cols[0].metric("Duration", f"{summary['duration_seconds']:.2f}s")
    cols[1].metric("Sample Rate", f"{summary['sample_rate']} Hz")
    cols[2].metric("Estimated BPM", f"{tempo['estimated_bpm']:.1f}")
    cols[3].metric("Detected Beats", str(tempo["detected_beats"]))
    cols[4].metric("IBI Std", "n/a" if tempo["ibi_std_seconds"] is None else f"{tempo['ibi_std_seconds']:.3f}s")
    cols[5].metric("Peak", f"{quality['peak_dbfs']:.1f} dBFS")
    cols[6].metric("Recording Quality", f"{recording_quality['score']:.0f}/100", recording_quality["grade"])

    if quality["is_clipping_suspected"]:
        st.warning("Clipping is suspected in this file. Some detected peaks may be unreliable.")
    if quality["speech_suppression_enabled"]:
        st.caption(
            "Speech/noise suppression is active: only "
            f"{quality['beat_window_coverage_fraction']:.0%} of the recording is retained at full level around detected beats; "
            f"the remaining audio is attenuated by {quality['between_beat_attenuation_db']:.0f} dB."
        )
    if tempo["detected_beats"] < 4:
        st.warning("Few beats were detected. Try lowering peak prominence or widening the BPM range.")
    if not recording_quality["is_recommended_for_loop"]:
        st.warning("This recording is not recommended for a production loop: " + " ".join(recording_quality["reasons"]))

    st.caption(
        "Best loop: "
        f"{loop['start_seconds']:.2f}s to {loop['end_seconds']:.2f}s, "
        f"{loop['num_beats']} beats, local BPM {loop['local_bpm']:.1f}, method {loop['method']}."
    )
    st.caption(
        f"BPM method: {tempo['method']}; consensus windows: "
        f"{tempo['consensus_window_count']}/{tempo['window_count']}."
    )
    st.caption(
        f"Template confirmation: {tempo['template_confirmed_beats']}/{tempo['initial_detected_beats']} beats; "
        f"median correlation: {tempo['template_median_correlation']}."
    )

    st.image(result["artifacts"]["diagnostic_plot.png"], caption="Diagnostic plot", width='stretch')

    audio_cols = st.columns(3)
    audio_cols[0].write("Cleaned heartbeat audio")
    audio_cols[0].audio(result["artifacts"]["cleaned.wav"], format="audio/wav")
    audio_cols[1].write("Speech-suppressed diagnostic audio")
    audio_cols[1].audio(result["artifacts"]["filtered_detection.wav"], format="audio/wav")
    audio_cols[2].write("Best loop")
    audio_cols[2].audio(result["artifacts"]["best_loop.wav"], format="audio/wav")

    dl_cols = st.columns(5)
    dl_cols[0].download_button(
        "Download result zip",
        result["zip_bytes"],
        file_name=f"{result['stem']}_heartbeat_outputs.zip",
        mime="application/zip",
        key=f"{result['stem']}_zip",
    )
    dl_cols[1].download_button(
        "tempo_summary.json",
        result["artifacts"]["tempo_summary.json"],
        file_name=f"{result['stem']}_tempo_summary.json",
        mime="application/json",
        key=f"{result['stem']}_json",
    )
    dl_cols[2].download_button(
        "beat_times.csv",
        result["artifacts"]["beat_times.csv"],
        file_name=f"{result['stem']}_beat_times.csv",
        mime="text/csv",
        key=f"{result['stem']}_beats",
    )
    dl_cols[3].download_button(
        "best_loop.wav",
        result["artifacts"]["best_loop.wav"],
        file_name=f"{result['stem']}_best_loop.wav",
        mime="audio/wav",
        key=f"{result['stem']}_loop",
    )
    dl_cols[4].download_button(
        "diagnostic_plot.png",
        result["artifacts"]["diagnostic_plot.png"],
        file_name=f"{result['stem']}_diagnostic_plot.png",
        mime="image/png",
        key=f"{result['stem']}_plot",
    )

    with st.expander("View tempo summary JSON"):
        st.json(summary)
    with st.expander("Recording-quality reasons and loop candidates"):
        st.json(recording_quality)
        st.dataframe(result["loop_candidates"], width='stretch')
        st.dataframe(result["template_analysis"], width='stretch')

    with st.expander("Manual beat and loop correction"):
        st.caption("Edit beat times in seconds, then optionally set an exact loop range. The exports will be regenerated.")
        if "input_data" not in result or "params" not in result:
            st.warning("This result was created before manual correction was available. Process the input again first.")
            return None
        revision = result.get("_ui_revision", 0)
        key_prefix = f"{result['stem']}_{revision}"
        edited_beats = st.data_editor(
            pd.DataFrame({"time_seconds": result["beat_times"]}),
            num_rows="dynamic",
            width='stretch',
            key=f"{key_prefix}_manual_beats",
        )
        loop_cols = st.columns(2)
        manual_start = loop_cols[0].number_input(
            "Manual loop start (seconds)",
            min_value=0.0,
            max_value=float(summary["duration_seconds"]),
            value=float(loop["start_seconds"]),
            step=0.01,
            key=f"{key_prefix}_manual_loop_start",
        )
        manual_end = loop_cols[1].number_input(
            "Manual loop end (seconds)",
            min_value=0.0,
            max_value=float(summary["duration_seconds"]),
            value=float(loop["end_seconds"]),
            step=0.01,
            key=f"{key_prefix}_manual_loop_end",
        )
        action_cols = st.columns(2)
        apply_changes = action_cols[0].button("Apply manual corrections", key=f"{key_prefix}_apply")
        reset_changes = action_cols[1].button("Restore automatic analysis", key=f"{key_prefix}_reset")
        if reset_changes:
            updated = process_audio_bytes(result["name"], result["input_data"], params=result["params"])
        elif apply_changes:
            manual_times = pd.to_numeric(edited_beats["time_seconds"], errors="coerce").dropna().to_numpy()
            updated = process_audio_bytes(
                result["name"],
                result["input_data"],
                params=result["params"],
                manual_beat_times=manual_times,
                manual_loop_range=(float(manual_start), float(manual_end)),
            )
        else:
            return None
        updated["_ui_revision"] = revision + 1
        if save_outputs:
            output_root = Path("outputs") / "manual_corrections" / datetime.now().strftime("%Y%m%d_%H%M%S")
            save_result_to_dir(updated, output_root)
        return updated

    return None


def main() -> None:
    st.title("LegaSynth Heartbeat Audio Analysis")
    st.write("Stage 1: upload heartbeat WAV or MP3 files, then export all preprocessing parameters, beat data, loop audio, and diagnostics.")

    params = sidebar_params()
    save_outputs = st.sidebar.checkbox("Also save outputs to D:\\Heartbeat\\outputs", value=True)

    files = st.file_uploader(
        "Upload heartbeat audio files",
        type=["wav", "mp3"],
        accept_multiple_files=True,
    )

    if not files:
        st.info("Drop WAV or MP3 heartbeat recordings here to begin. The app does not require a GPU.")
        return

    if st.button("Process uploaded files", type="primary"):
        results = []
        output_root = Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        progress = st.progress(0)
        for i, uploaded in enumerate(files):
            try:
                result = process_audio_bytes(uploaded.name, uploaded.getvalue(), params)
                results.append(result)
                if save_outputs:
                    save_result_to_dir(result, output_root)
            except Exception as exc:
                st.error(f"Failed to process {uploaded.name}: {exc}")
            progress.progress((i + 1) / len(files))

        if not results:
            st.error("No files were processed successfully.")
            return

        st.session_state["results"] = results
        if save_outputs:
            st.success(f"Saved outputs to {output_root.resolve()}")

    results = st.session_state.get("results", [])
    if results:
        st.download_button(
            "Download all processed outputs as one zip",
            make_batch_zip(results),
            file_name="legasynth_heartbeat_batch_outputs.zip",
            mime="application/zip",
        )
        for index, result in enumerate(results):
            with st.expander(result["name"], expanded=len(results) == 1):
                updated = render_result(result, save_outputs)
                if updated is not None:
                    results[index] = updated
                    st.session_state["results"] = results
                    st.rerun()


if __name__ == "__main__":
    main()
