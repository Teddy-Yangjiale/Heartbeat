from __future__ import annotations

import tempfile
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
from legasynth.phrase_generator import (
    KEY_OFFSETS,
    SCALES,
    PhraseRequest,
    generate_phrase_candidates,
    parse_anchor_degrees,
)
from legasynth.pipeline import process_one


st.set_page_config(page_title="LegaSynth Co-Composition Studio", layout="wide")


def sidebar_params() -> ProcessingParams:
    with st.sidebar.expander("Advanced heartbeat timing settings", expanded=False):
        low = st.slider("Band-pass low cutoff (Hz)", 5.0, 80.0, 25.0, 1.0)
        high = st.slider("Band-pass high cutoff (Hz)", 80.0, 500.0, 160.0, 5.0)
        env_lp = st.slider("Envelope low-pass (Hz)", 2.0, 20.0, 6.0, 0.5)
        min_bpm = st.slider("Minimum plausible BPM", 30.0, 80.0, 40.0, 1.0)
        max_bpm = st.slider("Maximum plausible BPM", 90.0, 180.0, 140.0, 1.0)
        prominence = st.slider("Peak prominence", 0.01, 0.60, 0.12, 0.01)
        height_pct = st.slider("Peak height percentile", 40.0, 90.0, 65.0, 1.0)
        double_peak_suppression = st.slider("Double-peak suppression", 0.40, 0.90, 0.65, 0.01)

        st.subheader("Speech and noise suppression")
        suppression_enabled = st.checkbox("Enable speech/noise suppression", value=True)
        suppression_profile = st.select_slider(
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
        st.subheader("Heartbeat template confirmation")
        template_enabled = st.checkbox("Confirm beats with heartbeat template", value=True)
        template_threshold = st.slider(
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


def stage1_tab(params: ProcessingParams, save_outputs: bool) -> None:
    st.write("Upload heartbeat WAV or MP3 files, then export all preprocessing parameters, beat data, loop audio, and diagnostics.")

    files = st.file_uploader(
        "Upload heartbeat audio files",
        type=["wav", "mp3"],
        accept_multiple_files=True,
        key="stage1_uploader",
    )

    if not files:
        st.info("Drop WAV or MP3 heartbeat recordings here to begin. The app does not require a GPU.")
        return

    if st.button("Process uploaded files", type="primary", key="stage1_run"):
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
            key="stage1_batchzip",
        )
        for index, result in enumerate(results):
            with st.expander(result["name"], expanded=len(results) == 1):
                updated = render_result(result, save_outputs)
                if updated is not None:
                    results[index] = updated
                    st.session_state["results"] = results
                    st.rerun()


# =====================================================================================
# Primary workflow: short musical idea candidates for therapist-patient co-composition
# =====================================================================================


def render_phrase_candidates(result: dict, heartbeat_result: dict) -> None:
    summary = heartbeat_result["summary"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Detected heartbeat", f"{result['source_heartbeat_bpm']:.1f} BPM")
    metric_cols[1].metric("Candidate tempo", f"{result['tempo_bpm']:.1f} BPM")
    metric_cols[2].metric("Reliable beats", str(summary["tempo"]["detected_beats"]))
    metric_cols[3].metric("Candidate ideas", str(len(result["candidates"])))
    st.caption(
        "Heartbeat timing was used as compositional material, not as an emotion diagnosis. "
        "Choose, reject, combine, or improvise on any candidate with the patient."
    )

    for index, candidate in enumerate(result["candidates"]):
        with st.container(border=True):
            title_col, info_col = st.columns([2, 3])
            title_col.subheader(f"{index + 1}. {candidate['name']}")
            info_col.write(candidate["rationale"])
            st.audio(candidate["wav_bytes"], format="audio/wav")
            actions = st.columns(3)
            actions[0].download_button(
                "Download MIDI",
                candidate["midi_bytes"],
                file_name=f"{candidate['id']}.mid",
                mime="audio/midi",
                key=f"{candidate['id']}_midi",
            )
            actions[1].download_button(
                "Download WAV preview",
                candidate["wav_bytes"],
                file_name=f"{candidate['id']}.wav",
                mime="audio/wav",
                key=f"{candidate['id']}_wav",
            )
            actions[2].caption(
                f"{candidate['note_count']} note events · {candidate['duration_seconds']:.1f}s"
            )

    st.download_button(
        "Download all editable candidates",
        result["zip_bytes"],
        file_name="legasynth_phrase_candidates.zip",
        mime="application/zip",
        type="primary",
        key="phrase_all_zip",
    )


def phrase_candidates_tab(params: ProcessingParams) -> None:
    st.write(
        "Generate several short musical starting points for a therapist and patient to shape together. "
        "The output is deliberately unfinished: editable MIDI plus a simple WAV preview."
    )

    heartbeat_file = st.file_uploader(
        "Heartbeat recording (.wav / .mp3)",
        type=["wav", "mp3"],
        accept_multiple_files=False,
        key="phrase_heartbeat",
    )

    st.markdown("#### Patient-provided musical material")
    patient_cols = st.columns([2, 3])
    anchor_text = patient_cols[0].text_input(
        "Chosen scale degrees",
        value="4, 2, 7",
        help="Keep the patient's order and meaning. Use numbers 1 to 7, separated by commas.",
        key="phrase_anchors",
    )
    patient_cols[1].text_area(
        "Meaning or story notes",
        value="",
        placeholder="Optional session notes for the therapist. These stay in the interface and are not interpreted by the generator.",
        key="phrase_story",
    )

    st.markdown("#### Therapist controls")
    row1 = st.columns(4)
    key = row1[0].selectbox("Key", list(KEY_OFFSETS), index=0, key="phrase_key")
    scale = row1[1].selectbox("Scale", list(SCALES), index=0, key="phrase_scale")
    intention = row1[2].selectbox(
        "Working character",
        ["Reflective", "Grounded", "Hopeful", "Spacious"],
        index=0,
        key="phrase_intention",
    )
    bars = row1[3].selectbox("Length", [2, 4, 8], index=1, format_func=lambda value: f"{value} bars", key="phrase_bars")

    row2 = st.columns(3)
    tempo_mode = row2[0].selectbox(
        "Tempo source", ["Use heartbeat BPM", "Set manually"], index=0, key="phrase_tempo_mode"
    )
    manual_tempo = row2[1].slider(
        "Manual tempo (BPM)", 40, 160, 72, 1, disabled=tempo_mode == "Use heartbeat BPM", key="phrase_tempo"
    )
    influence = row2[2].slider(
        "Heartbeat rhythm influence", 0.0, 1.0, 0.65, 0.05, key="phrase_influence"
    )

    if heartbeat_file is None:
        st.info("Upload a heartbeat recording to generate candidate ideas.")
        return

    if st.button("Generate candidate ideas", type="primary", key="phrase_generate"):
        try:
            anchor_degrees = parse_anchor_degrees(anchor_text)
            heartbeat_result = process_audio_bytes(heartbeat_file.name, heartbeat_file.getvalue(), params)
            phrase_request = PhraseRequest(
                key=key,
                scale=scale,
                bars=bars,
                tempo_bpm=None if tempo_mode == "Use heartbeat BPM" else float(manual_tempo),
                anchor_degrees=anchor_degrees,
                intention=intention,
                candidate_count=4,
                heartbeat_influence=float(influence),
            )
            phrase_result = generate_phrase_candidates(
                heartbeat_result["summary"], heartbeat_result["beat_times"], phrase_request
            )
            st.session_state["phrase_result"] = phrase_result
            st.session_state["phrase_heartbeat_result"] = heartbeat_result
        except Exception as exc:
            st.error(f"Could not generate candidates: {exc}")

    phrase_result = st.session_state.get("phrase_result")
    heartbeat_result = st.session_state.get("phrase_heartbeat_result")
    if phrase_result and heartbeat_result:
        render_phrase_candidates(phrase_result, heartbeat_result)


# =====================================================================================
# Legacy workflow: heartbeat-inspired music video (Features A + B + C)
# =====================================================================================

def _read_bytes(path: str | None) -> bytes | None:
    if not path:
        return None
    p = Path(path)
    return p.read_bytes() if p.exists() else None


def run_full_pipeline(
    heartbeat_files,
    video_file,
    params: ProcessingParams,
    effect_strength: float,
    heartbeat_gain_db: float,
    title_text: str,
    enable_emotion: bool,
    enable_beat_editing: bool,
    show_overlay: bool,
    duration_limit: float | None,
) -> list[dict]:
    reports: list[dict] = []
    run_root = Path("outputs") / ("mv_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    tmp_dir = Path(tempfile.mkdtemp(prefix="legasynth_mv_"))
    video_path = tmp_dir / video_file.name
    video_path.write_bytes(video_file.getvalue())

    progress = st.progress(0.0)
    status = st.empty()
    for i, hb in enumerate(heartbeat_files):
        status.write(
            f"Processing {hb.name} ({i + 1}/{len(heartbeat_files)}) — "
            "heartbeat analysis, mixing, styling, encode…"
        )
        hb_path = tmp_dir / hb.name
        hb_path.write_bytes(hb.getvalue())
        try:
            report = process_one(
                heartbeat_path=hb_path,
                video_path=video_path,
                out_root=run_root,
                params=params,
                heartbeat_gain_db=heartbeat_gain_db,
                effect_strength=effect_strength,
                duration_limit=duration_limit,
                title_text=title_text,
                enable_emotion=enable_emotion,
                enable_beat_editing=enable_beat_editing,
                show_overlay=show_overlay,
            )
            reports.append(report)
        except Exception as exc:
            st.error(f"Failed to process {hb.name}: {exc}")
        progress.progress((i + 1) / len(heartbeat_files))
    status.write(f"Done. Outputs saved under {run_root.resolve()}")
    return reports


def render_pipeline_report(report: dict) -> None:
    st.subheader(Path(report["heartbeat_file"]).name)

    emotion = report.get("emotion")
    if emotion:
        affect = emotion["affect"]
        style = emotion["style_profile"]
        cols = st.columns(5)
        cols[0].metric("Mood", emotion["mood_zh"])
        cols[1].metric("Valence", f"{affect['valence']:.2f}")
        cols[2].metric("Arousal", f"{affect['arousal']:.2f}")
        cols[3].metric("Mean HR", f"{emotion['features']['mean_heart_rate_bpm']:.0f} bpm")
        cols[4].metric("Grade / cut", f"{style['grade_name']} · {style['beats_per_cut']}b")
        st.caption(emotion["disclaimer"])

    final_video = _read_bytes(report["outputs"]["final_video_mp4"])
    if final_video:
        st.video(final_video)

    dl = st.columns(4)
    stem = Path(report["heartbeat_file"]).stem
    if final_video:
        dl[0].download_button("final_video.mp4", final_video, file_name=f"{stem}_final_video.mp4",
                              mime="video/mp4", key=f"{stem}_mv")
    fa = _read_bytes(report["outputs"]["final_audio_wav"])
    if fa:
        dl[1].download_button("final_audio.wav", fa, file_name=f"{stem}_final_audio.wav",
                              mime="audio/wav", key=f"{stem}_fa")
    fm = _read_bytes(report["outputs"].get("final_audio_mp3"))
    if fm:
        dl[2].download_button("final_audio.mp3", fm, file_name=f"{stem}_final_audio.mp3",
                              mime="audio/mpeg", key=f"{stem}_fm")
    zp = _read_bytes(report["outputs"]["all_outputs_zip"])
    if zp:
        dl[3].download_button("all_outputs.zip", zp, file_name=f"{stem}_all_outputs.zip",
                              mime="application/zip", key=f"{stem}_zip_full")

    with st.expander("Diagnostic report (emotion)"):
        if emotion:
            st.json(emotion)


def full_pipeline_tab(params: ProcessingParams) -> None:
    st.write(
        "Upload one or more **heartbeat** recordings and one **English music video (.mp4)**. "
        "The app keeps the song's original audio, lays the heartbeat under it as a rhythmic bed, "
        "styles the video by the heart's emotion (A), and cuts it to the heartbeat (B)."
    )

    up_cols = st.columns(2)
    heartbeat_files = up_cols[0].file_uploader(
        "Heartbeat recordings (.wav / .mp3)", type=["wav", "mp3"],
        accept_multiple_files=True, key="mv_heartbeats",
    )
    video_file = up_cols[1].file_uploader(
        "Music video (.mp4)", type=["mp4", "mov", "m4v"], accept_multiple_files=False, key="mv_video",
    )

    st.markdown("#### Options")
    o1, o2, o3 = st.columns(3)
    effect_strength = o1.slider("Effect strength", 0.0, 1.5, 0.9, 0.05, key="mv_effect")
    heartbeat_gain_db = o2.slider("Heartbeat mix gain (dB)", -30.0, 0.0, -15.0, 1.0, key="mv_gain")
    duration_limit = o3.number_input("Duration limit (s, 0 = full)", 0.0, 600.0, 0.0, 5.0, key="mv_dur")
    title_text = st.text_input("Title / dedication overlay", value="", key="mv_title")

    f1, f2, f3 = st.columns(3)
    enable_emotion = f1.checkbox("A · Emotion styling", value=True, key="mv_emotion")
    enable_beat_editing = f2.checkbox("B · Heartbeat-driven cuts", value=True, key="mv_beat")
    show_overlay = f3.checkbox("Diagnostic overlay (debug)", value=False, key="mv_overlay",
                               help="Show the heartbeat waveform / BPM HUD. Off for a clean, natural music video.")

    can_run = bool(heartbeat_files) and video_file is not None
    if not can_run:
        st.info("Upload at least one heartbeat file and one music video to enable generation.")
        return

    if st.button("Generate heartbeat music video", type="primary", key="mv_run"):
        with st.spinner("Rendering… this can take a while for long videos."):
            reports = run_full_pipeline(
                heartbeat_files, video_file, params, effect_strength, heartbeat_gain_db,
                title_text, enable_emotion, enable_beat_editing, show_overlay,
                duration_limit if duration_limit > 0 else None,
            )
        st.session_state["mv_reports"] = reports

    reports = st.session_state.get("mv_reports", [])
    if reports:
        st.success(f"Generated {len(reports)} music video(s).")
        for report in reports:
            with st.expander(Path(report["heartbeat_file"]).name, expanded=len(reports) == 1):
                render_pipeline_report(report)


def main() -> None:
    st.title("LegaSynth · Therapist Co-Composition Studio")
    st.caption("Heartbeat-informed musical prompts for therapist-patient co-creation")
    params = sidebar_params()
    save_outputs = st.sidebar.checkbox("Also save outputs to D:\\Heartbeat\\outputs", value=True)

    tab_phrases, tab_stage1, tab_mv = st.tabs(
        ["Musical idea candidates", "Heartbeat timing analysis", "Legacy MV experiment"]
    )
    with tab_phrases:
        phrase_candidates_tab(params)
    with tab_stage1:
        stage1_tab(params, save_outputs)
    with tab_mv:
        full_pipeline_tab(params)


if __name__ == "__main__":
    main()
