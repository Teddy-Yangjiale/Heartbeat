from __future__ import annotations

import streamlit as st

from heartbeat_preprocessor.core import ProcessingParams, process_audio_bytes


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_DURATION_SECONDS = 30.0


st.set_page_config(page_title="Heartbeat WAV Denoiser", layout="wide")


def sidebar_params() -> ProcessingParams:
    st.sidebar.header("Denoising")
    profile = st.sidebar.select_slider(
        "Strength",
        options=["Mild", "Balanced", "Strong"],
        value="Balanced",
        help="Use Mild when S1/S2 sounds thin. Strong is intended for clearly contaminated recordings.",
    )
    profiles = {
        "Mild": {
            "spectral_reduction_strength": 0.80,
            "spectral_floor_db": -24.0,
            "cycle_outlier_attenuation_db": -12.0,
            "between_beat_attenuation_db": -18.0,
        },
        "Balanced": {
            "spectral_reduction_strength": 1.00,
            "spectral_floor_db": -30.0,
            "cycle_outlier_attenuation_db": -18.0,
            "between_beat_attenuation_db": -28.0,
        },
        "Strong": {
            "spectral_reduction_strength": 1.25,
            "spectral_floor_db": -36.0,
            "cycle_outlier_attenuation_db": -24.0,
            "between_beat_attenuation_db": -36.0,
        },
    }
    export_peak_dbfs = st.sidebar.slider(
        "Output peak (dBFS)",
        -24.0,
        -0.1,
        -1.9,
        0.1,
        help="Changes export level only; it does not make the denoising more aggressive.",
    )
    st.sidebar.caption(
        "The algorithm learns repetition only within this WAV. It attenuates inconsistent energy and never copies a heartbeat template into the output."
    )
    st.sidebar.caption(
        "Privacy: uploads and generated outputs stay in this browser session and are not saved by the web app."
    )
    return ProcessingParams(export_peak_dbfs=export_peak_dbfs, **profiles[profile])


def format_metric(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def render_result(result: dict) -> None:
    summary = result["summary"]
    tempo = summary["tempo"]
    quality = summary["quality"]
    recording = summary["recording_quality"]
    cycle = result["cycle_consistency"]
    rhythm = result["rhythm_preservation"]
    focal = result["focal_cycle_contamination"]

    metrics = st.columns(7)
    metrics[0].metric("Duration", f"{summary['duration_seconds']:.2f}s")
    metrics[1].metric("Estimated BPM", f"{tempo['estimated_bpm']:.1f}")
    metrics[2].metric(
        "Cycles before / after",
        f"{rhythm['expected_beat_count']} / {rhythm['processed_beat_count']}",
    )
    metrics[3].metric(
        "Rhythm match",
        f"{rhythm['matched_fraction']:.0%}",
    )
    metrics[4].metric(
        "Inter-beat reduction",
        format_metric(quality["interbeat_noise_reduction_db"], " dB"),
    )
    metrics[5].metric(
        "S1/S2 preservation",
        format_metric(quality["heartbeat_preservation_correlation"]),
    )
    metrics[6].metric("Quality", f"{recording['score']:.0f}/100", recording["grade"])

    if recording["needs_rerecording"]:
        st.error("Re-recording is recommended. " + " ".join(recording["rerecord_reasons"]))
    elif recording["denoising_status"] == "limited":
        st.warning("The result is usable with caution. " + " ".join(recording["reasons"]))
    else:
        st.success("The recording passed the automated preservation and consistency checks.")

    st.caption(
        f"Independent post-denoising rhythm check: {rhythm['matched_beat_count']}/"
        f"{rhythm['expected_beat_count']} beats matched; count delta {rhythm['count_delta']}; "
        f"median timing error {format_metric(rhythm['median_timing_error_ms'], ' ms')}."
    )
    if focal["applied"]:
        st.caption(
            f"Focal contamination check: {focal['severe_cycle_count']} severe cycle(s); "
            f"maximum RMS ratio {format_metric(focal['max_rms_ratio'])}; "
            f"maximum peak ratio {format_metric(focal['max_peak_ratio'])}."
        )

    if cycle["applied"]:
        st.caption(
            f"Cycle consistency used {cycle['cycles_used']} complete cycles; "
            f"median correlation {cycle['median_cycle_correlation']:.3f}; "
            f"non-repeating transient coverage {cycle['outlier_fraction']:.1%}."
        )
    else:
        st.caption("Cycle consistency was not applied: " + str(cycle["reason"]))

    audio = st.columns(3)
    audio[0].write("Before: input reference")
    audio[0].audio(result["artifacts"]["input_reference.wav"], format="audio/wav")
    audio[1].write("Intermediate: cycle-consistent signal")
    audio[1].audio(result["artifacts"]["filtered_detection.wav"], format="audio/wav")
    audio[2].write("After: final denoised heartbeat")
    audio[2].audio(result["artifacts"]["cleaned.wav"], format="audio/wav")

    segment = result["cleanest_segment"]
    st.subheader("Cleanest consecutive heartbeat segment")
    st.caption(
        f"Selected {segment['cycle_count']} real cycles from "
        f"{segment['adjusted_start_seconds']:.2f}s to {segment['adjusted_end_seconds']:.2f}s; "
        f"quality score {segment['quality_score']:.1f}/100. No template audio is copied into this segment."
    )
    loop_audio = st.columns(2)
    loop_audio[0].write("Faithful clean loop")
    loop_audio[0].audio(result["artifacts"]["cleanest_heartbeat_loop.wav"], format="audio/wav")
    loop_audio[1].write("Playback-loud loop")
    loop_audio[1].audio(result["artifacts"]["cleanest_heartbeat_loop_loud.wav"], format="audio/wav")

    st.image(result["artifacts"]["diagnostic_plot.png"], caption="Denoising diagnostics", width="stretch")

    downloads = st.columns(6)
    downloads[0].download_button(
        "Download all outputs",
        result["zip_bytes"],
        file_name=f"{result['stem']}_denoising_outputs.zip",
        mime="application/zip",
    )
    downloads[1].download_button(
        "Download cleaned.wav",
        result["artifacts"]["cleaned.wav"],
        file_name=f"{result['stem']}_cleaned.wav",
        mime="audio/wav",
    )
    downloads[2].download_button(
        "Download quality report",
        result["artifacts"]["recording_quality.json"],
        file_name=f"{result['stem']}_recording_quality.json",
        mime="application/json",
    )
    downloads[3].download_button(
        "Download diagnostic plot",
        result["artifacts"]["diagnostic_plot.png"],
        file_name=f"{result['stem']}_diagnostic_plot.png",
        mime="image/png",
    )
    downloads[4].download_button(
        "Download clean loop",
        result["artifacts"]["cleanest_heartbeat_loop.wav"],
        file_name=f"{result['stem']}_cleanest_heartbeat_loop.wav",
        mime="audio/wav",
    )
    downloads[5].download_button(
        "Download loud loop",
        result["artifacts"]["cleanest_heartbeat_loop_loud.wav"],
        file_name=f"{result['stem']}_cleanest_heartbeat_loop_loud.wav",
        mime="audio/wav",
    )

    with st.expander("Technical details"):
        st.json(summary)


def main() -> None:
    st.title("Heartbeat WAV Denoiser")
    st.write(
        "Upload one approximately 15-second heartbeat WAV. The processor reduces persistent background noise "
        "and non-repeating friction while preserving recurrent S1/S2 energy."
    )
    st.caption("This is an audio preprocessor, not a medical diagnostic system.")

    params = sidebar_params()
    uploaded = st.file_uploader(
        "Upload one heartbeat WAV",
        type=["wav"],
        accept_multiple_files=False,
        help="Maximum 25 MB and 30 seconds. The web app processes the recording in memory.",
    )
    if uploaded is None:
        st.info("A mono or stereo PCM WAV is accepted. Stereo input is converted to mono.")
        return

    uploaded_bytes = uploaded.getvalue()
    if len(uploaded_bytes) > MAX_UPLOAD_BYTES:
        st.error("The WAV is larger than 25 MB. Please upload an approximately 15-second recording.")
        return

    if st.button("Denoise heartbeat", type="primary"):
        try:
            result = process_audio_bytes(
                uploaded.name,
                uploaded_bytes,
                params,
                max_duration_seconds=MAX_DURATION_SECONDS,
            )
            st.session_state["denoising_result"] = result
            st.success("Processing complete. Results are available below and have not been saved on the server.")
        except Exception as exc:
            st.error(f"Failed to process the WAV: {exc}")
            return

    result = st.session_state.get("denoising_result")
    if result is not None:
        if st.button("Clear result from this session"):
            del st.session_state["denoising_result"]
            st.rerun()
        render_result(result)


if __name__ == "__main__":
    main()
