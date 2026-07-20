from __future__ import annotations

import hashlib
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from heartbeat_preprocessor.core import ProcessingParams, process_audio_bytes
from music_processor.core import MixParams, RegionEdit, analyze_song_bytes, process_music_bytes


MAX_HEARTBEAT_BYTES = 25 * 1024 * 1024
MAX_HEARTBEAT_DURATION_SECONDS = 30.0
MAX_SONG_BYTES = 100 * 1024 * 1024
MAX_SONG_DURATION_SECONDS = 5.0 * 60.0

PULSE_LABELS = {
    "auto": "自动匹配心率",
    "bar": "每小节一次",
    "half": "每两拍一次",
    "normal": "每拍一次",
    "double": "每拍两次",
    "mute": "该区域不放置心跳",
    "inherit": "继承全局设置",
}
FIT_LABELS = {
    "gap": "保留原心音，间隔补静音",
    "stretch": "拉伸到每个节拍区间",
    "inherit": "继承全局设置",
}


st.set_page_config(
    page_title="Heartbeat Music Processor",
    page_icon="🎚️",
    layout="wide",
)


def sidebar_denoising_params() -> ProcessingParams:
    st.sidebar.header("心跳预处理")
    profile = st.sidebar.select_slider(
        "降噪强度",
        options=["Mild", "Balanced", "Strong"],
        value="Balanced",
        format_func={"Mild": "轻度", "Balanced": "均衡", "Strong": "强力"}.get,
        help="心音变薄时选轻度；摩擦或持续背景声较重时选强力。",
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
        "心跳预处理输出峰值 (dBFS)",
        -24.0,
        -0.1,
        -1.9,
        0.1,
        help="只控制预处理文件的导出电平，不改变降噪强度。",
    )
    st.sidebar.caption(
        "预处理只衰减不一致能量，不会把模板复制或合成到心跳中。"
    )
    return ProcessingParams(export_peak_dbfs=export_peak_dbfs, **profiles[profile])


def upload_signature(heartbeat_name: str, heartbeat_data: bytes, song_name: str, song_data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(heartbeat_name.encode("utf-8", errors="replace"))
    digest.update(heartbeat_data)
    digest.update(song_name.encode("utf-8", errors="replace"))
    digest.update(song_data)
    return digest.hexdigest()


def optional_number(label: str, enabled_label: str, default: float, **kwargs: object) -> float | None:
    enabled = st.checkbox(enabled_label, value=False)
    value = st.number_input(label, value=float(default), disabled=not enabled, **kwargs)
    return float(value) if enabled else None


def empty_region_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "区域名称": pd.Series(dtype="str"),
            "开始时间(秒)": pd.Series(dtype="float"),
            "结束时间(秒)": pd.Series(dtype="float"),
            "歌曲增益(dB)": pd.Series(dtype="float"),
            "心跳增益(dB)": pd.Series(dtype="float"),
            "心跳密度": pd.Series(dtype="str"),
            "周期适配": pd.Series(dtype="str"),
            "边界淡化(ms)": pd.Series(dtype="float"),
        }
    )


def parse_region_table(frame: pd.DataFrame, duration: float) -> list[RegionEdit]:
    edits: list[RegionEdit] = []
    for index, row in frame.iterrows():
        start = row.get("开始时间(秒)")
        end = row.get("结束时间(秒)")
        if pd.isna(start) and pd.isna(end):
            continue
        if pd.isna(start) or pd.isna(end):
            raise ValueError(f"第 {index + 1} 个区域必须同时填写开始和结束时间。")
        label = row.get("区域名称")
        pulse = row.get("心跳密度")
        fit = row.get("周期适配")
        edits.append(
            RegionEdit(
                start_seconds=float(start),
                end_seconds=float(end),
                label="区域" if pd.isna(label) or not str(label).strip() else str(label).strip(),
                song_gain_db=0.0 if pd.isna(row.get("歌曲增益(dB)")) else float(row["歌曲增益(dB)"]),
                heartbeat_gain_db=0.0 if pd.isna(row.get("心跳增益(dB)")) else float(row["心跳增益(dB)"]),
                pulse_mode="inherit" if pd.isna(pulse) else str(pulse),
                fit_mode="inherit" if pd.isna(fit) else str(fit),
                fade_ms=80.0 if pd.isna(row.get("边界淡化(ms)")) else float(row["边界淡化(ms)"]),
            )
        )
    for edit in edits:
        if edit.start_seconds < 0 or edit.end_seconds > duration + 1e-6:
            raise ValueError(f"区域“{edit.label}”必须位于 0–{duration:.2f} 秒内。")
    return edits


def render_timeline(song_analysis: dict, edits: list[RegionEdit]) -> None:
    audio = np.asarray(song_analysis["audio"], dtype=np.float32)
    sample_rate = int(song_analysis["sample_rate"])
    mono = np.mean(audio, axis=1)
    step = max(1, len(mono) // 10000)
    indices = np.arange(0, len(mono), step)
    figure, axis = plt.subplots(figsize=(15, 3.4))
    axis.plot(indices / sample_rate, mono[indices], color="#4063d8", linewidth=0.45, alpha=0.8)
    beats = np.asarray(song_analysis["beat_grid_times_seconds"])
    for index, beat in enumerate(beats):
        axis.axvline(
            beat,
            color="#f3a712" if index % 4 == 0 else "#9aa5b1",
            linewidth=0.7 if index % 4 == 0 else 0.35,
            alpha=0.55,
        )
    colors = ["#ef476f", "#06d6a0", "#8338ec", "#ff7b00", "#118ab2"]
    for index, edit in enumerate(edits):
        axis.axvspan(
            edit.start_seconds,
            edit.end_seconds,
            color=colors[index % len(colors)],
            alpha=0.18,
            label=edit.label,
        )
    axis.set_xlim(0, float(song_analysis["duration_seconds"]))
    axis.set_xlabel("时间（秒）")
    axis.set_ylabel("歌曲波形")
    axis.set_title("歌曲时间线：橙色为每 4 拍参考线，彩色区域为局部编辑")
    if edits:
        axis.legend(loc="upper right", ncols=min(4, len(edits)))
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def show_analysis(heartbeat_result: dict, song_analysis: dict) -> None:
    heartbeat_quality = heartbeat_result["recording_quality"]
    metrics = st.columns(6)
    metrics[0].metric("心跳 BPM", f"{heartbeat_result['summary']['tempo']['estimated_bpm']:.1f}")
    metrics[1].metric("心跳质量", f"{heartbeat_quality['score']:.0f}/100")
    metrics[2].metric("歌曲 BPM", f"{song_analysis['estimated_bpm']:.1f}")
    metrics[3].metric("首拍位置", f"{song_analysis['first_beat_seconds']:.3f}s")
    metrics[4].metric("节拍置信度", f"{song_analysis['beat_tracking_confidence']:.0%}")
    metrics[5].metric("歌曲时长", f"{song_analysis['duration_seconds']:.1f}s")

    if heartbeat_quality["needs_rerecording"]:
        st.error("心跳录音建议重新录制：" + " ".join(heartbeat_quality["rerecord_reasons"]))
    elif heartbeat_quality["denoising_status"] == "limited":
        st.warning("心跳预处理结果可谨慎使用：" + " ".join(heartbeat_quality["reasons"]))
    else:
        st.success("心跳预处理和节律保持检查通过。")
    for warning in song_analysis["warnings"]:
        st.warning(warning)

    left, middle, right = st.columns(3)
    left.write("原始心跳")
    left.audio(heartbeat_result["artifacts"]["input_reference.wav"], format="audio/wav")
    middle.write("预处理后心跳")
    middle.audio(heartbeat_result["artifacts"]["cleaned.wav"], format="audio/wav")
    right.write("送入音乐处理器的真实连续周期")
    right.audio(heartbeat_result["artifacts"]["cleanest_heartbeat_loop.wav"], format="audio/wav")


def render_outputs(result: dict) -> None:
    report = result["report"]
    st.subheader("渲染结果")
    metrics = st.columns(5)
    metrics[0].metric("渲染时长", f"{result['duration_seconds']:.1f}s")
    metrics[1].metric("心跳事件", str(report["render"]["pulse_count"]))
    metrics[2].metric("实际心跳模式", PULSE_LABELS[report["render"]["pulse_mode_resolved"]])
    metrics[3].metric("最终响度", f"{report['master']['output_lufs']:.1f} LUFS")
    metrics[4].metric("最终峰值", f"{report['master']['output_peak_dbfs']:.2f} dBFS")

    st.write("最终混音")
    st.audio(result["artifacts"]["final_mix.wav"], format="audio/wav")
    tracks = st.columns(3)
    tracks[0].write("对齐后的独立心跳轨")
    tracks[0].audio(result["artifacts"]["heartbeat_aligned.wav"], format="audio/wav")
    tracks[1].write("处理后的歌曲轨")
    tracks[1].audio(result["artifacts"]["song_processed.wav"], format="audio/wav")
    tracks[2].write("节拍点击检查轨")
    tracks[2].audio(result["artifacts"]["debug_click_mix.wav"], format="audio/wav")

    downloads = st.columns(4)
    downloads[0].download_button(
        "下载最终音乐 WAV",
        result["artifacts"]["final_mix.wav"],
        file_name="final_heartbeat_music.wav",
        mime="audio/wav",
        type="primary",
    )
    downloads[1].download_button(
        "下载独立心跳轨",
        result["artifacts"]["heartbeat_aligned.wav"],
        file_name="heartbeat_aligned.wav",
        mime="audio/wav",
    )
    downloads[2].download_button(
        "下载处理报告",
        result["artifacts"]["mix_report.json"],
        file_name="mix_report.json",
        mime="application/json",
    )
    downloads[3].download_button(
        "下载全部工程文件",
        result["zip_bytes"],
        file_name="heartbeat_music_project.zip",
        mime="application/zip",
    )
    with st.expander("完整处理参数和技术报告"):
        st.json(report)


def main() -> None:
    st.title("🎚️ Heartbeat Music Processor")
    st.write(
        "上传一段心跳 WAV 和一首歌曲 WAV：网页会完成心跳预处理、歌曲节拍分析、"
        "心跳对齐、分段编辑、响度平衡和最终整首渲染。"
    )
    st.caption(
        "文件会上传到 Streamlit 服务器内存中处理；本程序不会主动持久化保存上传音频或生成结果。"
    )
    denoising_params = sidebar_denoising_params()

    st.header("1. 上传两路音频")
    heartbeat_column, song_column = st.columns(2)
    heartbeat_upload = heartbeat_column.file_uploader(
        "心跳录音 WAV",
        type=["wav"],
        accept_multiple_files=False,
        help="建议约 15 秒；最大 25 MB、30 秒。",
    )
    song_upload = song_column.file_uploader(
        "目标歌曲 WAV",
        type=["wav"],
        accept_multiple_files=False,
        help="最大 100 MB、5 分钟。保留原歌曲声道和采样率。",
    )
    if heartbeat_upload is None or song_upload is None:
        st.info("请同时上传心跳 WAV 和歌曲 WAV。")
        return

    heartbeat_data = heartbeat_upload.getvalue()
    song_data = song_upload.getvalue()
    if len(heartbeat_data) > MAX_HEARTBEAT_BYTES:
        st.error("心跳 WAV 超过 25 MB。")
        return
    if len(song_data) > MAX_SONG_BYTES:
        st.error("歌曲 WAV 超过 100 MB。")
        return
    signature = upload_signature(
        heartbeat_upload.name,
        heartbeat_data,
        song_upload.name,
        song_data,
    )

    st.header("2. 分析心跳与歌曲")
    analysis_controls = st.columns(3)
    with analysis_controls[0]:
        manual_bpm = optional_number(
            "歌曲 BPM",
            "手动指定歌曲 BPM",
            120.0,
            min_value=20.0,
            max_value=300.0,
            step=0.1,
        )
    with analysis_controls[1]:
        manual_first_beat = optional_number(
            "第一拍时间（秒）",
            "手动指定歌曲第一拍",
            0.0,
            min_value=0.0,
            step=0.01,
        )
    with analysis_controls[2]:
        force_constant = st.checkbox(
            "使用固定 BPM 网格",
            value=False,
            help="关闭时保留检测到的局部速度变化；指定 BPM 时会自动使用固定网格。",
        )

    if st.button("分析两路音频", type="primary", width="stretch"):
        try:
            with st.spinner("正在预处理心跳并分析歌曲节拍……"):
                heartbeat_result = process_audio_bytes(
                    heartbeat_upload.name,
                    heartbeat_data,
                    denoising_params,
                    max_duration_seconds=MAX_HEARTBEAT_DURATION_SECONDS,
                )
                song_analysis = analyze_song_bytes(
                    song_upload.name,
                    song_data,
                    manual_bpm=manual_bpm,
                    manual_first_beat=manual_first_beat,
                    force_constant_grid=force_constant,
                )
                if song_analysis["duration_seconds"] > MAX_SONG_DURATION_SECONDS:
                    raise ValueError("歌曲超过 5 分钟的网页处理限制。")
                st.session_state["processor_analysis"] = {
                    "signature": signature,
                    "heartbeat": heartbeat_result,
                    "song": song_analysis,
                }
                st.session_state.pop("processor_render", None)
            st.success("分析完成。请检查节拍和心跳质量，然后调整处理参数。")
        except Exception as exc:
            st.error(f"分析失败：{exc}")

    saved = st.session_state.get("processor_analysis")
    if saved is None or saved.get("signature") != signature:
        return
    heartbeat_result = saved["heartbeat"]
    song_analysis = saved["song"]
    show_analysis(heartbeat_result, song_analysis)

    st.header("3. 全局音乐处理参数")
    duration = float(song_analysis["duration_seconds"])
    row1 = st.columns(4)
    pulse_mode = row1[0].selectbox(
        "全局心跳密度",
        ["auto", "bar", "half", "normal", "double"],
        format_func=PULSE_LABELS.get,
    )
    fit_mode = row1[1].selectbox(
        "周期适配方式",
        ["gap", "stretch"],
        format_func=FIT_LABELS.get,
    )
    beats_per_bar = row1[2].number_input("每小节拍数", 1, 12, 4, 1)
    heartbeat_range = row1[3].slider(
        "心跳出现范围（秒）",
        0.0,
        duration,
        (0.0, duration),
        step=max(0.01, min(0.1, duration / 1000.0)),
    )

    row2 = st.columns(4)
    song_gain_db = row2[0].slider("歌曲整体增益 (dB)", -18.0, 12.0, 0.0, 0.5)
    heartbeat_gain_db = row2[1].slider("心跳整体增益 (dB)", -24.0, 24.0, 0.0, 0.5)
    auto_balance = row2[2].checkbox("自动响度平衡", value=True)
    heartbeat_relative_lu = row2[3].slider(
        "心跳相对响度 (LU)", -12.0, 12.0, 1.0, 0.5,
        help="正值让心跳比歌曲活跃段更突出。",
    )

    with st.expander("高级混音与母带参数"):
        advanced = st.columns(5)
        song_target_lufs = advanced[0].slider("歌曲目标 LUFS", -24.0, -10.0, -18.0, 0.5)
        ducking_db = advanced[1].slider("心跳触发低频闪避 (dB)", 0.0, 9.0, 2.5, 0.5)
        ducking_cutoff = advanced[2].slider("低频闪避截止 (Hz)", 80.0, 600.0, 280.0, 10.0)
        master_target = advanced[3].slider("最终目标 LUFS", -24.0, -10.0, -16.0, 0.5)
        ceiling = advanced[4].slider("输出峰值上限 (dBFS)", -6.0, -0.1, -1.0, 0.1)

    st.header("4. 对特定时间段进行编辑")
    st.write(
        "点击表格底部的 `+` 添加区域。每个区域可单独调整歌曲音量、心跳音量、"
        "心跳密度和周期适配；区域不能互相重叠。把心跳密度设为 `mute` 可制作无心跳段落。"
    )
    region_frame = st.data_editor(
        empty_region_table(),
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        column_config={
            "区域名称": st.column_config.TextColumn(default="区域"),
            "开始时间(秒)": st.column_config.NumberColumn(min_value=0.0, max_value=duration, step=0.1),
            "结束时间(秒)": st.column_config.NumberColumn(min_value=0.0, max_value=duration, step=0.1),
            "歌曲增益(dB)": st.column_config.NumberColumn(default=0.0, min_value=-60.0, max_value=18.0, step=0.5),
            "心跳增益(dB)": st.column_config.NumberColumn(default=0.0, min_value=-60.0, max_value=30.0, step=0.5),
            "心跳密度": st.column_config.SelectboxColumn(
                options=["inherit", "bar", "half", "normal", "double", "mute"],
                default="inherit",
            ),
            "周期适配": st.column_config.SelectboxColumn(
                options=["inherit", "gap", "stretch"],
                default="inherit",
            ),
            "边界淡化(ms)": st.column_config.NumberColumn(default=80.0, min_value=0.0, max_value=5000.0, step=10.0),
        },
        key="music_region_editor",
    )
    try:
        region_edits = parse_region_table(region_frame, duration)
        render_timeline(song_analysis, region_edits)
    except Exception as exc:
        st.error(f"区域设置无效：{exc}")
        return

    quality_override = False
    if heartbeat_result["recording_quality"]["needs_rerecording"]:
        quality_override = st.checkbox(
            "我已试听并确认仍要使用这段心跳生成音乐",
            value=False,
        )

    mix_params = MixParams(
        pulse_mode=pulse_mode,
        fit_mode=fit_mode,
        beats_per_bar=int(beats_per_bar),
        heartbeat_start_seconds=float(heartbeat_range[0]),
        heartbeat_end_seconds=float(heartbeat_range[1]),
        song_gain_db=float(song_gain_db),
        heartbeat_gain_db=float(heartbeat_gain_db),
        auto_balance=bool(auto_balance),
        song_target_lufs=float(song_target_lufs),
        heartbeat_relative_lu=float(heartbeat_relative_lu),
        ducking_db=float(ducking_db),
        ducking_cutoff_hz=float(ducking_cutoff),
        master_target_lufs=float(master_target),
        output_ceiling_dbfs=float(ceiling),
    )

    st.header("5. 试听或生成最终音乐")
    preview_length = min(45.0, duration)
    buttons = st.columns(3)
    preview_clicked = buttons[0].button(
        f"生成前 {preview_length:.0f} 秒试听",
        width="stretch",
        disabled=heartbeat_result["recording_quality"]["needs_rerecording"] and not quality_override,
    )
    final_clicked = buttons[1].button(
        "生成整首最终音乐",
        type="primary",
        width="stretch",
        disabled=heartbeat_result["recording_quality"]["needs_rerecording"] and not quality_override,
    )
    if buttons[2].button("清除分析和渲染结果", width="stretch"):
        st.session_state.pop("processor_analysis", None)
        st.session_state.pop("processor_render", None)
        st.rerun()

    if preview_clicked or final_clicked:
        try:
            label = "试听" if preview_clicked else "整首歌曲"
            with st.spinner(f"正在渲染{label}……"):
                result = process_music_bytes(
                    song_upload.name,
                    song_data,
                    heartbeat_result,
                    song_analysis,
                    mix_params,
                    region_edits,
                    render_duration_seconds=preview_length if preview_clicked else None,
                )
                st.session_state["processor_render"] = {
                    "signature": signature,
                    "result": result,
                }
            st.success(f"{label}渲染完成。")
        except Exception as exc:
            st.error(f"渲染失败：{exc}")

    rendered = st.session_state.get("processor_render")
    if rendered is not None and rendered.get("signature") == signature:
        render_outputs(rendered["result"])


if __name__ == "__main__":
    main()
