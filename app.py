from __future__ import annotations

import hashlib
import html
import logging
import math
import shutil
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from heartbeat_preprocessor.core import ProcessingParams, process_audio_bytes
from music_processor.core import (
    STYLE_PRESETS,
    MixParams,
    RegionEdit,
    analyze_song_bytes,
    get_style_preset,
    process_music_bytes,
)
from music_processor.sync_adapter import (
    adapt_sync_result,
    build_sync_command,
    discover_sync_service,
    run_sync_cli,
)


MAX_HEARTBEAT_BYTES = 25 * 1024 * 1024
MAX_HEARTBEAT_DURATION_SECONDS = 30.0
MAX_SONG_BYTES = 100 * 1024 * 1024
MAX_SONG_DURATION_SECONDS = 5.0 * 60.0
JOB_ROOT = Path(tempfile.gettempdir()) / "heartbeat_music_jobs"
RENDER_SEMAPHORE = threading.BoundedSemaphore(1)
LOGGER = logging.getLogger(__name__)

PULSE_LABELS = {
    "auto": "音乐感知自动模式",
    "adaptive": "局部自适应",
    "downbeat": "只在小节重拍",
    "kick": "底鼓角色（通常 1/3 拍）",
    "backbeat": "反拍角色（通常 2/4 拍）",
    "every-beat": "每拍一次",
    "bar": "每小节一次",
    "half": "每两拍一次",
    "normal": "每拍一次",
    "double": "每拍两次",
    "mute": "该区域不放置心跳",
    "inherit": "继承全局设置",
}
FIT_LABELS = {
    "preserve": "保留 S1/S2、静息段留白（推荐）",
    "gap": "保留完整原始周期",
    "stretch": "受限时间拉伸",
    "inherit": "继承全局设置",
}


def initialize_job_root() -> Path:
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - 6 * 60 * 60
    for child in JOB_ROOT.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child)
        except OSError:
            continue
    return JOB_ROOT


def compact_heartbeat_result(result: dict) -> dict:
    preview_names = {
        "input_reference.wav",
        "cleaned.wav",
        "cleanest_heartbeat_loop.wav",
        "heartbeat_cycle_pool_preview.wav",
        "heartbeat_detection_mix.wav",
    }
    return {
        key: result[key]
        for key in (
            "name",
            "sample_rate",
            "beat_times",
            "summary",
            "recording_quality",
            "cleanest_segment",
            "cleanest_audio",
            "cycle_pool",
            "s1_times",
            "s2_times",
        )
    } | {
        "artifacts": {
            name: value
            for name, value in result["artifacts"].items()
            if name in preview_names
        }
    }


def cleanup_render_result(saved: dict | None) -> None:
    if not saved:
        return
    output_dir = saved.get("output_dir")
    if not output_dir:
        return
    path = Path(output_dir).resolve()
    root = JOB_ROOT.resolve()
    if root in path.parents and path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def deferred_file(path: str):
    return lambda: open(path, "rb")


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
            "phase_noise_reduction_strength": 0.65,
        },
        "Balanced": {
            "spectral_reduction_strength": 1.00,
            "spectral_floor_db": -30.0,
            "cycle_outlier_attenuation_db": -18.0,
            "between_beat_attenuation_db": -28.0,
            "phase_noise_reduction_strength": 0.90,
        },
        "Strong": {
            "spectral_reduction_strength": 1.25,
            "spectral_floor_db": -36.0,
            "cycle_outlier_attenuation_db": -24.0,
            "between_beat_attenuation_db": -36.0,
            "phase_noise_reduction_strength": 1.10,
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
    enable_hum = st.sidebar.checkbox(
        "自动抑制 50/60 Hz 工频嗡声",
        value=True,
        help="只有检测到持续、窄带且显著的工频峰时才启用陷波。",
    )
    st.sidebar.caption(
        "预处理只衰减不一致能量，不会把模板复制或合成到心跳中。"
    )
    return ProcessingParams(
        export_peak_dbfs=export_peak_dbfs,
        enable_hum_suppression=enable_hum,
        **profiles[profile],
    )


def upload_signature(
    heartbeat_name: str,
    heartbeat_data: bytes | memoryview,
    song_name: str,
    song_data: bytes | memoryview,
) -> str:
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
            "听感偏移(ms)": pd.Series(dtype="float"),
            "自然波动(ms)": pd.Series(dtype="float"),
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
                timing_offset_ms=(
                    None if pd.isna(row.get("听感偏移(ms)")) else float(row["听感偏移(ms)"])
                ),
                humanize_ms=(
                    None if pd.isna(row.get("自然波动(ms)")) else float(row["自然波动(ms)"])
                ),
            )
        )
    for edit in edits:
        if edit.start_seconds < 0 or edit.end_seconds > duration + 1e-6:
            raise ValueError(f"区域“{edit.label}”必须位于 0–{duration:.2f} 秒内。")
    return edits


def build_timeline_svg(song_analysis: dict, edits: list[RegionEdit]) -> str:
    times = np.asarray(song_analysis["waveform_overview_times_seconds"], dtype=np.float64)
    waveform = np.asarray(song_analysis["waveform_overview_values"], dtype=np.float32)
    duration = max(float(song_analysis["duration_seconds"]), 1e-6)
    width, height = 1200.0, 300.0
    left, right, top, bottom = 58.0, 18.0, 22.0, 42.0
    plot_width = width - left - right
    plot_height = height - top - bottom

    max_points = 2400
    if len(times) and len(waveform):
        length = min(len(times), len(waveform))
        step = max(1, int(math.ceil(length / max_points)))
        sampled_times = times[:length:step]
        sampled_waveform = waveform[:length:step]
        peak = max(float(np.max(np.abs(sampled_waveform))), 1e-6)
        points = " ".join(
            f"{left + float(t) / duration * plot_width:.2f},"
            f"{top + plot_height / 2.0 - float(value) / peak * plot_height * 0.46:.2f}"
            for t, value in zip(sampled_times, sampled_waveform)
        )
    else:
        points = ""

    beats = np.asarray(song_analysis["beat_grid_times_seconds"], dtype=np.float64)
    downbeats = np.sort(
        np.asarray(song_analysis.get("downbeat_times_seconds", []), dtype=np.float64)
    )

    def is_downbeat(beat: float) -> bool:
        if not len(downbeats):
            return False
        index = int(np.searchsorted(downbeats, beat))
        candidates = downbeats[max(0, index - 1) : min(len(downbeats), index + 1)]
        return bool(len(candidates) and np.min(np.abs(candidates - beat)) < 0.04)

    region_parts: list[str] = []
    colors = ["#ef476f", "#06d6a0", "#8338ec", "#ff7b00", "#118ab2"]
    for index, edit in enumerate(edits):
        start_x = left + edit.start_seconds / duration * plot_width
        end_x = left + edit.end_seconds / duration * plot_width
        color = colors[index % len(colors)]
        label = html.escape(edit.label)
        region_parts.append(
            f'<rect x="{start_x:.2f}" y="{top:.2f}" width="{max(0.0, end_x - start_x):.2f}" '
            f'height="{plot_height:.2f}" fill="{color}" opacity="0.16"><title>{label}</title></rect>'
        )

    beat_parts: list[str] = []
    for beat in beats:
        if beat < 0.0 or beat > duration:
            continue
        x = left + float(beat) / duration * plot_width
        downbeat = is_downbeat(float(beat))
        beat_parts.append(
            f'<line x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{top + plot_height:.2f}" '
            f'stroke="{"#f3a712" if downbeat else "#9aa5b1"}" '
            f'stroke-width="{1.25 if downbeat else 0.55}" opacity="0.60" />'
        )

    tick_parts: list[str] = []
    for index in range(7):
        seconds = duration * index / 6.0
        x = left + plot_width * index / 6.0
        tick_parts.append(
            f'<line x1="{x:.2f}" y1="{top + plot_height:.2f}" x2="{x:.2f}" '
            f'y2="{top + plot_height + 5.0:.2f}" stroke="#6b7280" />'
            f'<text x="{x:.2f}" y="{height - 14.0:.2f}" text-anchor="middle" '
            f'font-size="12" fill="#6b7280">{seconds:.1f}s</text>'
        )

    return (
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" width="100%" height="300" '
        f'role="img" aria-label="歌曲时间线" xmlns="http://www.w3.org/2000/svg">'
        '<style>text{font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif}</style>'
        f'<rect x="{left:.2f}" y="{top:.2f}" width="{plot_width:.2f}" height="{plot_height:.2f}" '
        'fill="#f8fafc" stroke="#d1d5db" />'
        + "".join(region_parts)
        + "".join(beat_parts)
        + f'<line x1="{left:.2f}" y1="{top + plot_height / 2.0:.2f}" '
        f'x2="{left + plot_width:.2f}" y2="{top + plot_height / 2.0:.2f}" '
        'stroke="#cbd5e1" stroke-width="0.6" />'
        + (f'<polyline points="{points}" fill="none" stroke="#4063d8" stroke-width="1.0" opacity="0.88" />' if points else "")
        + "".join(tick_parts)
        + '<text x="600" y="15" text-anchor="middle" font-size="14" fill="#374151">'
        '歌曲时间线：橙色为小节重拍，灰色为普通节拍，彩色区域为局部编辑</text>'
        + '</svg>'
    )


def render_timeline(song_analysis: dict, edits: list[RegionEdit]) -> None:
    st.html(build_timeline_svg(song_analysis, edits))


def show_analysis(heartbeat_result: dict, song_analysis: dict) -> None:
    heartbeat_quality = heartbeat_result["recording_quality"]
    metrics = st.columns(7)
    metrics[0].metric("心跳 BPM", f"{heartbeat_result['summary']['tempo']['estimated_bpm']:.1f}")
    metrics[1].metric("心跳质量", f"{heartbeat_quality['score']:.0f}/100")
    metrics[2].metric("歌曲 BPM", f"{song_analysis['estimated_bpm']:.1f}")
    metrics[3].metric("首拍位置", f"{song_analysis['first_beat_seconds']:.3f}s")
    metrics[4].metric("节拍置信度", f"{song_analysis['beat_tracking_confidence']:.0%}")
    metrics[5].metric("歌曲时长", f"{song_analysis['duration_seconds']:.1f}s")
    metrics[6].metric(
        "重拍置信度",
        f"{song_analysis.get('downbeat_confidence', 0.0):.0%}",
    )
    trim = song_analysis.get("content_trim", {})
    if trim.get("enabled"):
        st.caption(
            f"歌曲内容裁剪：前端移除 {float(trim.get('removed_leading_seconds', 0.0)):.2f}s，"
            f"尾端移除 {float(trim.get('removed_trailing_seconds', 0.0)):.2f}s；"
            f"用于对齐的有效时长 {song_analysis['duration_seconds']:.2f}s。"
        )

    if heartbeat_quality["needs_rerecording"]:
        st.error("心跳录音建议重新录制：" + " ".join(heartbeat_quality["rerecord_reasons"]))
    elif heartbeat_quality["denoising_status"] == "limited":
        st.warning("心跳预处理结果可谨慎使用：" + " ".join(heartbeat_quality["reasons"]))
    else:
        st.success("心跳预处理和节律保持检查通过。")
    for warning in song_analysis["warnings"]:
        st.warning(warning)

    left, middle, right, diagnostic = st.columns(4)
    left.write("原始心跳")
    left.audio(heartbeat_result["artifacts"]["input_reference.wav"], format="audio/wav")
    middle.write("预处理后心跳")
    middle.audio(heartbeat_result["artifacts"]["cleaned.wav"], format="audio/wav")
    right.write("送入音乐处理器的多周期真实素材池")
    pool_preview = heartbeat_result["artifacts"].get(
        "heartbeat_cycle_pool_preview.wav",
        heartbeat_result["artifacts"]["cleanest_heartbeat_loop.wav"],
    )
    right.audio(pool_preview, format="audio/wav")
    diagnostic.write("S1/S2 检测检查轨")
    diagnostic.audio(
        heartbeat_result["artifacts"]["heartbeat_detection_mix.wav"],
        format="audio/wav",
    )


def render_outputs(result: dict) -> None:
    report = result["report"]
    st.subheader("渲染结果")
    metrics = st.columns(6)
    metrics[0].metric("渲染时长", f"{result['duration_seconds']:.1f}s")
    metrics[1].metric("心跳事件", str(report["render"]["pulse_count"]))
    metrics[2].metric("实际心跳模式", PULSE_LABELS[report["render"]["pulse_mode_resolved"]])
    metrics[3].metric("最终响度", f"{report['master']['output_lufs']:.1f} LUFS")
    metrics[4].metric("最终峰值", f"{report['master']['output_peak_dbfs']:.2f} dBFS")
    metrics[5].metric(
        "峰值内存",
        f"{report.get('memory', {}).get('peak_observed_mb', 0.0):.0f} MB",
    )
    render_report = report["render"]
    st.caption(
        f"声学锚点：{render_report.get('anchor_mode', 'unknown')} · "
        f"模型节拍 {render_report.get('model_backed_pulse_count', 0)} · "
        f"引导脉冲 {render_report.get('guide_pulse_count', 0)} · "
        f"片头/片尾 {render_report.get('effective_intro_pulses', 0)}/"
        f"{render_report.get('outro_pulses', 0)} · "
        f"歌曲偏移 {render_report.get('song_offset_seconds', 0.0):.3f}s · "
        f"最大排程误差 {render_report.get('maximum_anchor_alignment_error_ms', 0.0):.3f} ms"
    )
    warnings = result.get("sync_summary", {}).get(
        "warnings", report.get("warnings", [])
    )
    for warning in list(dict.fromkeys(str(item) for item in warnings if item))[:6]:
        st.warning(warning)

    paths = result.get("artifact_paths", {})
    artifacts = result.get("artifacts", {})

    def media(name: str):
        return paths.get(name, artifacts.get(name))

    output_info = report.get("output", {})
    final_name = str(output_info.get("filename", "final_mix.wav"))
    final_mime = str(output_info.get("mime", "audio/wav"))
    final_mix = media(final_name)
    st.write("最终混音")
    st.audio(final_mix, format=final_mime)
    available_tracks = [
        ("heartbeat_aligned.wav", "对齐后的独立心跳轨"),
        ("song_processed.wav", "处理后的歌曲轨"),
        ("debug_click_mix.wav", "节拍点击检查轨"),
        ("heartbeat_detection_mix.wav", "S1/S2 心跳检测检查轨"),
    ]
    available_tracks = [(name, label) for name, label in available_tracks if media(name) is not None]
    if available_tracks:
        tracks = st.columns(len(available_tracks))
        for column, (name, label) in zip(tracks, available_tracks):
            column.write(label)
            column.audio(media(name), format="audio/wav")

    downloads = st.columns(4)
    downloads[0].download_button(
        "下载最终音乐",
        deferred_file(paths[final_name]) if final_name in paths else final_mix,
        file_name=f"final_heartbeat_music{Path(final_name).suffix}",
        mime=final_mime,
        type="primary",
        on_click="ignore",
    )
    heartbeat_stem = media("heartbeat_aligned.wav")
    downloads[1].download_button(
        "下载独立心跳轨",
        deferred_file(paths["heartbeat_aligned.wav"])
        if "heartbeat_aligned.wav" in paths
        else (heartbeat_stem or b""),
        file_name="heartbeat_aligned.wav",
        mime="audio/wav",
        disabled=heartbeat_stem is None,
        on_click="ignore",
    )
    report_name = "analysis_report.json" if media("analysis_report.json") is not None else "mix_report.json"
    report_data = media(report_name)
    downloads[2].download_button(
        "下载处理报告",
        deferred_file(paths[report_name])
        if report_name in paths
        else report_data,
        file_name=report_name,
        mime="application/json",
        on_click="ignore",
    )
    zip_path = result.get("zip_path")
    downloads[3].download_button(
        "下载全部工程文件",
        deferred_file(zip_path) if zip_path else result.get("zip_bytes", b""),
        file_name="heartbeat_music_project.zip",
        mime="application/zip",
        disabled=not bool(zip_path or result.get("zip_bytes")),
        on_click="ignore",
    )
    with st.expander("完整处理参数和技术报告"):
        st.json(report)


def main() -> None:
    initialize_job_root()
    sync_service = discover_sync_service()
    st.title("🎚️ Heartbeat Music Processor")
    st.write(
        "上传一段心跳 WAV 和一首 WAV/MP3 歌曲：网页会完成心跳预处理、歌曲节拍分析、"
        "心跳对齐、分段编辑、响度平衡和最终整首渲染。"
    )
    st.caption(
        "上传内容只用于当前会话；中间大文件写入临时任务目录，下载按需读取，会话清理后删除。"
    )
    denoising_params = sidebar_denoising_params()

    st.header("1. 上传两路音频")
    heartbeat_column, song_column = st.columns(2)
    heartbeat_upload = heartbeat_column.file_uploader(
        "心跳录音 WAV",
        type=["wav"],
        accept_multiple_files=False,
        max_upload_size=25,
        help="建议约 15 秒；最大 25 MB、30 秒。",
    )
    song_upload = song_column.file_uploader(
        "目标歌曲 WAV / MP3",
        type=["wav", "mp3"],
        accept_multiple_files=False,
        max_upload_size=100,
        help="支持 WAV 和 MP3；最大 100 MB、5 分钟。保留解码后的歌曲声道和采样率。",
    )
    if heartbeat_upload is None or song_upload is None:
        st.info("请同时上传心跳 WAV 和 WAV/MP3 歌曲。")
        return

    heartbeat_data = heartbeat_upload.getbuffer()
    song_data = song_upload.getbuffer()
    if len(heartbeat_data) > MAX_HEARTBEAT_BYTES:
        st.error("心跳 WAV 超过 25 MB。")
        return
    if len(song_data) > MAX_SONG_BYTES:
        st.error("歌曲文件超过 100 MB。")
        return
    signature = upload_signature(
        heartbeat_upload.name,
        heartbeat_data,
        song_upload.name,
        song_data,
    )

    st.header("2. 分析心跳与歌曲")
    analysis_controls = st.columns(5)
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
        manual_first_downbeat = optional_number(
            "第一小节重拍（秒）",
            "手动指定第一小节重拍",
            0.0,
            min_value=0.0,
            step=0.01,
        )
    with analysis_controls[3]:
        analysis_meter = st.number_input("歌曲拍号（每小节拍数）", 2, 12, 4, 1)
    with analysis_controls[4]:
        force_constant = st.checkbox(
            "使用固定 BPM 网格",
            value=False,
            help="关闭时保留检测到的局部速度变化；指定 BPM 时会自动使用固定网格。",
        )
    with st.expander("歌曲内容范围与 heartbeat_sync 后端", expanded=False):
        trim_columns = st.columns(4)
        trim_silence = trim_columns[0].checkbox(
            "自动裁剪首尾静音",
            value=True,
            help="使用迁移文档规定的强/保守双阈值检测，避免把安静前奏误删。",
        )
        trim_top_db = trim_columns[1].slider(
            "静音阈值 top-dB", 10.0, 60.0, 30.0, 1.0
        )
        with trim_columns[2]:
            song_start_seconds = optional_number(
                "手动歌曲起点（秒）",
                "覆盖自动检测的歌曲内容起点",
                0.0,
                min_value=0.0,
                step=0.01,
            )
        with trim_columns[3]:
            song_end_seconds = optional_number(
                "手动歌曲终点（秒）",
                "覆盖自动检测的歌曲内容终点",
                0.0,
                min_value=0.0,
                step=0.01,
            )
        engine_options = ["native"]
        if sync_service is not None:
            engine_options.append("heartbeat_sync_cli")
        processing_engine = st.selectbox(
            "渲染引擎",
            engine_options,
            format_func={
                "native": "网页原生兼容引擎（支持分段编辑）",
                "heartbeat_sync_cli": "官方 heartbeat_sync 黑盒 CLI（Beat This）",
            }.get,
        )
        analysis_backend = st.selectbox(
            "歌曲节拍分析后端",
            ["auto", "librosa"] if processing_engine == "native" else ["auto", "beat-this", "librosa"],
            help="auto 在官方 CLI 中优先 Beat This 并可回退；网页原生引擎使用 librosa。",
        )
        beat_device = st.selectbox(
            "Beat This 设备",
            ["auto", "cpu", "cuda"],
            disabled=processing_engine == "native",
        )
        if sync_service is None:
            st.caption(
                "当前部署未配置 Windows heartbeat_sync 服务，因此使用 Linux 可运行的兼容引擎；"
                "设置 HEARTBEAT_SYNC_REPO 后会自动出现官方 CLI 选项。"
            )
        elif processing_engine == "heartbeat_sync_cli":
            st.warning(
                "官方 CLI 严格按迁移合同运行，但不接受网页的分段风格编辑；"
                "需要分段编辑时请选择网页原生兼容引擎。"
            )
    signature = hashlib.sha256(
        (
            signature
            + "|"
            + repr(denoising_params)
            + "|"
            + repr(
                (
                    manual_bpm,
                    manual_first_beat,
                    manual_first_downbeat,
                    int(analysis_meter),
                    bool(force_constant),
                    bool(trim_silence),
                    float(trim_top_db),
                    song_start_seconds,
                    song_end_seconds,
                    analysis_backend,
                    processing_engine,
                    beat_device,
                )
            )
        ).encode("utf-8")
    ).hexdigest()

    if st.button("分析两路音频", type="primary", width="stretch"):
        try:
            with st.spinner("正在预处理心跳并分析歌曲节拍……"):
                heartbeat_result = process_audio_bytes(
                    heartbeat_upload.name,
                    heartbeat_upload,
                    denoising_params,
                    max_duration_seconds=MAX_HEARTBEAT_DURATION_SECONDS,
                    artifact_profile="web",
                    create_zip=False,
                )
                song_analysis = analyze_song_bytes(
                    song_upload.name,
                    song_upload,
                    manual_bpm=manual_bpm,
                    manual_first_beat=manual_first_beat,
                    manual_first_downbeat=manual_first_downbeat,
                    manual_meter=int(analysis_meter),
                    force_constant_grid=force_constant,
                    trim_silence=trim_silence,
                    trim_top_db=float(trim_top_db),
                    song_start_seconds=song_start_seconds,
                    song_end_seconds=song_end_seconds,
                    analysis_backend="librosa" if processing_engine == "heartbeat_sync_cli" else analysis_backend,
                    max_duration_seconds=MAX_SONG_DURATION_SECONDS,
                )
                cleanup_render_result(st.session_state.get("processor_render"))
                st.session_state["processor_analysis"] = {
                    "signature": signature,
                    "heartbeat": compact_heartbeat_result(heartbeat_result),
                    "song": song_analysis,
                }
                st.session_state.pop("processor_render", None)
            st.success("分析完成。请检查节拍和心跳质量，然后调整处理参数。")
        except Exception as exc:
            LOGGER.exception("Audio analysis failed")
            st.error(f"分析失败：{exc}")

    saved = st.session_state.get("processor_analysis")
    if saved is None or saved.get("signature") != signature:
        return
    heartbeat_result = saved["heartbeat"]
    song_analysis = saved["song"]
    show_analysis(heartbeat_result, song_analysis)

    st.header("3. 全局音乐处理参数")
    duration = float(song_analysis["duration_seconds"])
    style_name = st.selectbox(
        "音乐风格预设",
        list(STYLE_PRESETS),
        index=list(STYLE_PRESETS).index("cinematic"),
        format_func=lambda name: str(STYLE_PRESETS[name]["label"]),
        help="预设会同时改变心跳密度、律动松紧、音色、空间感与闪避强度，所有参数仍可手动覆盖。",
    )
    style = get_style_preset(style_name)
    st.caption(
        "推荐优先尝试电影配乐、Trip-hop、极简电子、Lo-fi 和暗氛围；"
        "它们通常为低频心跳留出更稳定的频谱与编排空间。"
    )
    row1 = st.columns(4)
    pulse_options = ["auto", "downbeat", "kick", "backbeat", "every-beat", "bar", "half", "normal", "double"]
    default_pulse = str(style["pulse_mode"])
    pulse_mode = row1[0].selectbox(
        "心跳节奏角色",
        pulse_options,
        index=pulse_options.index(default_pulse),
        format_func=PULSE_LABELS.get,
        key=f"pulse_mode_{style_name}",
    )
    fit_options = ["preserve", "gap", "stretch"]
    default_fit = str(style["fit_mode"])
    fit_mode = row1[1].selectbox(
        "周期适配方式",
        fit_options,
        index=fit_options.index(default_fit),
        format_func=FIT_LABELS.get,
        key=f"fit_mode_{style_name}",
    )
    beats_per_bar = row1[2].number_input(
        "每小节拍数",
        2,
        12,
        int(song_analysis.get("meter", 4)),
        1,
    )
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
        "心跳相对响度 (LU)", -12.0, 12.0, float(style["heartbeat_relative_lu"]), 0.5,
        help="正值让心跳比歌曲活跃段更突出。",
        key=f"heartbeat_relative_{style_name}",
    )

    with st.expander("高级混音与母带参数"):
        advanced = st.columns(5)
        song_target_lufs = advanced[0].slider("歌曲目标 LUFS", -24.0, -10.0, -18.0, 0.5)
        ducking_db = advanced[1].slider(
            "心跳触发低频闪避 (dB)", 0.0, 9.0, float(style["ducking_db"]), 0.1,
            key=f"ducking_{style_name}",
        )
        ducking_cutoff = advanced[2].slider("低频闪避截止 (Hz)", 80.0, 600.0, 280.0, 10.0)
        master_target = advanced[3].slider("最终目标 LUFS", -24.0, -10.0, -16.0, 0.5)
        ceiling = advanced[4].slider("输出峰值上限 (dBFS)", -6.0, -0.1, -1.0, 0.1)
        alignment = st.columns(7)
        pulse_min_bpm = alignment[0].slider("最低心跳密度", 35.0, 100.0, 55.0, 1.0)
        pulse_max_bpm = alignment[1].slider("最高心跳密度", 70.0, 180.0, 110.0, 1.0)
        timing_offset_ms = alignment[2].slider("听感偏移 (ms)", -120.0, 120.0, 0.0, 1.0)
        section_strength = alignment[3].slider("乐段动态强度", 0.0, 1.0, 0.65, 0.05)
        fade_in_seconds = alignment[4].slider("心跳渐入 (秒)", 0.0, 20.0, 4.0, 0.5)
        fade_out_seconds = alignment[5].slider("心跳渐出 (秒)", 0.0, 20.0, 5.0, 0.5)
        max_stretch_ratio = alignment[6].slider("最大拉伸倍率", 1.02, 1.50, 1.10, 0.01)
        groove = st.columns(7)
        quantize_strength = groove[0].slider(
            "节拍吸附强度", 0.0, 1.0, float(style["quantize_strength"]), 0.01,
            key=f"quantize_{style_name}",
            help="降低后会保留心跳自身的呼吸感，避免每次都钉死在网格上。",
        )
        humanize_ms = groove[1].slider(
            "自然微时差(ms)", 0.0, 40.0, float(style["humanize_ms"]), 1.0,
            key=f"humanize_{style_name}",
        )
        swing = groove[2].slider(
            "Swing", -0.30, 0.30, float(style["swing"]), 0.01,
            key=f"swing_{style_name}",
        )
        presence_db = groove[3].slider(
            "心跳存在感(dB)", -6.0, 6.0, float(style["presence_db"]), 0.5,
            key=f"presence_{style_name}",
        )
        saturation = groove[4].slider(
            "心跳饱和度", 0.0, 1.0, float(style["saturation"]), 0.01,
            key=f"saturation_{style_name}",
        )
        reverb_mix = groove[5].slider(
            "心跳空间感", 0.0, 0.45, float(style["reverb_mix"]), 0.01,
            key=f"reverb_{style_name}",
        )
        reverb_decay_ms = groove[6].slider(
            "空间衰减(ms)", 60.0, 1200.0, float(style["reverb_decay_ms"]), 10.0,
            key=f"decay_{style_name}",
        )
        contract = st.columns(4)
        beats_per_loop = contract[0].number_input(
            "每个心跳循环包含周期数", 1, 16, 4, 1,
            help="从质量最好的连续真实心跳周期中选取循环素材。",
        )
        intro_pulses = contract[1].number_input("歌曲前心跳次数", 0, 16, 4, 1)
        outro_pulses = contract[2].number_input("歌曲后心跳次数", 0, 16, 4, 1)
        intro_outro_boost_db = contract[3].slider(
            "片头/片尾心跳增强 (dB)", 0.0, 12.0, 4.0, 0.5
        )
        output_format = st.selectbox(
            "最终文件格式",
            ["mp3", "flac16", "wav16", "wav24"],
            format_func={
                "mp3": "MP3（推荐：体积最小，方便分享）",
                "flac16": "FLAC 16-bit（无损，通常比 WAV 小）",
                "wav16": "WAV 16-bit（兼容性优先）",
                "wav24": "WAV 24-bit（制作母版，体积最大）",
            }.get,
        )
        export_project_files = st.checkbox(
            "生成 heartbeat_sync 五文件兼容包、歌曲轨和工程 ZIP",
            value=False,
            help=(
                "额外生成 24-bit preview_mix、heartbeat_aligned、debug_click_mix、"
                "heartbeat_detection_mix 和 analysis_report；会增加渲染时间与临时磁盘占用。"
            ),
        )

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
                options=[
                    "inherit",
                    "auto",
                    "downbeat",
                    "kick",
                    "backbeat",
                    "every-beat",
                    "bar",
                    "half",
                    "normal",
                    "double",
                    "mute",
                ],
                default="inherit",
            ),
            "周期适配": st.column_config.SelectboxColumn(
                options=["inherit", "preserve", "gap", "stretch"],
                default="inherit",
            ),
            "边界淡化(ms)": st.column_config.NumberColumn(default=80.0, min_value=0.0, max_value=5000.0, step=10.0),
            "听感偏移(ms)": st.column_config.NumberColumn(min_value=-250.0, max_value=250.0, step=1.0),
            "自然波动(ms)": st.column_config.NumberColumn(min_value=0.0, max_value=60.0, step=1.0),
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
        style_preset=style_name,
        pulse_mode=pulse_mode,
        fit_mode=fit_mode,
        beats_per_bar=int(beats_per_bar),
        beats_per_loop=int(beats_per_loop),
        intro_pulses=int(intro_pulses),
        outro_pulses=int(outro_pulses),
        intro_outro_boost_db=float(intro_outro_boost_db),
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
        pulse_min_bpm=float(pulse_min_bpm),
        pulse_max_bpm=float(max(pulse_min_bpm, pulse_max_bpm)),
        timing_offset_ms=float(timing_offset_ms),
        quantize_strength=float(quantize_strength),
        humanize_ms=float(humanize_ms),
        swing=float(swing),
        section_adaptive_strength=float(section_strength),
        heartbeat_fade_in_seconds=float(fade_in_seconds),
        heartbeat_fade_out_seconds=float(fade_out_seconds),
        max_stretch_ratio=float(max_stretch_ratio),
        heartbeat_presence_db=float(presence_db),
        heartbeat_saturation=float(saturation),
        heartbeat_reverb_mix=float(reverb_mix),
        heartbeat_reverb_decay_ms=float(reverb_decay_ms),
        output_format=str(output_format),
        sync_contract_exports=bool(export_project_files),
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
        cleanup_render_result(st.session_state.get("processor_render"))
        st.session_state.pop("processor_analysis", None)
        st.session_state.pop("processor_render", None)
        st.rerun()

    if preview_clicked or final_clicked:
        if not RENDER_SEMAPHORE.acquire(blocking=False):
            st.warning("服务器正在处理另一个渲染任务，请稍后再试。")
        else:
            output_dir: str | None = None
            try:
                label = "试听" if preview_clicked else "整首歌曲"
                output_dir = tempfile.mkdtemp(prefix="job_", dir=initialize_job_root())
                with st.spinner(f"正在渲染{label}……"):
                    if processing_engine == "heartbeat_sync_cli":
                        if sync_service is None:
                            raise RuntimeError("heartbeat_sync CLI service is not configured")
                        job_path = Path(output_dir)
                        input_path = job_path / "inputs"
                        input_path.mkdir(parents=True, exist_ok=True)
                        heartbeat_path = input_path / "heartbeat_cleaned.wav"
                        song_suffix = Path(song_upload.name).suffix.lower()
                        if song_suffix not in {".wav", ".mp3"}:
                            song_suffix = ".wav"
                        song_path = input_path / f"song{song_suffix}"
                        heartbeat_path.write_bytes(
                            heartbeat_result["artifacts"]["cleaned.wav"]
                        )
                        song_path.write_bytes(bytes(song_data))
                        sync_output_root = job_path / "heartbeat_sync_outputs"
                        command = build_sync_command(
                            sync_service,
                            heartbeat_path=heartbeat_path,
                            song_path=song_path,
                            output_root=sync_output_root,
                            song_analysis_backend=analysis_backend,
                            beat_device=beat_device,
                            manual_song_bpm=manual_bpm,
                            manual_first_beat=manual_first_beat,
                            trim_silence=bool(trim_silence),
                            trim_top_db=float(trim_top_db),
                            song_start_seconds=song_start_seconds,
                            song_end_seconds=song_end_seconds,
                            max_song_seconds=preview_length if preview_clicked else None,
                            beats_per_loop=int(beats_per_loop),
                            intro_pulses=int(intro_pulses),
                            outro_pulses=int(outro_pulses),
                            pulse_mode={
                                "downbeat": "bar",
                                "kick": "half",
                                "backbeat": "half",
                                "every-beat": "normal",
                                "mute": "bar",
                            }.get(pulse_mode, pulse_mode),
                            pulse_min=float(pulse_min_bpm),
                            pulse_max=float(max(pulse_min_bpm, pulse_max_bpm)),
                            song_gain_db=float(song_gain_db),
                            heartbeat_gain_db=float(heartbeat_gain_db),
                            auto_loudness=bool(auto_balance),
                            song_target_lufs=float(song_target_lufs),
                            heartbeat_relative_lu=float(heartbeat_relative_lu),
                            intro_outro_boost_db=float(intro_outro_boost_db),
                            fit_mode=fit_mode,
                        )
                        result = adapt_sync_result(
                            run_sync_cli(
                                sync_service,
                                command,
                                output_root=sync_output_root,
                            )
                        )
                    else:
                        result = process_music_bytes(
                            song_upload.name,
                            song_upload,
                            heartbeat_result,
                            song_analysis,
                            mix_params,
                            region_edits,
                            render_duration_seconds=preview_length if preview_clicked else None,
                            output_dir=output_dir,
                            export_stems=bool(export_project_files),
                            export_debug=bool(export_project_files),
                            create_zip=bool(export_project_files),
                        )
                    result["job_id"] = Path(output_dir).name
                    cleanup_render_result(st.session_state.get("processor_render"))
                    st.session_state["processor_render"] = {
                        "signature": signature,
                        "result": result,
                        "output_dir": output_dir,
                    }
                st.success(f"{label}渲染完成。")
            except Exception as exc:
                LOGGER.exception("Music render failed")
                if output_dir:
                    cleanup_render_result({"output_dir": output_dir})
                st.error(f"渲染失败：{exc}")
            finally:
                RENDER_SEMAPHORE.release()

    rendered = st.session_state.get("processor_render")
    if rendered is not None and rendered.get("signature") == signature:
        render_outputs(rendered["result"])


if __name__ == "__main__":
    main()
