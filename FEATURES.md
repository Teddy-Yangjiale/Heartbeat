# LegaSynth 完整流程与两大功能（A / B）

本文件说明「心跳音乐视频」完整流程里的两个进阶功能。总编排在 [legasynth/pipeline.py](legasynth/pipeline.py) 的 `process_one()`：

```
心跳 wav ──► Stage-1 分析（BPM / beat_times / best_loop）
                     │
                     ├─► A 情绪分析 (emotion.py) ─────────► 视觉风格 style_profile
                     │
歌曲 mp4 ──► 提取音频 + 歌曲 BPM (video_audio.py)
                     │
                     ▼
   保留原声 + 心跳 loop 对齐歌曲 BPM 铺底混音 (mixing.py) ──► final_audio.wav/mp3
                     │
                     ▼
        B 心跳驱动渲染 (video_render.py) ─────────────► final_video.mp4
```

**关键点**：最终音频 = **歌曲原声 + 心跳鼓点铺底**。心跳 `best_loop.wav` 被时间伸缩对齐到歌曲 BPM 后平铺，作为节奏底铺在原声下方（音量由 `heartbeat_gain_db` 控制）。不做人声分离、不做中文翻唱。

---

## A · 心跳情绪 → 视频调性（`legasynth/emotion.py`）

**做什么**：读取 Stage-1 已经算好的心率与 inter-beat interval（IBI），派生一小组心率变异性（HRV）特征，放到经典的 **valence（效价）/ arousal（唤醒）** 情绪平面上，再把这个点翻译成一份具体的「视觉风格档案」（`style_profile`）供 B 使用。

**特征**（`compute_hrv_features`）：平均心率、SDNN、RMSSD、pNN50、IBI 变异系数、包络对比度。

**情绪映射**（`estimate_valence_arousal`，均为透明的规则，可调）：

- **arousal ∝ 心率**：55 bpm→平静，110 bpm→高度激活。
- **valence ∝ 健康程度的 HRV**：RMSSD 太低（过于机械）或过高（不规则/噪声）都拉低效价，中等偏高最「正向放松」。高心率再额外下调效价（压力读数）。

**风格档案**（`build_style_profile`）：warmth（冷暖）、saturation、brightness、contrast、vignette（暗角）、pulse_intensity（每拍脉冲深度）、flash_strength（闪白）、beats_per_cut（每几拍一个剪辑重音，越激动越快切）、grade_name。四象限命名：欢快激昂 / 紧张不安 / 宁静安详 / 沉静伤感。

**输出**：`reports/emotion_report.json`。

> ⚠️ 这是驱动艺术效果的启发式情绪映射，**不是医疗或诊断结论**，报告里也带了 disclaimer。

---

## B · 心跳驱动的剪辑节奏（`legasynth/video_render.py`）

源视频保持线性时间轴（与混音严格同步），在其之上叠加心跳同步动态：

- **每拍缩放 + 亮度脉冲**：画面随每一次心跳「呼吸」（`nearest_pulse_strength` 给出以每拍为中心的高斯包络）。
- **强拍/剪辑拍闪白**：`flash_strength` 控制。
- **按拍剪辑重音**：每 `beats_per_cut` 拍一个更硬的缩放冲击（`select_cut_beats`）；该值来自 A，心跳越快切得越碎。
- **情绪调色**：把 A 的 warmth / saturation / brightness / contrast / vignette 应用到每一帧（`apply_color_grade`，用亮度混合做低成本饱和度，避免每帧两次 HSV 转换）。
- **诊断叠层**（`show_overlay`，**默认关闭**）：心跳波形 + BPM + 情绪标签的 HUD，只在需要展示技术/调试时开。默认关闭是为了自然度——开着会像分析工具而非音乐视频。

关键参数（`render_heartbeat_video`）：`effect_strength`（总强度）、`style_profile`（来自 A）、`show_overlay`（默认 False）、`enable_beat_editing`、`duration_limit`。输出 `video/video_render_report.json` + `final_video.mp4`。

### 自然度评估（真实素材实测）

在真实 CC-BY MV（Yoga Lin – Elephant Slide）+ 真实 PCG 心跳上实测：情绪调色、缩放脉冲、按拍剪辑都**克制自然**;把默认诊断 HUD 关掉后成品更像真正的音乐视频。可继续增强的亮点：句首淡入的标题/献词卡、角落里精致的小号心率徽标（替代整块 HUD）、按段落（主歌/副歌）自适应的特效强度、剪辑拍上的转场（叠化/白闪的更柔和版本）。

---

## 依赖说明

A、B 只用已有依赖（numpy / scipy / librosa / opencv / imageio-ffmpeg），无需 torch 或任何在线服务。
