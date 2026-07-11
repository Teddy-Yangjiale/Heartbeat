# LegaSynth 完整流程与三大功能（A / B / C）

本文件说明"心跳音乐视频"完整流程里的三个进阶功能。总编排在 [legasynth/pipeline.py](legasynth/pipeline.py) 的 `process_one()`：

```
心跳 wav ──► Stage-1 分析（BPM / beat_times / best_loop）
                     │
                     ├─► A 情绪分析 (emotion.py) ─────────► 视觉风格 style_profile
                     │
歌曲 mp4 ──► 提取音频 + 歌曲 BPM (video_audio.py)
                     │
                     ├─► C 中文翻唱 (chinese_cover.py) ──► cover_audio.wav（可选，替换原唱）
                     │
                     ▼
        心跳 loop 对齐歌曲 BPM 并混音 (mixing.py) ──► final_audio.wav/mp3
                     │
                     ▼
        B 心跳驱动渲染 (video_render.py) ─────────────► final_video.mp4
```

---

## A · 心跳情绪 → 视频调性（`legasynth/emotion.py`）

**做什么**：读取 Stage-1 已经算好的心率与 inter-beat interval（IBI），派生一小组 HRV 特征，放到经典的 **valence（效价）/ arousal（唤醒）** 情绪平面上，再把这个点翻译成一份具体的"视觉风格档案"（`style_profile`）供 B 使用。

**特征**（`compute_hrv_features`）：平均心率、SDNN、RMSSD、pNN50、IBI 变异系数、包络对比度。

**情绪映射**（`estimate_valence_arousal`，均为透明的规则，可调）：

- **arousal ∝ 心率**：55 bpm→平静，110 bpm→高度激活。
- **valence ∝ 健康程度的 HRV**：RMSSD 太低（过于机械）或过高（不规则/噪声）都拉低效价，中等偏高最"正向放松"。高心率再额外下调效价（压力读数）。

**风格档案**（`build_style_profile`）：warmth（冷暖）、saturation、brightness、contrast、vignette（暗角）、pulse_intensity（每拍脉冲深度）、flash_strength（闪白）、beats_per_cut（每几拍一个剪辑重音，越激动越快切）、grade_name。四象限命名：欢快激昂 / 紧张不安 / 宁静安详 / 沉静伤感。

**输出**：`reports/emotion_report.json`。

> ⚠️ 这是驱动艺术效果的启发式情绪映射，**不是医疗或诊断结论**，报告里也带了 disclaimer。

---

## B · 心跳驱动的剪辑节奏（`legasynth/video_render.py`）

源视频保持线性时间轴（与混音严格同步），在其之上叠加心跳同步动态：

- **每拍缩放 + 亮度脉冲**：画面随每一次心跳"呼吸"（`nearest_pulse_strength` 给出以每拍为中心的高斯包络）。
- **强拍/剪辑拍闪白**：`flash_strength` 控制。
- **按拍剪辑重音**：每 `beats_per_cut` 拍一个更硬的缩放冲击（`select_cut_beats`）；该值来自 A，心跳越快切得越碎。
- **情绪调色**：把 A 的 warmth / saturation / brightness / contrast / vignette 应用到每一帧（`apply_color_grade`，用亮度混合做低成本饱和度，避免每帧两次 HSV 转换）。
- **中文歌词卡拉OK字幕**（`draw_subtitle` / `active_subtitle`）：功能 C 的逐句歌词以居中、半透明底、淡入淡出的字幕烧录进画面——这是让成品"像 MV"的关键，默认开启。
- **诊断叠层**（`show_overlay`，**默认关闭**）：心跳波形 + BPM + 情绪标签的 HUD，只在需要展示技术/调试时开。默认关闭是为了自然度——开着会像分析工具而非音乐视频。

关键参数（`render_heartbeat_video`）：`effect_strength`（总强度）、`style_profile`（来自 A）、`subtitles`（来自 C）、`show_overlay`（默认 False）、`enable_beat_editing`、`duration_limit`。输出 `video/video_render_report.json` + `final_video.mp4`。

### 自然度评估（真实素材实测）

在真实 CC-BY MV（Yoga Lin – Elephant Slide）+ 真实 PCG 心跳上实测：情绪调色、缩放脉冲、按拍剪辑都**克制自然**;把默认诊断 HUD 关掉、改用**中文卡拉OK字幕**后，成品明显更像真正的音乐视频。可继续增强的亮点：句首淡入的标题/献词卡、角落里精致的小号心率徽标（替代整块 HUD）、按段落（主歌/副歌）自适应的特效强度、剪辑拍上的转场（叠化/白闪的更柔和版本）。

---

## C · 中文翻唱：lyrics sung in Chinese（`legasynth/chinese_cover.py`）

一条 5 段管线，每段在缺少可选/重型依赖时都能优雅降级：

1. **人声/伴奏分离**（`separate_vocal_instrumental`）：默认卡拉OK**中置声道消除**（`L-R` 抵消居中的主唱，`(L+R)/2` 作为提取旋律用的粗人声）。零重依赖、即时。单声道输入无法消除时回退为"原音轨直通 + 中文人声叠加"。（未来可选 Demucs 提升质量。）
2. **歌词与时间**（`parse_lyrics`）：支持带时间戳的 **LRC**；或纯文本逐行（自动检测人声活跃区间后均匀分布）。
3. **旋律**（`extract_melody_f0`）：用 `librosa.pyin` 提取原唱基频轮廓；每句取窗口内中位音高，并**八度折叠**到合理演唱音域（避免被低音和弦带偏、避免极端变调发浑）。
4. **中文演唱合成**（`synthesize_chinese_vocal`）：逐句用 **edge-tts**（微软在线 TTS，轻量、免 torch）合成中文语音，再**时间伸缩**填满该句时槽、**变调**到旋律目标音高 → 得到跟随歌曲旋律的中文"演唱"。无网络/edge-tts 时回退为跟随旋律的正弦哼鸣占位（清楚标注）。
5. **混音**（`generate_chinese_cover`）：合成中文人声 + 伴奏 → `cover_audio.wav`，随后作为"歌曲音频"进入心跳混音与视频渲染。

**参数**：`lyrics` / `lrc` / `voice`（edge-tts 音色）/ `vocal_gain_db` / `instrumental_gain_db`。输出 `chinese_cover/chinese_cover_report.json`、`cover_audio.wav`、`chinese_vocal.wav`、`instrumental.wav`。

**已知局限（写进报告的 future_work）**：目前是**句级**一个旋律音（非音节级）；EN→ZH 尚未做"可唱性"（音节数/声调约束）翻译，需用户提供中文歌词；音色为 TTS 而非神经歌声合成（DiffSinger 等）。这些是明确的进阶方向，也是与参考论文差异化、避免重复工作的地方。

---

## 依赖说明

A、B 只用已有依赖（numpy / scipy / librosa / opencv / imageio-ffmpeg）。C 额外需要 `edge-tts`（已加入 `requirements.txt` / `environment.yml`）；中文演唱需要联网访问微软 TTS 端点，离线时自动回退为占位哼鸣。
