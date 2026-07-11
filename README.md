# Heartbeat / LegaSynth

Repository: [Teddy-Yangjiale/Heartbeat](https://github.com/Teddy-Yangjiale/Heartbeat)

LegaSynth 把一段**心跳录音**和一段**英文歌曲音乐视频（.mp4）**，自动合成为一支**个性化的心跳音乐视频**。

网页 App 现在有两个标签页：

- **🎬 Heartbeat music video（完整流程）**：上传心跳 `.wav/.mp3` + 歌曲 `.mp4`，一键生成并下载 `final_video.mp4` / `final_audio.wav/mp3`。**保留歌曲原声**，把心跳作为鼓点铺底，并做两项视频处理：
  - **A · 情绪调色**：从心跳的心率 / 心率变异性（HRV）估计情绪（valence/arousal），自动决定视频的色温、饱和度、对比度、暗角和剪辑节奏。*（启发式艺术映射，非医疗诊断。）*
  - **B · 心跳驱动剪辑**：视频随心跳"呼吸"——每拍缩放脉冲、强拍闪白、按心率每 N 拍一个剪辑重音。
- **🔬 Stage 1 · heartbeat analysis**：只做心跳音频分析（预处理、节拍检测、BPM、稳定 loop、诊断图/报告）；本文件第 4 节起主要讲这部分参数与调优。

命令行 [scripts/process_files.py](scripts/process_files.py) 同时支持 Stage-1 批处理和完整音乐视频流程（`--heartbeat X --video Y`，见第 5 节）。完整流程各功能与实现细节见 [FEATURES.md](FEATURES.md)。

## 1. 功能概览

输入：

- 单个或多个心跳 `.wav`
- 单个或多个心跳 `.mp3`

输出：

- 心跳 BPM 估计
- beat times，也就是每一次心跳被检测到的时间点
- IBI，也就是 inter-beat interval，相邻心跳之间的时间间隔
- local BPM，也就是每个 IBI 对应的局部心率
- 最稳定的一段心跳 loop
- 清洗后的音频
- 用于检测的滤波音频
- envelope 包络数据
- 参数 JSON
- 诊断 Markdown 报告
- 诊断 PNG 图
- 录音质量评分、风险原因和是否建议用于 loop
- 多时间窗 BPM 共识分析
- 按节律稳定性与心音包络质量排名的 loop 候选表
- Streamlit 页面中的单文件 ZIP 和批量 ZIP 下载

处理流程：

```text
heartbeat .wav/.mp3
-> decode audio
-> convert to mono float waveform
-> remove DC offset
-> normalize
-> band-pass filter for heartbeat detection
-> envelope extraction
-> autocorrelation BPM estimation
-> peak detection
-> S1/S2 double-peak suppression
-> beat_times / IBI / local BPM
-> choose most stable loop window
-> zero-crossing adjustment and edge fade
-> export JSON / CSV / WAV / PNG / ZIP
```

### 1.1 面对说话声和环境噪声的预处理

听诊器录音不是纯心音：讲话、空调、手持摩擦和低频振动都可能被录入。当前版本默认启用 `Speech and noise suppression`，处理链路为：

```text
DC removal
-> 25-160 Hz heart-sound band-pass
-> harmonic/percussive separation (抑制持续语音的基频与谐波)
-> adaptive spectral noise gate (压低稳定环境噪声)
-> BPM / beat detection
-> beat-synchronous soft gate (保留每拍 S1/S2 附近，衰减拍间声音)
```

因此 `cleaned.wav`、`filtered_detection.wav` 和 `best_loop.wav` 都来自抑制后的信号；`best_loop.wav` 不再从只去 DC 的原始录音截取。Balanced 模式会让拍间音频降低约 `28 dB`，但不会把它硬切为静音，以避免明显的门限爆音。

这是一种 CPU-only 的信号处理方案，不是通用的人声分离模型。若讲话声刚好与 S1/S2 同时发生，单支听诊器麦克风没有独立参考信号，无法保证完全移除而不伤害心音。此时应优先使用 `Strong`，或重新安静录一段至少 10-15 秒的素材。

## 2. 从零开始启动

### 2.1 安装基础软件

需要 Windows + conda。推荐安装 Anaconda 或 Miniconda。

检查 conda 是否可用：

```powershell
conda --version
```

如果能看到版本号，例如 `conda 25.x.x`，说明 conda 可用。

### 2.2 获取项目

如果你还没有本地代码：

```powershell
cd /d D:\
git clone https://github.com/Teddy-Yangjiale/Heartbeat.git
cd Heartbeat
```

如果你已经在本机 `D:\Heartbeat`：

```powershell
cd /d D:\Heartbeat
```

### 2.3 创建或更新 conda 环境

项目默认使用 conda 环境名：

```text
heartbeat
```

一键安装或更新依赖：

```powershell
setup_env.bat
```

等价的手动命令：

```powershell
conda env update -n heartbeat -f environment.yml --prune
```

如果本机还没有 `heartbeat` 环境，`setup_env.bat` 会尝试根据 `environment.yml` 创建。

核心依赖：

```text
python=3.11
numpy
scipy
pandas
matplotlib
streamlit
opencv-python
moviepy
pydub
librosa
soundfile
imageio-ffmpeg
```

第一阶段主要使用 `numpy`、`scipy`、`pandas`、`matplotlib`、`streamlit`、`librosa` 和 `soundfile`。音视频相关包保留给后续阶段。

## 3. 启动 Web App

### 3.1 本机模式

```powershell
run_app.bat
```

然后打开：

```text
http://127.0.0.1:8501
```

这个模式只适合本机访问。

### 3.2 局域网 / 虚拟网卡模式

```powershell
run_app_lan.bat
```

这个脚本会绑定：

```text
0.0.0.0:8501
```

本机仍然可以访问：

```text
http://127.0.0.1:8501
```

其他设备或虚拟网卡另一端访问：

```text
http://你的Windows虚拟网卡IPv4:8501
```

查看 Windows IPv4：

```powershell
ipconfig
```

如果本机 `127.0.0.1:8501` 可以打开，但其他设备打不开，优先检查：

- Windows 防火墙是否允许 TCP 8501 入站
- 访问的是否是正确的虚拟网卡 IPv4
- 对方设备是否和该虚拟网卡网络互通

### 3.3 为什么启动脚本带这些参数

当前启动脚本使用：

```text
--server.headless true
--browser.gatherUsageStats false
--no-capture-output
python -m streamlit
```

含义：

- `--browser.gatherUsageStats false`：跳过 Streamlit 首次启动时的 email onboarding prompt。
- `--server.headless true`：不自动打开浏览器，适合命令行和远程模式。
- `--no-capture-output`：让 `conda run` 不吞掉 Streamlit 的长驻输出，减少启动失败。
- `python -m streamlit`：通过当前 conda 环境里的 Python 调用 Streamlit，避免 `streamlit.exe` 包装路径差异。

## 4. Web App 使用方法

1. 打开页面。
2. 在左侧参数区选择处理参数。
3. 在上传区选择一个或多个 `.wav` / `.mp3` 心跳音频。
4. 点击 `Process uploaded files`。
5. 页面会显示每个文件的：
   - duration
   - sample rate
   - estimated BPM
   - detected beats
   - IBI std
   - peak dBFS
   - best loop 起止时间
   - diagnostic plot
   - cleaned audio preview
   - filtered detection audio preview
   - best loop audio preview
6. 下载单文件结果 ZIP，或下载所有文件的 batch ZIP。

如果勾选保存输出，结果会写入：

```text
outputs/YYYYMMDD_HHMMSS/
```

每个输入文件会有一个子目录。

## 5. 命令行使用方法

单个文件：

```powershell
conda run -n heartbeat python scripts\process_files.py "C:\path\heartbeat.wav" --out outputs\stage1
```

MP3 文件：

```powershell
conda run -n heartbeat python scripts\process_files.py "C:\path\heartbeat.mp3" --out outputs\stage1_mp3
```

多个文件：

```powershell
conda run -n heartbeat python scripts\process_files.py `
  "C:\Users\33480\Desktop\20260710231622\202607071630425298.wav" `
  "C:\Users\33480\Desktop\20260710231622\202607081521080317.wav" `
  "C:\Users\33480\Desktop\20260710231622\202607091053425340.wav" `
  --out outputs\stage1_wav_validation
```

调整目标 loop 长度：

```powershell
conda run -n heartbeat python scripts\process_files.py "C:\path\heartbeat.wav" --out outputs\stage1 --loop-beats 6
```

## 6. 输出文件说明

### `recording_quality.json`

自动化、非医疗诊断的录音可用性评估。包含 `score`（0-100）、`grade`、是否建议用于 loop，以及具体风险原因，例如录音过短、clipping、心音频段能量不足、不同时间窗 BPM 不一致或心音包络对比度不足。

### `window_analysis.csv`

将录音按重叠时间窗分别估计 BPM。`is_consensus_inlier` 为 `true` 的窗口会参与最终 BPM 的中位数共识；不一致的窗口会被标记，不再单独主导最终 BPM。

### `loop_candidates.csv`

候选 loop 的质量排名。除 `regularity_score` 外，还包含 `envelope_snr_db` 和 `quality_score`。系统优先选择规则性高、心音峰相对背景更清晰的候选，而不是只按 IBI 方差排序。

### `template_analysis.csv` 和 `heartbeat_template.csv`

模板确认的可解释数据。`heartbeat_template.csv` 是由候选心拍中位数得到的归一化波形；`template_analysis.csv` 给出每个候选的相关性、是否匹配模板、是否最终保留和决定原因。为避免漏拍，只有候选数明显高于 BPM 预期时，模板才会删除不匹配的峰；否则只记录相似度并保留节拍序列。

### `manual_corrections.json`

记录是否在 Web App 中人工修改了 beat 时间或 loop 起止时间。展开每个处理结果的 `Manual beat and loop correction` 后，可直接在表格中增删/编辑以秒为单位的 beat，再指定 loop 起止时间并点击应用。系统会重新运行导出链路；`Restore automatic analysis` 会恢复自动结果。

每个输入音频会生成以下文件。

### `tempo_summary.json`

最完整的机器可读总结文件。包含：

- `filename`：原始文件名。
- `sample_rate`：采样率。
- `duration_seconds`：音频时长。
- `source`：输入音频元数据，例如格式、通道数、样本数、dtype。
- `quality`：信号质量指标。
- `tempo`：BPM、beat times、IBI 等节拍分析结果。
- `best_loop`：系统选出的最稳定 loop。
- `parameters`：本次运行使用的全部处理参数。

推荐把这个文件作为后续阶段混音、视频生成和报告分析的主要输入。

### `processing_parameters.json`

只保存本次运行使用的参数，例如：

- `bandpass_low_hz`
- `bandpass_high_hz`
- `envelope_lowpass_hz`
- `min_bpm`
- `max_bpm`
- `peak_prominence`
- `peak_height_percentile`
- `double_peak_suppression`
- `target_loop_beats`
- `crossfade_ms`

用途是复现实验。如果两次输出不同，先对比这个文件。

### `diagnostic_report.md`

人类可读的 Markdown 报告。包含：

- 输入音频信息
- 信号质量
- BPM 和 beat 统计
- 最佳 loop 起止时间
- 所有处理参数

适合写项目报告或调试记录。

### `beat_times.csv`

每一行是一颗被检测到的心跳：

| 字段 | 含义 |
| --- | --- |
| `beat_index` | 第几个心跳，从 0 开始 |
| `time_seconds` | 该心跳在音频中的时间，单位秒 |

后续如果要把心跳和音乐、动画、视频闪烁对齐，主要使用这个文件。

### `ibi.csv`

IBI 是 inter-beat interval，相邻两次心跳之间的时间差。

| 字段 | 含义 |
| --- | --- |
| `interval_index` | 第几个心跳间隔 |
| `start_time_seconds` | 该间隔的起始心跳时间 |
| `ibi_seconds` | 两次心跳之间的时间差，单位秒 |
| `local_bpm` | 该间隔换算出的局部 BPM，计算方式是 `60 / ibi_seconds` |

IBI 越稳定，说明这段心跳越适合作为 loop。

### `envelope.csv`

检测包络数据。它不是原始波形，而是经过滤波、Hilbert envelope 和低通平滑后的检测曲线。

| 字段 | 含义 |
| --- | --- |
| `time_seconds` | 时间，单位秒 |
| `envelope` | 归一化后的包络强度 |

peak detection 主要是在 envelope 上做的。

### `cleaned.wav`

清洗后的心跳音频：

- 转 mono
- 去 DC offset
- 心音频段 band-pass
- harmonic/percussive separation 和自适应频谱降噪
- 以检测到的心跳为中心的软门控，压低拍间说话声
- normalize

适合人耳试听，也适合作为后续音频处理的基础版本。

### `filtered_detection.wav`

用于试听和诊断的滤波音频。内部节拍检测在 heart-sound band-pass、harmonic/percussive separation 和自适应频谱降噪之后完成；导出的这个文件还会施加与 `cleaned.wav` 相同的软门控，确保拍间说话声不会因为“检测音频”的历史命名而残留。

注意：这个文件主要用于诊断检测效果，不一定是最好听的版本。

### `best_loop.wav`

系统选出的最稳定心跳 loop：

- 根据 beat_times 找 IBI 方差最低的窗口
- 根据目标 beat 数裁剪
- 对边界做 zero-crossing 调整
- 对开头和结尾做 fade，减少 loop click

后续做音乐混音时，优先使用这个文件。

### `diagnostic_plot.png`

诊断图包含五部分：

1. 原始 mono waveform
2. 抑制语音后的 detection signal
3. beat-synchronous cleaned heartbeat audio
4. envelope、detected beats 和 selected loop
5. local BPM by IBI

如果 BPM 或 beat count 不对，先看这张图：

- 红线是否落在真正心跳上
- 绿色 loop 区间是否稳定
- envelope 是否过平或噪声过多

## 7. 参数说明与调整建议

### `Band-pass low cutoff (Hz)`

默认：`25 Hz`

作用：控制检测滤波器的低频下限。

调高时：

- 会去掉更多低频漂移和手持/环境震动。
- 但可能损失低沉的心跳成分。

调低时：

- 会保留更多低频心跳能量。
- 但也更容易引入呼吸、触碰、麦克风漂移等低频噪声。

建议：

- 普通心跳录音先用 `20 Hz`。
- 如果 envelope 被低频漂移带偏，调到 `30-40 Hz`。
- 如果心跳很低沉且检测不到，调到 `10-15 Hz`。

### `Band-pass high cutoff (Hz)`

默认：`160 Hz`

作用：控制检测滤波器的高频上限。

调高时：

- 会保留更多敲击、摩擦、尖锐瞬态。
- 可能让 peak 更明显，也可能引入噪声。

调低时：

- 会让检测更平滑。
- 但可能削弱 S1/S2 的清晰边缘。

建议：

- 默认 `180 Hz` 适合多数心跳。
- 噪声很多时降到 `120-150 Hz`。
- 心跳很闷、不清晰时可试 `200-250 Hz`。

### `Envelope low-pass (Hz)`

默认：`6 Hz`

作用：对 envelope 做低通平滑。

调高时：

- envelope 反应更快。
- 更容易保留 S1/S2 双峰。
- 可能误检更多峰。

调低时：

- envelope 更平滑。
- 更容易把一组心跳合成一个主峰。
- 但过低会让峰位置变钝。

建议：

- 双峰误检明显时调低到 `4-5 Hz`。
- 心跳峰太平、检测不到时调高到 `8-10 Hz`。

### `Minimum plausible BPM`

默认：`40 BPM`

作用：BPM 搜索范围下限。

如果真实心率低于这个值，autocorrelation 可能估计错误。

建议：

- 成年人静息心跳一般 `50-100 BPM`。
- 如果录音是很慢的心跳，可以调到 `30-35 BPM`。
- 如果只想避免慢速误判，可以调到 `50 BPM`。

### `Maximum plausible BPM`

默认：`140 BPM`

作用：BPM 搜索范围上限。

如果真实心率高于这个值，系统可能找不到正确周期。

建议：

- 普通心跳用 `120-140 BPM`。
- 运动后心跳可调到 `160-180 BPM`。
- 如果 S1/S2 双峰被误认为两次心跳，可以适当降低上限。

### `Peak prominence`

默认：`0.12`

作用：控制 peak 必须比周围明显多少才算心跳。

调高时：

- 检测更严格。
- 可以减少噪声误检。
- 但可能漏掉弱心跳。

调低时：

- 检测更敏感。
- 可以找回弱心跳。
- 但可能把噪声或 S2 也当成心跳。

建议：

- beat count 太多：调高，例如 `0.18-0.30`。
- beat count 太少：调低，例如 `0.05-0.10`。

### `Peak height percentile`

默认：`65`

作用：设置 peak 的最低高度阈值，基于 envelope 分位数。

调高时：

- 只接受更高的峰。
- 更适合噪声多、心跳强的录音。

调低时：

- 弱峰也能被接受。
- 更适合音量不稳定的录音。

建议：

- 噪声误检多：调到 `70-80`。
- 心跳强弱变化大：调到 `50-60`。

### `Double-peak suppression`

默认：`0.65`

作用：抑制 S1/S2 双峰误检。它会根据估计周期设置相邻 peak 的最小距离。

调高时：

- 相邻 peak 必须离得更远。
- 更能防止 S1/S2 被拆成两次心跳。
- 但真实快心率可能被漏掉。

调低时：

- 允许更近的 peak。
- 对快心率更友好。
- 但更容易双峰误检。

建议：

- beat count 约为真实值两倍：调高到 `0.75-0.90`。
- 快心率漏检：调低到 `0.45-0.60`。

### `Target loop length (beats)`

默认：`4`

作用：选择 best_loop 时希望包含几个心跳周期。

调高时：

- loop 更长，更自然。
- 但更难找到完全稳定的窗口。

调低时：

- loop 更短，更容易稳定。
- 但重复感更明显。

建议：

- 后续做音乐节奏铺底：`4` 或 `8`。
- 只想要短素材：`2`。
- 录音很稳定：可以试 `8-12`。

### `Loop edge fade (ms)`

默认：`12 ms`

作用：对 best_loop 开头和结尾做短 fade，减少循环时的 click。

调高时：

- click 更少。
- 但会软化每次 loop 边界。

调低时：

- 保留更多原始瞬态。
- 但可能有边界爆音。

建议：

- 有 click：调到 `20-40 ms`。
- 心跳 transient 很重要：保持 `5-12 ms`。

## 7.1 语音和噪声抑制参数

### `Enable speech/noise suppression`

默认开启。关闭后系统只做带通和节拍分析，适合需要保留原始听诊器声音、或想比较算法前后差异的情况；但 `cleaned.wav` 和 `best_loop.wav` 会更容易保留说话声。

### `Suppression strength`

- `Mild`：拍间衰减约 `18 dB`，对微弱心音最保守，适合录音本身较干净。
- `Balanced`：默认，拍间衰减约 `28 dB`，兼顾说话声压制和 S1/S2 保留。
- `Strong`：拍间衰减约 `36 dB`，并加强持续语音/环境声抑制；适合明显有人说话的录音，但弱心音可能变轻。

这三个预设会写入 `processing_parameters.json`：`hpss_margin` 控制对持续谐波成分的抑制，`spectral_reduction_strength` 控制频谱门限强度，`beat_gate_pre_ms` / `beat_gate_post_ms` 控制每拍前后被完整保留的时间窗，`between_beat_attenuation_db` 控制拍间衰减量。

### 新版三个 WAV 输出的关系

- `filtered_detection.wav`：内部检测信号经过限带、去持续语音、自适应降噪及导出软门控后的可听诊断版本；诊断图第二行展示的是门控前、真正用于找 beat 的信号。
- `cleaned.wav`：在上述信号上，再按检测到的心跳时刻施加软门控的可听心音版本。
- `best_loop.wav`：从 `cleaned.wav` 的最稳定 IBI 窗口截取，并在边界做 zero-crossing 和 fade 的后续混音素材。

诊断图现在有五部分：原始波形、抑制语音后的检测信号、最终 cleaned 音频、envelope/beat/loop，以及 local BPM。页面还会显示 `Beat-window coverage`，表示有多少录音时长处在保留心音的时间窗内。

## 8. 如何判断结果好不好

推荐检查顺序：

1. 看页面上的 `Estimated BPM` 是否接近预期心率。
2. 看 `Detected Beats` 是否和音频时长匹配。例如 13 秒、70 BPM，大约应该有 15 次心跳。
3. 打开 `diagnostic_plot.png`，确认红色 beat 线是否落在真实心跳峰上。
4. 看 `IBI Std`。越小表示节奏越稳定。
5. 试听 `best_loop.wav`，确认 loop 没有明显断裂或爆音。
6. 如果检测不准，优先调 `Peak prominence`、`Double-peak suppression` 和 BPM 范围。

## 9. 已验证样例

本地已在 `heartbeat` conda 环境验证：

```text
C:\Users\33480\Desktop\20260710231622\202607071630425298.wav
C:\Users\33480\Desktop\20260710231622\202607081521080317.wav
C:\Users\33480\Desktop\20260710231622\202607091053425340.wav
```

命令行输出：

```text
202607071630425298.wav: BPM=69.6, beats=15
202607081521080317.wav: BPM=75.5, beats=13
202607091053425340.wav: BPM=69.7, beats=15
```

另将第一段 WAV 转成 MP3 后验证：

```text
stage1_mp3_validation_input.mp3: BPM=69.6, beats=15
```

## 10. 常见问题

### Streamlit 要求输入 email，然后 bat 失败

使用当前仓库里的 `run_app.bat` 或 `run_app_lan.bat`。脚本已经加入：

```text
--browser.gatherUsageStats false
```

这会跳过 Streamlit 首次启动 email prompt。

### 页面能在本机打开，但虚拟网卡访问不了

如果：

```text
http://127.0.0.1:8501
```

可以打开，但：

```text
http://虚拟网卡IPv4:8501
```

打不开，通常不是 App 问题。检查：

- 是否使用 `run_app_lan.bat`
- Windows 防火墙是否允许 TCP 8501
- 虚拟网卡 IPv4 是否正确
- 访问方是否能 ping 通该 IPv4

### beat 数量明显太多

可能原因：

- S1/S2 双峰被当成两次心跳
- peak threshold 太低
- 高频噪声太多

优先尝试：

- 增大 `Double-peak suppression`
- 增大 `Peak prominence`
- 增大 `Peak height percentile`
- 降低 `Band-pass high cutoff`

### beat 数量明显太少

可能原因：

- peak threshold 太高
- envelope 太平滑
- BPM 搜索范围不合适

优先尝试：

- 降低 `Peak prominence`
- 降低 `Peak height percentile`
- 增大 `Envelope low-pass`
- 放宽 `Minimum plausible BPM` / `Maximum plausible BPM`

### best_loop 有爆音

优先尝试：

- 增大 `Loop edge fade`
- 换更长或更短的 `Target loop length`
- 检查原始录音是否有突发噪声

## 11. 项目结构

```text
D:\Heartbeat
  app.py                         Streamlit Web App（音乐视频 + Stage 1 两个标签页）
  heartbeat_preprocessor/
    core.py                      心跳音频预处理、BPM、peak、loop、导出逻辑
  legasynth/
    pipeline.py                  完整流程编排：Stage1 -> 情绪 -> 混音（原声+心跳）-> 视频
    emotion.py                   功能 A：心跳 -> 情绪(valence/arousal) -> 视觉风格
    video_audio.py               从 mp4 提取歌曲音频并估计歌曲 BPM/beat
    mixing.py                    心跳 loop time-stretch 到歌曲 BPM 并混音
    video_render.py              功能 B：心跳驱动的调色 / 缩放脉冲 / 闪白 / 按拍剪辑
  scripts/
    process_files.py             命令行入口（Stage-1 批处理 + 完整音乐视频流程）
  environment.yml                conda 环境定义
  requirements.txt               pip 依赖列表
  setup_env.bat                  创建或更新 heartbeat 环境
  run_app.bat                    本机启动
  run_app_lan.bat                LAN / 虚拟网卡启动
  outputs/                       本地输出目录，不上传 Git
```

## 12. 完整音乐视频流程（已实现）

完整流程已经落地，端到端把心跳 + 歌曲 mp4 变成 `final_video.mp4`，**保留歌曲原声、心跳作鼓点铺底**：

- 上传英文歌曲音乐视频 `.mp4`，提取歌曲音频、估计歌曲 BPM 和 beat track
- 将 `best_loop.wav` time-stretch 到歌曲 BPM，铺在歌曲原声下方混音（不替换人声）
- 生成心跳驱动的视觉效果，导出 `final_video.mp4`

两个进阶功能（advanced features）：

- **A 情绪条件化（emotional conditioning）**：心跳 HRV -> 情绪 -> 视频调色与剪辑节奏（[legasynth/emotion.py](legasynth/emotion.py)）
- **B 心跳驱动剪辑**：随心跳缩放/闪白/按拍剪辑（[legasynth/video_render.py](legasynth/video_render.py)）

未来可继续增强的方向：句首献词标题卡、角落心率徽标、主歌/副歌自适应特效、剪辑拍柔性转场。

命令行示例（完整流程）：

```powershell
conda run -n heartbeat python scripts\process_files.py `
  --heartbeat "C:\path\heartbeat.wav" `
  --video "C:\path\song.mp4" `
  --out outputs\mv --title "For Mom"
```
