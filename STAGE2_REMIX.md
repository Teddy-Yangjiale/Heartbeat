# Stage 2：歌曲对齐的心跳鼓点混音

## 1. 本阶段已经实现什么

Stage 2 保留歌曲原有速度和节拍结构，只改变心跳信号。处理链路是：

```text
raw heartbeat WAV/MP3
-> Stage 1 cleanup / BPM / beat detection / best_loop.wav

song WAV/MP3
-> BeatNet beat / downbeat / meter / tempo map
-> librosa fallback or manual BPM / phase override
-> beat grid / bar labels

best_loop.wav + song beat grid
-> phase-vocoder time stretch without pitch shift
-> begin at song first-beat anchor
-> stretch each loop block to exact dynamic beat-grid boundaries
-> heartbeat_layer.wav
-> mix with the unchanged song
-> final_audio.wav / final_audio.mp3
```

BeatNet 成功时使用第一个 downbeat 作为相位锚点；librosa 回退只能提供第一个检测 beat。弱节奏、自由速度或切分较强的歌曲仍需试听，程序允许手工覆盖 `BPM` 和 `first beat`。

## 2. 最快启动方式

先按主 README 创建 `heartbeat` conda 环境。推荐启动多轨 Studio：

```powershell
run_studio.bat
```

打开：

```text
http://127.0.0.1:8503
```

如果只需要简单的上传表单，也可运行 `run_remix_app.bat` 并访问 `http://127.0.0.1:8504`。

操作顺序：

1. 上传原始心跳 WAV/MP3。
2. 上传目标歌曲 WAV/MP3。
3. 选择拍号；普通流行歌通常先试 `4`，华尔兹通常先试 `3`。
4. 先保留自动 BPM 和自动 first beat，生成一次结果。
5. 试听 `Aligned heartbeat layer` 和 `Final mix`。
6. 如果心跳始终落在两个鼓点之间，关闭自动 phase，填写正确的首拍秒数后重跑。
7. 如果越到后面越偏，说明 BPM 估计有误；关闭自动 BPM，填入 DAW 或人工确认的 BPM 后重跑。

## 3. 命令行使用

最小命令：

```powershell
conda run --no-capture-output -n heartbeat python scripts\remix_song.py `
  --heartbeat "C:\path\heartbeat.wav" `
  --song "C:\path\song.mp3" `
  --out "outputs\remix"
```

手工确认歌曲为 120 BPM、4/4，第一拍在 0.37 秒：

```powershell
conda run --no-capture-output -n heartbeat python scripts\remix_song.py `
  --heartbeat "C:\path\heartbeat.wav" `
  --song "C:\path\song.wav" `
  --out "outputs\remix" `
  --bpm 120 `
  --first-beat 0.37 `
  --beats-per-bar 4 `
  --loop-beats 4 `
  --heartbeat-gain-db -15
```

常用参数：

| 参数 | 含义 |
| --- | --- |
| `--bpm` | 覆盖自动歌曲 BPM；未提供时自动估计 |
| `--first-beat` | 心跳层开始进入的歌曲相位锚点，单位秒 |
| `--beats-per-bar` | 用于 beat grid 的小节标注，不会改变歌曲 |
| `--loop-beats` | 每个心跳 loop 包含的心跳间隔数 |
| `--heartbeat-gain-db` | 心跳层混音增益，默认 `-15 dB` |

## 4. 输出文件

每次运行产生：

```text
outputs/remix/<heartbeat>__<song>/
  inputs/
  heartbeat_analysis/
    <heartbeat>/best_loop.wav
    <heartbeat>/tempo_summary.json
    ...Stage 1 outputs
  song_analysis/
    song_analysis.json
    song_beat_grid.csv
    detected_song_beats.csv
    song_tempo_map.csv
  remix/
    heartbeat_layer.wav
    final_audio.wav
    final_audio.mp3
    aligned_heartbeat_beats.csv
    mix_report.json
  reports/run_report.json
  all_outputs.zip
```

其中：

- `song_analysis.json`：backend、歌曲 BPM、beat/downbeat、meter、first-beat phase、confidence 和完整 beat grid。
- `song_beat_grid.csv`：每拍的时间、bar index 和 beat-in-bar。
- `song_tempo_map.csv`：相邻 beat 的间隔、瞬时 BPM 和平滑局部 BPM。
- `heartbeat_layer.wav`：已经按歌曲 BPM 和首拍相位对齐的纯心跳层，便于在 DAW 单独检查。
- `final_audio.wav`：原歌曲加心跳层；歌曲本身没有被 time-stretch。
- `mix_report.json`：心跳源 loop 时长、目标 loop 时长、stretch rate、增益和导出路径。

## 5. 人声/伴奏分离和旋律

这部分采用可选后端，避免基础安装强制下载大型 PyTorch 模型。

安装隔离的 BeatNet + Demucs + Basic Pitch 模型环境：

```powershell
setup_music_env.bat
```

首次运行会下载所选模型。启用分离和旋律提取：

```powershell
conda run --no-capture-output -n heartbeat python scripts\remix_song.py `
  --heartbeat "C:\path\heartbeat.wav" `
  --song "C:\path\song.mp3" `
  --separate-stems `
  --extract-melody
```

输出包括：

- `vocals.wav`：Demucs 人声 stem。
- `no_vocals.wav`：Demucs 伴奏 stem。
- `vocal_note_events.csv` / `vocal_melody.mid`：Basic Pitch 预训练模型输出；模型失败时回退为 pYIN 的 `vocal_melody.csv` 逐帧轨迹。
- `vocal_melody_summary.json`：旋律覆盖率和中位音高等摘要。

这里的旋律是单声部基频轨迹。和声、混响、说唱、强伴奏泄漏会降低准确率，最终用于歌声合成前还需要音符切分、歌词音节对齐和人工修正。

## 6. 中文翻唱尚缺的真实依赖

歌词翻译与中文歌声合成尚未标记为完成，因为仅有歌曲音频不能稳定、合法地自动获得准确歌词，也没有仓库内已选定的歌声模型和授权音色。下一阶段至少需要确定：

1. 歌词输入来源：用户提供原歌词，还是接入有授权的歌词服务。
2. 翻译目标：逐字翻译，还是按音节数、重音和押韵重新填词。
3. 人声模型：DiffSinger、OpenUtau/ENUNU、ACE Studio 或其他已授权后端。
4. 音色：自有录音训练、许可声库，或无身份模仿的通用音色。
5. 对齐策略：`vocal_melody.csv + 中文音素/音节 + note segmentation` 的中间格式。

在这五项确定前，直接接一个文本翻译 API 并不能得到能唱、合拍、自然的中文歌词；直接声称已经“合成中文歌声”也不可验证。

## 7. 验证

运行全部测试：

```powershell
conda run --no-capture-output -n heartbeat python -m unittest discover -s tests -v
```

Stage 2 测试会检查：

- BPM/first-beat override 是否生成正确 phase-anchored beat grid。
- 3/4 等拍号的小节标注是否正确。
- 心跳层在 first beat 之前是否为静音。
- loop 是否按 BPM 变为精确目标时长。
- aligned heartbeat CSV 是否从指定首拍开始。
