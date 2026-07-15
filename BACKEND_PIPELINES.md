# Backend Pipelines V2

本阶段先把后端变成可验证、可恢复的处理系统；Studio 前端仍可继续使用兼容接口，但不再把所有步骤绑成一个不可观察的长请求。

## 1. 两个主 Pipeline

### Heartbeat

```text
听诊器 WAV/MP3
-> mono / DC removal / heart-band filter / speech-noise suppression
-> multi-window autocorrelation BPM consensus
-> envelope peaks + template confirmation
-> cardiac-cycle regularization (漏拍恢复带 detected/recovered 标记)
-> beat/IBI/cycle quality gate
-> best-loop 候选排名
-> global BPM 一致性、loop IBI CV、recovered 比例严格验收
-> best_loop.wav + JSON/CSV/PNG
```

录音可能得到 `ok` 或 `manual_review`。质量分高不代表一定可混音；`recording_quality.is_recommended_for_loop` 和 `best_loop.validation.status` 都必须通过。被恢复的心拍比例超过 25% 时不会自动批准，即使局部 loop 本身看起来稳定。

### Song

```text
歌曲 WAV/MP3
-> BeatNet CRNN + DBN (优先)
-> beat number / downbeat / meter / local tempo map
-> librosa dynamic beat fallback
-> BPM、相位、confidence、tempo stability 与人工复核门控
-> song_analysis.json + beat grid / detected beats / tempo-map CSV
```

BeatNet 成功时自动相位锚点使用第一个 downbeat，动态网格从该 downbeat 开始。若用户同时指定 BPM 和 first beat，覆盖值是权威网格，程序会跳过不必要的 BeatNet 推理。

歌曲阶段仍会计算 librosa tempo 作为独立校验。BeatNet 与 librosa 在考虑 half/double-time 后仍相差超过 12% 时，阶段标记为 `manual_review`；不会静默选择一个看似精确的 BPM。

## 2. 预训练模型

模型运行在独立 `heartbeat-music` 环境，避免 BeatNet/madmom 的 Python 3.9、NumPy 1.20 约束破坏 Python 3.11 Web API。

```powershell
cd D:\Heartbeat
setup_music_env.bat
```

该环境提供：

- BeatNet：beat、downbeat、meter 和动态 tempo map。
- Demucs `htdemucs`：`vocals.wav` 与 `no_vocals.wav`。
- Basic Pitch：从 vocal stem 导出 note-events CSV 和 MIDI。
- pYIN：Basic Pitch 不可用或失败时的确定性逐帧旋律回退。

第一次安装会下载 CPU PyTorch；脚本也会预取约 80 MB 的 Demucs 权重。服务通过 `GET /api/health` 报告每个 backend，而不是把缺少模型当作整个任务失败。

## 3. Job 与阶段 API

上传一完成就原子写入 `manifest.json`，其中包括输入 SHA-256、阶段状态、backend、warnings、error、相对输出路径和 render revisions。

```text
POST /api/jobs
POST /api/jobs/{id}/heartbeat/analyze
POST /api/jobs/{id}/song/analyze
POST /api/jobs/{id}/stems/analyze
POST /api/jobs/{id}/render
GET  /api/jobs/{id}
GET  /api/health
```

阶段状态包括 `pending/running/ok/manual_review/failed/unavailable/skipped`。Demucs 或 Basic Pitch 不可用时，已经完成的心跳和歌曲结果仍保留。`render` 默认只接受 heartbeat/song 均为 `ok`；人工听审后可显式提交 `allow_manual_review=true`。

旧 Studio 继续使用 `POST /api/process`。这个兼容入口内部也先落盘 manifest，再按独立阶段执行。

## 4. 动态混音

歌曲本身不变速。心跳 loop 按歌曲实际 beat grid 每 N 拍分段 time-stretch，并把每段边界固定到对应 beat，避免 constant-BPM 循环在长曲或 live tempo 上累计漂移。报告中的 `grid_mode` 为：

- `dynamic_tempo_map`：使用 BeatNet 实际网格；
- `constant_grid`：使用人工 BPM/phase 或可靠回退网格。

每次渲染写入独立 `remix/revision_NNN`，并记录目标 BPM、相位、心跳增益、峰值和输出文件。

## 5. 验证

普通回归：

```powershell
conda run --no-capture-output -n heartbeat python -m unittest discover -s tests -v
```

本机三条已标注真实心跳回归：

```powershell
$env:HEARTBEAT_REAL_FIXTURE_DIR='C:\Users\33480\Desktop\20260710231622'
conda run --no-capture-output -n heartbeat python -m unittest tests.test_real_heartbeat_regression -v
```

完整真实模型链：

```powershell
conda run --no-capture-output -n heartbeat python scripts\validate_backend_v2.py `
  "C:\path\heartbeat.wav" "C:\path\song.wav" "outputs\backend_v2_validation" `
  --allow-manual-review
```

验收时优先检查 `manifest.json` 的五个阶段状态、心跳 loop validation、歌曲 backend/grid mode、旋律 note-event 数，以及 remix 的 tempo-mapped segment 数。
