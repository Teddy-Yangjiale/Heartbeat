# LegaSynth Demo · 治疗师协作式音乐灵感候选

这个文件夹是一次**完整流程的展示**：用一段**真实心跳**生成 4 个可编辑的**音乐片段候选**，供治疗师与患者现场共同作曲时挑选、否决、组合、即兴。

## 本次运行的参数

| 项 | 值 |
|---|---|
| 心跳输入 | `inputs/heartbeat_202607071630425298.wav`（真实录音） |
| 心跳分析 | BPM ≈ 69 · 检测到 14 次心跳 · 录音质量 good |
| 个人节奏动机 | `[0.75, 0.75, 1.5, 1.5, 1.0, 1.0, 0.75, 1.5]`（由 IBI 量化得到） |
| 患者选择的音级 | `4, 2, 7`（有序） |
| 治疗师控制 | Key C · Major · 4 小节 · 性格 Reflective · 速度用心跳 BPM · 心跳影响度 0.65 |
| 输出 | 4 个候选（每个 ≈ 13.8s） |

## 文件夹内容

```
demo/
  inputs/
    heartbeat_202607071630425298.wav      原始心跳素材（唯一输入）
  candidates/
    candidate_01.wav / .mid / _notes.csv  Pulse motif        —— 最忠实保留心跳律动
    candidate_02.wav / .mid / _notes.csv  Call and response  —— 动机 + 变化应答
    candidate_03.wav / .mid / _notes.csv  Expanded arc       —— 拉成更长的上行再返回
    candidate_04.wav / .mid / _notes.csv  Spacious variation —— 更多留白，给即兴留空间
    candidate_manifest.json               每个候选怎么来的 + scope 说明
    phrase_candidates.zip                 全部候选打包
  videos/
    candidate_01_Pulse_motif.mp4          钢琴卷帘 + 播放头 + 音频（可放幻灯片）
    candidate_02_Call_and_response.mp4
    candidate_03_Expanded_arc.mp4
    candidate_04_Spacious_variation.mp4
  screenshots/
    candidates_overview.png               4 候选 2×2 总览（红=主旋律 青=和声 蓝=低音）
  README.md
```

**候选视频**里的钢琴卷帘：横轴是小节/拍，纵轴是音高;**红色**是主旋律、**青色**是和声、**蓝色**是低音;红色竖线是播放头。四个候选并排对比见 `screenshots/candidates_overview.png`。

**MIDI 是主要交付格式**——治疗师可以直接把 `.mid` 拖进钢琴/DAW 继续修改、即兴。WAV 只是快速试听。

> 这些是**未完成的草稿/灵感起点**，不是成品作品，也不是临床建议。治疗师和患者保留选择权和作者身份。心跳只用作**节奏素材**，不推断情绪或临床状态。

## 如何复现 / 查看

启动网页应用（主标签 "Co-composition studio"）：

```powershell
conda run -n heartbeat python -m streamlit run app.py --server.port 8502
```

然后浏览器打开 `http://localhost:8502`：上传 `inputs/` 里的心跳 → 在 "Chosen scale degrees" 填 `4,2,7` → 设治疗师控制 → 点 **Generate candidate ideas** → 并排试听/下载 4 个候选。

本 demo 的候选与视频由 `scripts`/生成脚本一次性产出，参数如上表，随机种子由「心跳节奏 + 请求参数」哈希决定，因此**可复现**。
