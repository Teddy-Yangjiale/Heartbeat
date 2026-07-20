# Heartbeat Music Processor

Heartbeat Music Processor 是一个完整的心跳驱动音乐处理网页。用户上传一段心跳 WAV 和一首歌曲 WAV，系统会自动完成：

1. 心跳降噪、节律检测、质量评估和真实连续周期选取；
2. 歌曲 BPM、第一拍和动态节拍网格分析；
3. 将干净心跳周期按歌曲节拍进行半速、原速、双速或小节级排布；
4. 对指定时间区域单独调整歌曲音量、心跳音量、心跳密度和周期适配；
5. 自动响度平衡、心跳触发的低频闪避、最终 LUFS 调整与峰值保护；
6. 在线试听并导出最终音乐、独立轨、检查轨和完整处理报告。

本程序不是医疗诊断工具。心跳质量判断只用于决定录音是否适合安全降噪和音乐制作。

## 从零启动

推荐 Python 3.11 和 Conda：

```powershell
cd D:\Heartbeat
conda env create -f environment.yml
conda activate heartbeat
python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
```

已有环境可以更新：

```powershell
conda env update -f environment.yml --prune
```

也可以双击 `run_app.bat`。默认地址为 `http://127.0.0.1:8501`。

## 网页工作流

### 1. 上传输入

- 心跳：PCM/浮点 WAV，建议约 15 秒，最大 25 MB、30 秒；单声道和立体声都可，预处理时会转成单声道。
- 歌曲：WAV，最大 100 MB、5 分钟；最终渲染保留歌曲采样率和单/立体声结构。

网页不依赖系统 FFmpeg，因此当前公开版不接受 MP3。需要处理 MP3 时请先在本地转换为 WAV。

### 2. 分析

点击“分析两路音频”。系统会显示心跳 BPM、心跳质量、歌曲 BPM、歌曲第一拍、节拍置信度和歌曲时长。

歌曲分析默认使用 librosa 动态节拍网格。弱节奏、自由速度或切分较强的歌曲应试听 `debug_click_mix.wav`，必要时：

- 勾选“手动指定歌曲 BPM”；
- 勾选“手动指定歌曲第一拍”；
- 或启用固定 BPM 网格。

若心跳被判定为 `needs_rerecording`，渲染按钮会被锁定。用户试听并明确确认后仍可覆盖此安全门控。

### 3. 全局参数

| 参数 | 作用 |
|---|---|
| 全局心跳密度 | 自动、小节一次、每两拍一次、每拍一次、每拍两次 |
| 周期适配 | `gap` 尽量保留原心音并补静音；`stretch` 将周期拉伸到完整节拍区间 |
| 每小节拍数 | 定义小节级排布和检查轨的重拍位置 |
| 心跳出现范围 | 让心跳只在歌曲的某个整体范围内出现 |
| 歌曲/心跳整体增益 | 在自动响度平衡结果上继续微调 |
| 心跳相对响度 | 正值使心跳比歌曲活跃段更突出 |
| 低频闪避 | 心跳出现时仅压低歌曲低频，为心音留出空间 |
| 最终目标 LUFS | 控制母带目标响度 |
| 输出峰值上限 | 防止最终 WAV 削波，默认 `-1 dBFS` |

### 4. 特定区域编辑

在区域表格底部点击 `+` 添加时间段。每一行支持：

- 开始时间与结束时间；
- 歌曲增益；
- 心跳增益；
- 独立心跳密度；
- 独立周期适配方式；
- 区域边界淡化时间。

把某一区域的心跳密度设为 `mute`，可以制作只有歌曲、没有心跳的间奏。区域不能重叠，避免多套自动化同时修改同一位置。

时间线中灰色竖线是歌曲节拍，橙色竖线是每四拍参考线，彩色背景是用户定义的编辑区域。

### 5. 渲染

- “生成前 45 秒试听”：快速检查节拍、音量和区域设置；短于 45 秒的歌曲会生成完整试听。
- “生成整首最终音乐”：按完全相同的参数处理整首歌。

每次修改 BPM、第一拍、心跳密度或区域设置后，应重新渲染并试听检查轨。

## 输出文件

| 文件 | 含义 |
|---|---|
| `final_mix.wav` | 可交付的最终歌曲与心跳混音 |
| `heartbeat_aligned.wav` | 已对齐并经过最终母带增益的独立心跳轨 |
| `song_processed.wav` | 经过区域增益、低频闪避和母带增益的歌曲轨 |
| `debug_click_mix.wav` | 歌曲加节拍点击声，用于检查 BPM 和第一拍 |
| `mix_report.json` | 输入分析、心跳质量、实际增益、区域设置、母带响度和峰值 |
| `heartbeat_timeline.csv` | 每个心跳事件的时间、密度模式、适配方式和所属区域 |
| `region_edits.json` | 可复查的区域编辑参数 |
| `heartbeat_music_project.zip` | 上述全部工程文件 |

心跳预处理阶段始终使用 `cleanest_heartbeat_loop.wav` 进入混音，不使用仅供手机试听的 `cleanest_heartbeat_loop_loud.wav`，避免重复增益和软削波。

## 处理架构

```text
heartbeat.wav
  -> process_audio_bytes()
  -> attenuation-only denoising
  -> quality gate
  -> cleanest real consecutive cycles
                                      -> pulse schedule
song.wav -> librosa beat grid --------> region automation
                                      -> loudness balance
                                      -> low-band ducking
                                      -> master gain + peak protection
                                      -> final_mix.wav
```

歌曲本身不会被整体 time-stretch。只有心跳周期会根据相邻歌曲节拍的实际长度进行局部适配，因此不会因为重复一个固定长度循环而在长歌曲中不断积累对齐误差。

## 隐私和部署

Streamlit 上传文件会进入服务器内存，不是只存在于用户浏览器。本项目的网页代码不会主动把上传文件或生成文件写入持久化目录；用户应在当前会话中直接下载结果。

Streamlit Community Cloud 部署：

1. 选择本仓库和目标分支；
2. Entrypoint 选择根目录 `app.py`；
3. Python 选择 3.11；
4. 依赖由 `environment.yml` 安装；
5. `.streamlit/config.toml` 将单文件上传上限设为 120 MB，应用内部将歌曲限制为 100 MB。

公开云端同时处理多首长歌曲可能消耗大量内存。当前版本限制歌曲为 5 分钟；如果未来需要多用户并发，应保持此网页入口不变，将 `music_processor/core.py` 放到独立异步 worker 中执行。

## 测试

```powershell
conda run --no-capture-output -n heartbeat python -m unittest discover -s tests -v
```

测试覆盖：

- 原心跳预处理、节律保持、局部污染门控和输出电平；
- 手动 BPM 与第一拍网格；
- 局部双速和静音区域；
- 心跳到歌曲的端到端渲染；
- 最终 WAV、独立轨、时间线、报告和 ZIP；
- 最终峰值上限。

## 已知边界

- librosa 无法保证第一拍一定是音乐上的 downbeat；检查点击轨和手动校正仍是产品流程的一部分。
- 自由速度、无明显打击乐、复杂切分或中途拍号变化的歌曲可能需要分段手动设置。
- 单通道心跳中与 S1/S2 完全重叠的摩擦或机械噪声无法可靠分离；此时应重新录制。
- 目标 LUFS 和峰值上限冲突时，峰值安全优先，最终响度可能低于目标。
