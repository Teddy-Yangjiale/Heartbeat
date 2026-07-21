# Heartbeat Music Processor

Heartbeat Music Processor 是一个完整的心跳驱动音乐处理网页。用户上传一段心跳 WAV 和一首 WAV/MP3 歌曲，系统会自动完成：

1. 自动工频抑制、心动静息相位噪声建模、节律检测、质量评估和多真实周期素材池；
2. 歌曲 BPM、动态节拍网格、小节重拍、真实低频底鼓 onset 和局部能量分析；
3. 检测每个心跳周期真正的 S1 onset，保留 S1/S2 有效段，并用可调吸附、微时差和 Swing 自然排布；
4. 对指定时间区域单独调整歌曲音量、心跳音量、心跳密度和周期适配；
5. 自动响度平衡、心跳触发的低频闪避、最终 LUFS 调整与峰值保护；
6. 原声、电影感、Lo-fi、Trip-hop、极简电子、暗氛围预设；
7. 在线试听并导出 FLAC、MP3、WAV16 或 WAV24；独立轨、检查轨和工程 ZIP 可按需生成。
8. 兼容 `heartbeat_sync` 迁移合同：混合歌曲内容裁剪、4 周期连续循环、片头/片尾脉冲、S1/S2 检查轨、五文件 24-bit 工程包，并可连接官方 Windows 黑盒 CLI。

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
- 歌曲：WAV 或 MP3，最大 100 MB、5 分钟；解码后保留歌曲采样率和单/立体声结构。

MP3 由 `soundfile` 随附的 `libsndfile` 直接解码，网页不依赖系统 FFmpeg。

### 2. 分析

点击“分析两路音频”。系统会显示心跳 BPM、心跳质量、歌曲 BPM、歌曲第一拍、节拍/重拍置信度和歌曲时长。分析结果只保存压缩波形概览和节拍元数据，不在会话中保留完整解码歌曲。

歌曲分析默认使用 librosa 动态节拍网格。弱节奏、自由速度或切分较强的歌曲应试听 `debug_click_mix.wav`，必要时：

- 勾选“手动指定歌曲 BPM”；
- 勾选“手动指定歌曲第一拍”；
- 勾选“手动指定第一小节重拍”并填写拍号；
- 或启用固定 BPM 网格。

若心跳被判定为 `needs_rerecording`，渲染按钮会被锁定。用户试听并明确确认后仍可覆盖此安全门控。

“歌曲内容范围与 heartbeat_sync 后端”中可以：

- 使用强/保守双阈值自动裁剪歌曲首尾静音，或手动覆盖歌曲起点与终点；
- 云端使用原生兼容引擎；配置 `HEARTBEAT_SYNC_REPO` 后选择官方 `heartbeat_sync` CLI；
- 官方 CLI 的 `auto` 后端优先 Beat This，失败时按原仓库合同回退 librosa，并把警告显示给用户；
- 预览最多处理前 45 秒有效内容，整首最终渲染不会意外携带这个限制。

### 3. 全局参数

| 参数 | 作用 |
|---|---|
| 心跳节奏角色 | 音乐感知自动、小节重拍、底鼓角色、反拍、每拍及传统密度模式 |
| 音乐风格预设 | 同时设置心跳密度、律动松紧、音色、空间感和闪避；仍可逐项覆盖 |
| 周期适配 | `preserve` 只保留 S1/S2 有效段并留白；`gap` 保留完整周期；`stretch` 受限拉伸 |
| 每小节拍数 | 定义小节级排布和检查轨的重拍位置 |
| 心跳出现范围 | 让心跳只在歌曲的某个整体范围内出现 |
| 歌曲/心跳整体增益 | 在自动响度平衡结果上继续微调 |
| 心跳相对响度 | 正值使心跳比歌曲活跃段更突出 |
| 低频闪避 | 心跳出现时仅压低歌曲低频，为心音留出空间 |
| 最终目标 LUFS | 控制母带目标响度 |
| 输出峰值上限 | 防止最终音频削波，默认 `-1 dBFS` |
| 最低/最高心跳密度 | 限制自动调度的局部脉冲速度，避免过密或过疏 |
| 听感偏移 | 在 S1 已准确落拍的基础上进行正负毫秒级人工微调 |
| 节拍吸附强度 | 100% 紧贴歌曲网格；降低可保留心跳自身的自然呼吸感 |
| 自然微时差 / Swing | 加入可复现的小幅 timing variation，避免机械重复 |
| 存在感 / 饱和 / 空间感 | 在线安全的心跳音色与短空间塑形，不替换原始心音 |
| 乐段动态强度 | 根据歌曲局部能量改变心跳响度，并在低能量段自动疏化 |
| 心跳渐入/渐出 | 只控制心跳层的长包络，不改变歌曲本身 |
| 最大拉伸倍率 | 仅限制手动选择 `stretch` 时的 time-stretch，默认 `1.10x` |
| 每个循环的真实周期数 | 默认选择质量最好的连续 4 个周期，降低单周期重复感 |
| 歌曲前/后心跳次数 | 默认各 4 次；歌曲整体后移，片尾在歌曲结束后继续 |
| 片头/片尾增强 | 默认比歌曲内心跳高 4 dB，单独控制边缘听感 |
| 最终文件格式 | 默认 MP3；另可选无损 FLAC16、WAV16 和制作母版 WAV24 |

### 4. 特定区域编辑

在区域表格底部点击 `+` 添加时间段。每一行支持：

- 开始时间与结束时间；
- 歌曲增益；
- 心跳增益；
- 独立心跳密度；
- 独立周期适配方式；
- 区域边界淡化时间。
- 独立听感偏移和自然微时差。

把某一区域的心跳密度设为 `mute`，可以制作只有歌曲、没有心跳的间奏。区域不能重叠，避免多套自动化同时修改同一位置。

时间线中灰色竖线是歌曲节拍，橙色竖线是推断或手动指定的小节重拍，彩色背景是用户定义的编辑区域。

### 5. 渲染

- “生成前 45 秒试听”：快速检查节拍、音量和区域设置；短于 45 秒的歌曲会生成完整试听。
- “生成整首最终音乐”：按完全相同的参数处理整首歌。

默认只生成最终混音以降低网页内存。勾选“生成 heartbeat_sync 五文件兼容包”后会额外生成五个固定 24-bit 合同文件、歌曲轨和 ZIP。每次修改 BPM、第一拍、心跳角色或区域设置后，应重新渲染并试听。

## 输出文件

| 文件 | 含义 |
|---|---|
| `final_mix.mp3/.flac/.wav` | 按网页所选格式导出的最终歌曲与心跳混音；默认 MP3 |
| `heartbeat_aligned.wav` | 可选：已对齐并经过最终母带增益的独立心跳轨 |
| `song_processed.wav` | 可选：经过区域增益、低频闪避和母带增益的歌曲轨 |
| `debug_click_mix.wav` | 可选：最终混音加节拍点击声，用于检查 BPM 和第一拍 |
| `mix_report.json` | 输入分析、心跳质量、实际增益、区域设置、母带响度和峰值 |
| `heartbeat_timeline.csv` | 每个心跳事件的时间、密度模式、适配方式和所属区域 |
| `region_edits.json` | 可复查的区域编辑参数 |
| `heartbeat_music_project.zip` | 上述全部工程文件 |

`heartbeat_sync` 五文件合同如下，名称固定且 WAV 均为 PCM 24-bit：

| 文件 | 含义 |
|---|---|
| `preview_mix.wav` | 合同版最终歌曲与心跳混音 |
| `heartbeat_aligned.wav` | 对齐后的独立心跳轨 |
| `debug_click_mix.wav` | 歌曲与节拍/重拍点击检查轨 |
| `heartbeat_detection_mix.wav` | 预处理心跳与 S1/S2 点击检查轨 |
| `analysis_report.json` | `run` 到 `alignment` 的完整迁移报告 |

心跳预处理阶段优先使用 `cycle_pool` 中最多 16 个真实周期进入混音；若旧结果没有素材池才回退到 `cleanest_heartbeat_loop.wav`。不会使用仅供手机试听的增响版本，避免重复增益和软削波。

## 处理架构

```text
heartbeat.wav
  -> process_audio_bytes()
  -> hum notch + phase-aware attenuation-only denoising
  -> quality gate
  -> quality-ranked real cycle pool -> per-cycle S1 onset anchors
                                      -> preserve S1/S2 active audio
                                      -> local adaptive pulse schedule
song.wav / song.mp3 -> beat/downbeat/energy grid -> region automation
                                      -> loudness balance
                                      -> low-band ducking
                                      -> master gain + peak protection
                                      -> final_mix.flac/.mp3/.wav
```

自动调度优先选择真实歌曲节拍，并根据心跳自然周期在局部改变密度。只有节拍模型出现长缺口时才插入报告中明确标记的 guide pulse，并在下一个真实节拍重新锁定。歌曲本身不会被整体 time-stretch；心跳周期只允许受限拉伸，S1 onset 在拉伸后会重新检测并落在目标时间。

## 隐私和部署

Streamlit 上传文件会进入服务器内存，不是只存在于用户浏览器。分析后只把小型元数据和心跳精选周期放入会话状态。最终 WAV 写入系统临时任务目录，播放器直接读取路径，下载按钮点击时才读取文件。用户清除任务时会删除目录，超过 6 小时的遗留任务也会被回收；这些目录不是持久化存储。

Streamlit Community Cloud 部署：

1. 选择本仓库和目标分支；
2. Entrypoint 选择根目录 `app.py`；
3. Python 选择 3.11；
4. 依赖由 `environment.yml` 安装；
5. `.streamlit/config.toml` 将单文件上传上限设为 120 MB，应用内部将歌曲限制为 100 MB。

当前版本限制歌曲为 5 分钟，并以全局单任务锁保护 Community Cloud。处理报告会记录各阶段 RSS 内存。若未来需要真正的多用户并发，应保持此网页入口不变，将 `music_processor/core.py` 放到独立异步 worker 中执行，并把输入/输出放入对象存储。

### 连接官方 heartbeat_sync Windows 服务

当前仓库不导入、不修改 `heartbeat_sync` 内部模块，而是严格使用其公开 CLI：

```powershell
$env:HEARTBEAT_SYNC_REPO = "D:\path\to\heartbeat_sync"
python -m streamlit run app.py
```

目标仓库必须已按其迁移文档建立仓库内 `.venv`，且存在 `.venv\Scripts\python.exe`。网页使用参数数组调用 `python.exe -m heartbeat_sync`，每个任务使用独立输入和输出目录，并验证 stdout JSON、输出路径边界和五文件完整性。Streamlit Community Cloud 是 Linux 环境，不能直接运行该 Windows/Beat This 环境，因此线上默认使用合同兼容的 librosa/DSP 引擎。

逐条迁移状态见 [`SYNC_MIGRATION_STATUS.md`](SYNC_MIGRATION_STATUS.md)。

## 测试

```powershell
conda run --no-capture-output -n heartbeat python -m unittest discover -s tests -v
```

测试覆盖：

- 原心跳预处理、节律保持、局部污染门控和输出电平；
- 手动 BPM 与第一拍网格；
- 小节重拍、局部变速、自适应真实节拍选择和 guide pulse；
- S1 非零预滚锚点与排程误差；
- 局部双速和静音区域；
- 低内存磁盘输出契约；
- 心跳到歌曲的端到端渲染；
- 最终 WAV、独立轨、时间线、报告和 ZIP；
- 最终峰值上限。
- Unicode/空格路径、CLI 非 shell 调用、输出目录越界和失败信息；
- 混合歌曲裁剪、片头片尾时轴和五个 PCM 24-bit 合同文件。

## 已知边界

- 当前云端安全模式使用 librosa 加低频重音推断 downbeat；弱鼓点歌曲仍应使用手动小节重拍。Beat This 等重模型应部署到独立 worker，不能直接加入 Community Cloud 主进程。
- 自由速度、无明显打击乐、复杂切分或中途拍号变化的歌曲可能需要分段手动设置。
- 单通道心跳中与 S1/S2 完全重叠的摩擦或机械噪声无法可靠分离；此时应重新录制。
- 目标 LUFS 和峰值上限冲突时，峰值安全优先，最终响度可能低于目标。
