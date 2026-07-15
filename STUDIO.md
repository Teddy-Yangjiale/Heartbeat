# Heartbeat Studio 多轨网页工作台

## 定位

`Heartbeat Studio` 是运行在本机浏览器中的音频工作台。它不试图复制 FL Studio 的所有编曲功能，而是围绕本项目的核心流程提供类似 DAW 的交互：

- transport：播放、暂停、停止、回到开头、时间和 bar/beat 显示；
- arrangement：多轨波形、beat/bar 网格、可缩放时间轴和播放头；
- track controls：每轨 mute、solo、volume 和 pan；
- mixer：对应轨道的电平与 fader；
- heartbeat alignment：拖动心跳轨相位、设置 BPM、拍号、snap 和心跳增益；
- render revisions：每次正式渲染保留一个 revision，并导出 WAV/MP3；
- analysis QA：beat confidence、constant-grid P95 error、心跳 BPM、loop 时长和人工复核警告。

浏览器端用 Web Audio API 做低延迟试听和 mixer 控制。正式导出仍由 Python 音频后端执行 phase-vocoder time stretch、相位对齐、循环和防削波处理，因此拖动后的最终文件不会使用会改变音高的浏览器 `playbackRate` 方案。

## 启动

安装或更新环境：

```powershell
setup_env.bat
```

启动：

```powershell
run_studio.bat
```

访问：

```text
http://127.0.0.1:8503
```

## 工作流

1. 在左侧 Project 面板选择心跳 WAV/MP3 和目标歌曲 WAV/MP3。
2. 保持 `AUTO BPM`，第一次让系统估计 BPM；如果已从 DAW 确认 BPM，可关闭 AUTO 后填写精确值。
3. 选择拍号和 loop intervals；拍号只定义歌曲网格，不改变歌曲信号。
4. 点击 `Analyze & build session`。
5. 检查左侧 Analysis：
   - confidence 低于 45% 时应人工检查；
   - beat interval CV 高于 8% 时可能存在变速或跟踪错误；
   - grid P95 error 超过一个 beat 的 15% 时不应直接接受自动网格。
6. 使用 transport 试听；用轨道 M/S、音量、pan 检查各层。
7. 横向拖动 Heartbeat layer 调整 phase；Snap 可设为整拍、1/2 拍、1/4 拍或关闭。
8. 修改 BPM、phase 或心跳 render gain 后，点击 `Render current arrangement`。
9. 从右侧 Export 下载最新 revision 的 WAV 或 MP3。

## 为什么参数修改后需要 Render

轨道 volume、pan、mute、solo 可以在浏览器里即时试听。BPM/phase 会改变心跳素材的实际采样位置或时长，需要后端重新运行高质量 time stretch；页面会先显示视觉位移，再明确提示 Render。这样可以同时得到交互性和可交付音质。

## 稳定性策略

歌曲分析不仅输出单一 BPM，还输出：

- 自动检测的原始 beat times；
- BeatNet 检测的 beat/downbeat/meter 和动态 tempo map；模型不可用时使用 librosa 网格；
- beat strength/coverage confidence；
- detected IBI coefficient of variation；
- detected beats 对 constant-tempo grid 的 median / P95 error；
- `requires_manual_review` 和具体 warnings。

混音阶段保证：

- 歌曲不做 time stretch；
- 心跳 loop 按实际 beat grid 分段拉伸，每段边界落在对应 beat 上，避免长曲累计漂移；
- first beat 之前心跳层为静音；
- 单独导出 `heartbeat_layer.wav`；
- 记录 pre-limiter peak 和最终 mix gain reduction；
- 所有 render revision 保存在 `outputs/studio/<job>/remix/revision_NNN/`。

## 人声/伴奏和旋律

Studio 会分别检测 BeatNet、Demucs 和 Basic Pitch。模型放在独立 Python 3.9 环境，避免 BeatNet 的旧 NumPy 约束影响 Web API。安装方式：

```powershell
setup_music_env.bat
```

安装后重启 Studio，即可选择：

- vocals / accompaniment two-stem separation；
- 从 vocals 用 Basic Pitch 提取 note-events CSV 和 MIDI；失败时回退 pYIN。

## 当前边界

- 这是本地音频工作台，不包含云端账号、多人项目和在线素材存储。
- BeatNet 会自动检测 meter/downbeat，但弱节奏、切分和自由速度歌曲仍应试听复核或人工覆盖。
- BeatNet 网格使用分段 tempo-map 渲染；人工 BPM/phase 覆盖使用 constant grid。
- 中文歌词翻译、音节重填词和歌声合成需要独立 Stage 3；详见 `STAGE2_REMIX.md` 的依赖说明。
