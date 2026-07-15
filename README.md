# Heartbeat WAV Denoiser

这是一个只处理心跳降噪的预处理器。输入是一条约 15 秒的 WAV 心跳录音；输出是降噪 WAV、处理前后对照和质量报告。

## 范围

- 只接受 WAV，不处理视频、MP3、音乐或混音。
- 削弱稳定空间底噪、持续人声和非周期性的细微摩擦声。
- 尽量保留重复出现的 S1/S2 音色、节奏和强弱变化。
- 不使用单条录音训练模型，也不会把平均模板复制到输出中。
- 无法安全修复时返回 `needs_rerecording=true`，不生成伪造的心音细节。
- 不是医疗诊断工具。

## 为什么 15 秒仍然可用

在 40–140 BPM 范围内，15 秒通常包含约 10–35 个心动周期。算法不尝试用这 15 秒训练神经网络，而是使用多个周期的中位数和 MAD 稳健统计，识别稳定重复的心音以及偶发、不重复的摩擦能量。

## 降噪链路

1. WAV 解码、立体声转单声道、去直流偏置。
2. 20–250 Hz 四阶 Butterworth 带通。
3. STFT 软频谱门控，估计并削弱稳定噪声底。
4. 包络、自相关和模板相关性定位心动周期。
5. 将完整周期对齐到统一相位，计算周期中位数模板和 MAD。
6. 对偏离周期模板的正向瞬态应用软衰减；重复出现的 S1/S2 核心区域受到保护。
7. 在心搏间隙应用软门控；周期定位不可靠时自动限制门控深度，优先保住 S1/S2，最后统一导出峰值。
8. 对所有连续 4 周期候选按节律、心搏/间隙对比、噪声底和包络信噪比排序，导出真实波形中最干净的一段。

周期模板只用于生成 0–1 的衰减掩码，永远不会被复制或替换进输出波形。

## 安装

```powershell
cd D:\Heartbeat
conda env create -f environment.yml
conda activate heartbeat
```

已有环境可以更新：

```powershell
conda env update -f environment.yml --prune
```

## 网页界面

```powershell
conda run --no-capture-output -n heartbeat python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
```

也可以双击 `run_app.bat`。界面一次上传一条 WAV，并提供 Mild、Balanced、Strong 三个档位。默认先使用 Balanced；如果 S1/S2 被削薄，改用 Mild。

## 命令行

```powershell
conda run --no-capture-output -n heartbeat python scripts\process_files.py heartbeat.wav --out outputs\cli
```

## 输出

- `input_reference.wav`：归一化后的处理前试听参考。
- `spectral_filtered.wav`：仅完成带通和稳定噪声底处理的中间结果。
- `filtered_detection.wav`：加入周期一致性摩擦衰减后的中间结果。
- `cleaned.wav`：最终心跳降噪结果。
- `cleanest_heartbeat_loop.wav`：从真实输出中截取的最干净连续 4 心搏，不复制模板、不合成心搏。
- `cleanest_heartbeat_loop_loud.wav`：仅用于手机/播放的响度优化副本；保真分析仍应使用上一条文件。
- `cleanest_segment.json`、`cleanest_segment_candidates.csv`：入选区间、候选评分和响度处理参数。
- `recording_quality.json`：是否需要重录、原因和保真指标。
- `cycle_consistency.json`：参与模板估计的周期数、周期相关性和异常瞬态比例。
- `rhythm_preservation.json`、`postprocess_beat_times.csv`：处理后独立重检的心搏、与处理前的一对一匹配率、数量差异和时序误差。
- `diagnostic_plot.png`：处理前、中间结果、最终结果、心搏位置和节律图。
- `tempo_summary.json`、`beat_times.csv`、`ibi.csv`：用于解释门控位置的节律数据。

关键质量字段：

- `interbeat_noise_reduction_db`：心搏间隙处理前后的 RMS 变化。
- `heartbeat_preservation_correlation`：受保护心搏窗口内的波形相关性。
- `cycle_outlier_fraction`：非重复瞬态覆盖比例。
- `rhythm_preservation.is_preserved`：处理后重新检测的心搏数量与节奏是否通过保真门槛。
- `rhythm_preservation.matched_fraction`：处理前后按时间单调一对一匹配的心搏比例。
- `needs_rerecording`：是否不应信任自动输出。
- `reconstruction_policy=attenuation_only_no_template_replacement`：确认算法只衰减，不重建伪造心搏。

## 已知边界

- 与 S1/S2 完全重叠且频谱相似的摩擦声无法被单通道算法可靠分离；此时算法会优先保留心音。
- 每个周期相同位置都出现的机械噪声可能被误认为稳定心音。
- 削波、接触严重松动、有效心搏太少或周期一致性很低时应重新录制。
- 单独追求更低的背景 RMS 可能损伤心音，因此 Strong 不是默认最优选择。

## 验证

```powershell
conda run --no-capture-output -n heartbeat python -m unittest discover -s tests -v
```

测试覆盖持续人声衰减、非重复摩擦衰减、S1/S2 保真、处理前后输出契约、不可修复削波判定、BPM 稳健性和导出电平。

公开数据的选择、许可证、下载命令、按受试者划分规则和 15 秒评估方法见 `data/README.md`。本地 starter 当前包含 CirCor 38 条和 PhysioNet 2016 validation 301 条 WAV；原始音频不会提交到 Git。

当前 C 版在 CirCor 38 条 15 秒窗口上，官方 S1/S2 标注区域保真相关性最低 0.850、中位 0.971，间隙噪声中位下降 10.8 dB；37/38 条通过处理后节奏重检，唯一失败样本只检测到 1 个可靠心搏并已要求重录。在 PhysioNet 2016 前 100 条上，内部心音保真相关性最低 0.952、中位 0.9997，间隙噪声中位下降 12.1 dB，100/100 条通过节奏重检。两组共 138 条均无处理错误，响度版峰值保持在约 -1 dBFS。报告位于 `data/generated/*_rhythm_*.json`，该目录不提交到 Git。

客观指标只负责排除明显损伤，最终听感仍需要用同一输入做身份隐藏的盲听。`scripts/create_blind_ab_pack.py` 会验证三份 WAV 的采样率、声道数和帧数完全一致，随机分配 A/B，生成评分表，并把答案表放在试听目录之外。A/B 已被实际试听否决后，应把新候选作为 C 与原始输入、旧 v1 重新比较，而不是继续调高旧候选的增益。

评分完成后使用 `scripts/analyze_blind_ab_scores.py 评分表.csv 私有答案表.json` 揭盲。判定门槛要求 S1/S2 饱满度、起音、节奏和伪影指标均不能比 v1 退步，同时空间噪声或摩擦声至少一项改善；缺失或超出 1–5 范围的评分会被拒绝。
