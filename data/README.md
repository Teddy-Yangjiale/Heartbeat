# Dataset plan

本目录只保存数据来源、许可证和复现实验说明。实际音频放在 `data/external/`，合成测试对放在 `data/generated/`，两者都不会提交到 Git。

## 优先级

### 1. CirCor DigiScope v1.0.1

- 用途：主要真实噪声验证集。
- 内容：WAV、逐文件 S1/S2/收缩期/舒张期 TSV 标注、受试者信息。
- 价值：官方说明明确包含听诊器摩擦、说话、哭泣和笑声等真实污染。
- 规模：约 559 MB 解压后。
- 许可证：Open Data Commons Attribution 1.0。
- 来源：https://physionet.org/content/circor-heart-sound/1.0.1/

Starter 下载按受试者选择，保留同一受试者的多个听诊位置，便于严格按受试者划分验证集。

### 2. PhysioNet/CinC Challenge 2016 v1.0.0

- 用途：跨设备、跨环境、短录音和质量拒绝测试。
- 内容：3126 条训练 PCG，约 5 秒到 120 秒，统一 2000 Hz WAV；含更新后的信号质量标注。
- 噪声：说话、听诊器移动、呼吸和肠鸣音。
- 许可证：Open Data Commons Attribution 1.0。
- 来源：https://physionet.org/content/challenge-2016/1.0.0/

Starter 只下载约 17.5 MB 的官方 validation 包；确认流程后再决定是否下载完整训练集。

### 3. EPHNOGRAM v1.0.0

- 用途：运动、接触松动、模拟前端饱和等压力测试。
- 内容：69 组同步 ECG/PCG，包含 30 秒和 30 分钟运动记录，并标注部分低质量记录。
- 注意：总下载约 6.4 GB，而且主要为 WFDB/MAT，不作为第一批数据。
- 许可证：Open Data Commons Attribution 1.0。
- 来源：https://physionet.org/content/ephnogram/1.0.0/

### 4. DEMAND

- 用途：与干净 PCG 合成具有参考答案的空间环境噪声测试对。
- 内容：15 种环境、16 通道 WAV，16 kHz/48 kHz。
- 注意：它不包含听诊器接触摩擦；摩擦声仍需用我们自己的设备采集。
- 许可证：CC BY-SA 3.0。
- 来源：https://zenodo.org/records/1227121

## 下载 starter 数据

```powershell
conda run --no-capture-output -n heartbeat python scripts\download_datasets.py starter
```

默认下载：

- CirCor 前 5 个受试者的全部听诊位置及其 TSV 标注；
- PhysioNet 2016 validation.zip；
- 对应许可证和元数据。

自定义 CirCor 受试者数量：

```powershell
conda run --no-capture-output -n heartbeat python scripts\download_datasets.py circor --circor-subjects 50
```

只下载 PhysioNet 2016 validation：

```powershell
conda run --no-capture-output -n heartbeat python scripts\download_datasets.py physionet2016-validation
```

## 当前本地验证集

- CirCor：10 位受试者、38 条 WAV，并保留逐文件 S1/S2 TSV 标注。
- PhysioNet 2016 validation：301 条 WAV。
- 合计：339 条公开心音；音频和评估输出均受 `.gitignore` 保护，不进入仓库。

以每条最多 15 秒的中心窗口运行评估：

```powershell
conda run --no-capture-output -n heartbeat python scripts\evaluate_dataset.py `
  data\external\circor-heart-sound-1.0.1 `
  --output data\generated\circor_evaluation.json

conda run --no-capture-output -n heartbeat python scripts\evaluate_dataset.py `
  data\external\physionet-challenge-2016-1.0.0 `
  --limit 100 `
  --output data\generated\physionet2016_evaluation.json
```

评估报告同时写出 JSON 和 CSV，并报告中位数之外的最小值、P05/P25/P50/P75/P95、重录比例、最佳循环回退比例、循环心音/间隙对比度、播放版 RMS/峰值，以及 CirCor 官方 S1/S2 标注区域的保留相关性。

## 我们还需要自己采集的数据

公开环境噪声不能准确模拟传感器与皮肤、衣物、线缆之间的机械耦合。建议用最终设备额外录制：

- 传感器轻微移动；
- 手指触碰或调整压力；
- 线缆摩擦衣物；
- 外壳轻碰；
- 安静静置底噪。

每种至少 20 段、每段 3–10 秒，保留原始 WAV。纯噪声片段用于与高质量 PCG 按已知 SNR 合成测试对；带真实心搏的故意摩擦片段只用于盲听和质量拒绝测试。

## 划分规则

- 必须按受试者划分，不能把同一人的不同听诊位置分到不同集合。
- 算法不使用测试音频训练参数。
- 公开干净 PCG 与噪声合成时记录原始文件、裁剪区间、随机种子和目标 SNR。
- 正常和异常心音都要保留；降噪器不能把杂音或病理性杂音当成环境噪声删除。
