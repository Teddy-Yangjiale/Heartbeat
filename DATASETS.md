# 测试数据集（心跳 + 歌曲 MV）

样本较少时，用下面这些**公开、版权安全**的数据来构建和测试管线。仓库里**不包含**这些媒体文件（体积/版权原因），下面给出获取方式与许可证。

## 1. 心跳（PCG 心音，与数字听诊器录音同类型）

**PhysioNet / CinC Challenge 2016 Heart Sound Database** — 3126 段单导联 PCG 心音，`.wav`，2000 Hz，5–120 s，采集自临床与非临床（含说话、听诊器移动、呼吸等真实噪声），与手机/数字听诊器 App 录出来的心跳同类型。

- 数据集主页：https://physionet.org/content/challenge-2016/1.0.0/
- 单文件直链示例（training-a）：
  ```bash
  curl -L -o a0001.wav https://physionet.org/files/challenge-2016/1.0.0/training-a/a0001.wav
  curl -L -o a0002.wav https://physionet.org/files/challenge-2016/1.0.0/training-a/a0002.wav
  ```
- 许可证：ODC-BY 1.0（开放数据，署名即可）。

本项目已用 `a0001 / a0002 / a0007 / a0011` 验证：Stage-1 均判为 `good`，估计心率 57–78 bpm，情绪映射在真实数据上给出 serene / melancholic 等不同结果。

## 2. 歌曲音乐视频（版权安全）

真实歌曲 MV 大多受版权保护；**你自己 demo 用最爱的英文歌 MV（例如 PPT 里的 "I Have A Dream"）即可**，但为了可复现的自动化测试，本项目用一支 **Creative Commons** 授权的真实 MV：

**Yoga Lin "Elephant Slide" (Official Music Video)** — 华语流行，含人声，3:54，720p，动画：Finger and Toe。

- 文件页：https://commons.wikimedia.org/wiki/File:Yoga_Lin_%22Elephant_Slide%22_(Official_Music_Video).webm
- 直链下载（Special:FilePath）：
  ```bash
  curl -L -o yogalin.webm 'https://commons.wikimedia.org/wiki/Special:FilePath/Yoga_Lin_"Elephant_Slide"_(Official_Music_Video).webm'
  ```
- **许可证：CC-BY-3.0**，允许再利用与改编，署名 **Finger and Toe**。
- 转成测试用小 mp4（取 60–90s、缩到 640×360）：
  ```bash
  ffmpeg -y -ss 60 -t 30 -i yogalin.webm -vf scale=640:360 -r 24 \
    -c:v libx264 -pix_fmt yuv420p -c:a aac -ac 2 real_mv.mp4
  ```

> 说明：该 MV 用于验证完整管线（提取歌曲音频 / 心跳对齐混音 / A 情绪调色 / B 心跳剪辑）。保留歌曲原声、心跳作鼓点铺底，不做人声分离或翻唱。demo 时换成你自己最爱的英文歌 MV 即可。

## 3. 更多可选来源

- 心音：PASCAL Heart Sounds Challenge;CirCor DigiScope 2022（PhysioNet）。
- CC 音乐/视频：Wikimedia Commons（`Category:Music videos`，逐个核对许可证）、Free Music Archive（CC-BY 歌曲）、Pixabay/Pexels（CC0 视频片段）。

引用：
[PhysioNet 2016](https://physionet.org/content/challenge-2016/1.0.0/) ·
[Yoga Lin – Elephant Slide (CC-BY-3.0, Finger and Toe)](https://commons.wikimedia.org/wiki/File:Yoga_Lin_%22Elephant_Slide%22_(Official_Music_Video).webm)
