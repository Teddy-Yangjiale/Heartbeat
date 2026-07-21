# heartbeat_sync 迁移实施矩阵

依据上游 [`SYNC_INTEGRATION.md`](https://github.com/aziefp/heartbeat_sync/blob/main/SYNC_INTEGRATION.md) 逐条实现。集成采用两层结构：Windows 有目标仓库环境时调用官方黑盒 CLI；公开 Linux Streamlit 网站运行合同兼容的本地 DSP 引擎。两层共享网页上传、心跳预处理、参数和结果展示，不复制或修改上游内部模块。

| 上游合同 | 官方 CLI 适配 | 网页原生实现 | 验证 |
|---|---|---|---|
| WAV 心跳；WAV/MP3 歌曲 | 原样写入任务输入目录 | 原生 WAV/MP3 解码 | 输入扩展名、声道、采样率测试 |
| `auto/beat-this/librosa` | 参数透传；上游负责 fallback | librosa，报告 Beat This 不可用警告 | 后端值校验与网页警告 |
| `auto/cpu/cuda` | 参数透传 | CPU | CLI 命令数组测试 |
| 混合静音裁剪 | 参数透传 | 强/保守 top-dB 双阈值与手动覆盖 | 自动/手动裁剪测试 |
| 预览 `max_song_seconds` | 只在预览传入 | 只在预览截取 | 整首不带 45 秒上限 |
| `beats_per_loop=4` | 参数透传 | 质量最高的连续周期窗口 | 默认值与五文件端到端测试 |
| 前后各 4 个脉冲 | 参数透传 | 平移歌曲时轴并安排 intro/outro | 片头片尾计数和最终时长测试 |
| bar/half/normal/double/auto | 参数透传 | 兼容全部模式，并保留网页的 downbeat/kick/backbeat/局部模式 | 调度测试 |
| 55–110 BPM 密度范围 | 参数透传 | 自适应网格约束 | 自适应节拍测试 |
| LUFS 与相对响度 | 参数透传 | pyloudnorm 自动平衡、低频 duck、母带保护 | 响度和峰值测试 |
| gap/stretch | 参数透传；网页 preserve 映射 gap | 额外提供 S1/S2 preserve | 周期适配测试 |
| stdout JSON | 解析并校验 | 生成同结构 `sync_summary` | 无效 JSON/错误退出测试 |
| 五个固定 24-bit 文件 | 校验上游实际产物 | 按需生成完全相同名称 | PCM_24 与字段测试 |
| 报告九个顶层区段 | 读取上游报告 | 生成九区段兼容报告 | 精确顶层集合测试 |
| Unicode、空格、并发不覆盖 | 参数列表与独立任务根 | `job_*` 独立目录 | Unicode/空格和路径越界测试 |

## 时间坐标

网页区域编辑继续使用“裁剪后歌曲时间”。最终输出坐标为：

```text
t_output = t_original - content_trim.used_start_seconds
           + arrangement.song_offset_seconds
```

`song_offset_seconds` 来自片头心跳排布。报告同时保存裁剪起止、处理源终点、歌曲偏移、平移后的 beat/downbeat 和全部心跳时间，因此可以把用户的原歌曲时间、编辑时间和最终文件时间互相转换。

## 部署边界

- Streamlit Community Cloud：原生兼容引擎，支持分段编辑和低资源保护。
- 配置 `HEARTBEAT_SYNC_REPO` 的 Windows 主机：网页自动显示官方 CLI，支持 Beat This 模型后端。
- 官方 CLI 是独立服务边界；本仓库不会依赖其私有 Python API。
