# 上游来源与许可

| 组件 | 本机核查的许可 | 使用方式 |
| --- | --- | --- |
| [PyRUA-Lean](https://github.com/DAGroup-PKU/PyRUA-Lean) | Apache-2.0；含 RPent 归属 NOTICE | 外部依赖；扩展调用现有 RGB robot/primitives；核查 commit `20dbb21e823030077ff4ddd4463cb91e36277515` |
| [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) | 根目录 MIT | 外部仿真/评测依赖；本地来源记录 commit `cb2a37d` |
| [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab) | Apache-2.0 | 外部 EWAM policy adapter；本地来源记录 commit `53c5da5` |
| [RoboDoctor](https://github.com/THUSI-Lab/RoboDoctor) | 仅概念引用，未 vendoring 源码 | 提案/前缀/诊断记忆/路由概念启发，不声称完整方法复现 |
| EWAM、Wan、Qwen、SAM3 权重 | 单独获取并遵守各自模型条款 | 不包含权重、tokenizer 文件或下载缓存 |
| NVIDIA Isaac Sim / Isaac Lab | 独立软件/资产条款 | 使用者自行安装；本仓库不分发运行时 |

`licenses/` 保留所读取的 PyRUA-Lean Apache-2.0、NOTICE、RoboDojo MIT 和 XPolicyLab Apache-2.0 原文。上游代码未整体复制；这些文本用于保留派生接口归属与说明运行依赖，不会给模型权重或仿真资产重新授权。

`projects/` 的导出代码增加了 PROJECTS_ROOT 可配置路径与修改说明。`provenance/original_sources/` 保存导出前源码的精确字节版本，不作执行入口；其原有版权/归属说明保持不变。导出工具和文档不构成对上游商标或实验准确率的认可声明。
