# PyRUA + EWAM 实验控制扩展

这是可审查的源码导出仓库，包含 RGB 机器人控制、冻结 EWAM 策略、SAM3 证据记录，以及参考 RoboDoctor 思路设计的监督协议。它不包含实验结果、对话、图片、API 请求记录、模型权重或运行环境。当前机器人实验继续使用原工作目录；本导出不会修改或重启它们。

## 目录与版本

- `projects/ewam_window_parallel24/`：当前窗口 AI 直接编写 Python 单元，通过文件队列控制机器人；没有 OpenAI API 调用或密钥读取。
- `projects/ewam_comparison/`：本地 EWAM 服务、三相机 RGB 会话、基础控制及运行资源路径准备。
- `projects/ewam_doctor_integration/`、`ewam_doctor_validation72/`、`ewam_doctor_efficiency48/`：历史 API 控制器、协议、分析与离线测试的独立版本。目录数字是实验设计规模，不代表本仓库提供了这些实验结果。
- `configs/*.example.json`：脱敏配置模板，已移除历史结果引用和冻结的任务经验；需要自行准备配置与独立 warmup。
- `provenance/source_manifest.json`：每个原文件和导出文件的 SHA256、路径与修改说明。
- `provenance/original_sources/<SHA256>.py`：精确原始源码，按内容去重。55 个 cohort 文件对应 40 个原始源码对象。它们仅用于版本审查，不作为入口运行。可执行目录只增加项目路径可配置性，未合并不同实验的控制语义。

## 环境边界与上游

本仓库**不能通过一次 pip install 获得完整机器人系统**。需要 Linux/NVIDIA GPU、匹配的驱动/CUDA、可用的 Isaac Sim/Isaac Lab 运行时，以及单独获取的机器人资产、EWAM、Wan/Qwen、SAM3 等依赖和权重。下载权限、权重许可证及 NVIDIA 条款由使用者分别满足。本仓库不重新分发这些内容。

原程序使用以下工作区布局（`PROJECTS_ROOT` 指向 projects 层）：

```text
$PROJECTS_ROOT/
  PyRUA-Lean/src/pyrualean/       # 单独获取，上游源码
  RoboDojo/XPolicyLab/           # 单独获取，含 EWAM adapter
  RoboDojo-resources/Assets/     # 单独获取
  RoboDojo-resources/backbones/  # 单独获取，含 sam3/sam3.pt
  ewam_comparison/
    .venv-ewam/bin/python       # 自行创建的策略/SAM/PyRUA Python 环境
    .venv-sim/bin/              # 自行配置的仿真环境
    runtime_assets/Assets/      # prepare_runtime_assets.py 生成
  ewam_window_parallel24/
  ewam_doctor_integration/
  ewam_doctor_validation72/
  ewam_doctor_efficiency48/
```

可以在全新工作区复制 `projects/` 中的目录，再放入上游检出与本地资源；不要覆盖正在运行的实验。也可使用此仓库的 `projects/` 作为工作区，按需链接外部上游资源与环境。

```bash
export PROJECTS_ROOT=/your/new/workspace/projects
export PYTHONPATH="$PROJECTS_ROOT/PyRUA-Lean/src:$PROJECTS_ROOT/ewam_comparison"
export EWAM_PYTHON="$PROJECTS_ROOT/ewam_comparison/.venv-ewam/bin/python"
```

`requirements-extension.txt` 仅列扩展直接使用的 Python 包；两个 `*_constraints.txt` 是原环境约束参考，不是统一安装锁。策略、SAM 和仿真环境有不同的 torch/CUDA 约束，需要分别配置。首次仿真前在新工作区运行 `prepare_runtime_assets.py`，并确认 Vulkan、缓存与仿真客户端路径符合当前机器。历史脚本的 GPU admission 阈值也需按实际硬件审查。

## 当前窗口控制：不使用 OpenAI API

先准备三个相机的 RGB/机器人状态通道，启动真实 EWAM 本地 Unix socket 服务。每个并行 lane 应使用独立服务目录；lane 内 EWAM/SAM/仿真可以按显存预算共用 GPU。下面只演示一个 lane，其他 lane 需独立 GPU 或进行资源准入，避免照搬并发数。

```bash
"$EWAM_PYTHON" "$PROJECTS_ROOT/ewam_comparison/ewam_service.py" \
  --root "$PROJECTS_ROOT/ewam_window_parallel24/services/A" --gpu 0
```

模板不会携带历史学习经验。渲染并审查配置后再运行；渲染器对未设置的占位变量报错。

```bash
python tools/render_config.py configs/ewam_window_parallel24.example.json /tmp/window-config.json
"$EWAM_PYTHON" "$PROJECTS_ROOT/ewam_window_parallel24/window_session.py" \
  --config /tmp/window-config.json --task classify_objects --eval-seed 0 \
  --layout 10 --out /your/private/runs/new-case --variant full --lane A
```

模板保留了历史 lane GPU 编号和服务相对位置作为示例，启动前应改成刚启动的本地服务路径。`--out` 必须是不存在的新目录。命令通过 `commands/*.json` 提交，响应写入 `responses/*.json`，Python 命名空间跨单元保留。

```bash
python "$PROJECTS_ROOT/ewam_window_parallel24/window_control.py" \
  --session /your/private/runs/new-case --note '根据当前 RGB 选择有意义的子目标' \
  --async python --file /tmp/cell.py
```

控制者必须实际查看 `latest.json` 指向的 RGB，再显式确认最新 observation index、当前 native action index 和所查看相机 PNG 的 SHA256；确认不会自动声称模型已看图。

```bash
python "$PROJECTS_ROOT/ewam_window_parallel24/window_control.py" \
  --session /your/private/runs/new-case --note '已查看当前相机图像' \
  ack_visual --observation-index 3 --native-action-index 0 \
  --camera-sha256 cam_head=REPLACE_WITH_EXACT_SHA256 \
  --evidence '描述图像中实际可见的物体、夹爪和障碍'
```

`full`：`robo.ewam_propose()` → 查看新 RGB 和机器人 FK 预览 → `ack_visual` → `robo.ewam_execute(prediction_id, reason, evidence, model_actions=...)`。

`sparse`：`robo.ewam_step(reason, evidence, model_actions=...)` 在宿主完成有限值/FK 检查后执行，不提供单独的窗口预览回合。它是不同监督方式，不应宣称与 full 具有完全相同的审核语义。

两组都要求先执行有意义的 EWAM 段再使用几何动作；每段最多 16 个模型动作/48 个原生动作；EWAM 之后必须查看并确认新 RGB，才能再次运动。共享官方原生预算不变，另有限制 100 个 Python 单元与 5400 秒控制时间。FK 只预测机器人运动，不能证明接触、抓取或物体结果。完成时使用 `finish`；官方客户端判定成功和原始得分，技术中断单独记录。

直接窗口模式的精确模型身份不可从程序验证，固定记录 `window_AI`、模型未验证、`api_requests=0`、token usage unavailable；不能据此声称与历史 API 模型相同，也不能拿它计算真实 API token 节省。

## 历史 API 模式

历史 `run_episode.py` 使用 `api_transport.py` 调用 Responses API，配置包含模型、推理强度、输出预算和密钥文件位置。这些代码是历史实验版本，不是当前窗口模式的依赖；切勿为了运行窗口控制而启动 API 驱动器。

如需单独审查/运行历史 API 模式，请自行设置 `API_KEY_FILE` 为本机私有密钥文件路径，再渲染模板。仓库没有真实密钥、密钥文件或历史请求。不要把密钥写进 JSON 或提交到 Git。历史 `run_study.py` / `run_validation.py` 还依赖独立 warmup、冻结 harness 和新生成声明；导出不含这些结果，所以不能直接声称复现既有完整队列或准确率。

## 离线检查

```bash
"$EWAM_PYTHON" "$PROJECTS_ROOT/ewam_window_parallel24/window_protocol_tests.py"
python tools/verify_export.py
```

协议测试使用合成 RGB/机器人状态和假 EWAM，不启动仿真或 API。它验证确认新鲜度、相机哈希、full 释放门控、EWAM-first、sparse FK 屏蔽、预算和文件队列。编译/离线测试通过不等于真实仿真在另一机器可直接启动。

## 归属与许可

参见 `THIRD_PARTY_NOTICES.md`、`NOTICE` 和 `licenses/`。PyRUA-Lean 及其 RPent 派生接口遵守 Apache-2.0；保留了上游 NOTICE。RoboDojo 根目录为 MIT，XPolicyLab 为 Apache-2.0；这些上游源码未被整体 vendoring。EWAM、模型资产和仿真软件各有自己的条款。

本扩展受到 [RoboDoctor](https://github.com/THUSI-Lab/RoboDoctor) 中策略提案、连续前缀、诊断记忆与策略/直接动作路由等概念启发；它不是对完整 RoboDoctor 方法的实现或复现实验，也没有复制其全部动作系统。几何动作仍调用现有 PyRUA 实现。

此仓库仅发布代码、配置模板和文档；实验数据与模型资产保存在各自的私有工作区。发布前应审查许可、配置及扫描报告；私有发布不会改变第三方依赖的许可与分发条件。
