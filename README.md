# RSI Framework

可组合的机器人自改进（RSI）框架。环境、机器人、VLA、决策者（LLM）、Memory、训练都是插件，按名字组合运行。基于 PyRUA + EWAM（https://github.com/changyicheng1234/pyrua-ewam-agent），原库完整保留在 [`pyrua-ewam-agent/`](pyrua-ewam-agent/)。

```
观察 → 决策（LLM / 无） → 动作（VLA / 工具） → 执行 → 记经验（Memory） → 更新模型（VLA 训练） → 发布
```

## 快速开始

```bash
pip install numpy pillow opencv-python httpx

python -m rsi_framework run --env robodojo --vla ewam                    # 原窗口实验
python -m rsi_framework run --env robodojo --vla ewam --controller api   # Agent + VLA
python -m rsi_framework run --env robodojo --vla none --controller api   # 只用 LLM（CaP）
python -m rsi_framework run --env robodojo --vla ewam --controller none  # 只用 VLA（原 EWAM 基线）
```

常用参数：`--controller window|api|none` · `--protocol` · `--memory off|frozen|learn` · `--task` `--layout` `--out`

## 运行原 GitHub 库

```bash
./run_original.sh             # 窗口实验 full
./run_original.sh sparse      # 窗口实验 sparse
./run_original.sh baseline    # 纯 EWAM 基线
```

原库原样放在 `pyrua-ewam-agent/`（含它的 README、LICENSE、NOTICE），未做任何改动；也可以进入该目录按它自己的 README 运行。

## 配置

| 位置 | 内容 |
| --- | --- |
| `configs/envs/<env>.json` | 默认任务、种子、GPU |
| `configs/vlas/<vla>.json` | VLA 服务 |
| `configs/controllers/<controller>.json` | LLM 接口（`LLM_BASE_URL` 等） |
| `configs/profiles/` | 固定组合（Memory、训练），用 `--profile` 运行 |

## 接入自己的环境 / VLA

```python
from rsi_framework.core.registry import register
from rsi_framework.environments.base import NativeEnvAdapter
from rsi_framework.vla_backends.base import InProcessVLA

@register('environment', 'my_env')
class MyEnv(NativeEnvAdapter):
    robot = 'my_robot'                                   # 已注册的 RobotSpec
    defaults = {'task': 'reach', 'eval_seed': 0, 'layout': 0}
    def make_native(self, run):                          # get_obs / get_status / step / episode_over / conclude / result
        return MySimulator(run.case)

@register('vla', 'my_vla')
class MyVLA(InProcessVLA):
    contract = {'cameras': ['front'], 'state_keys': ['arm'], 'action_keys': {'arm': 7}, 'stride': 1}
    def predict(self, images, state, language):
        return actions                                   # [{'arm': ...}, ...]
```

```bash
python -m rsi_framework run --env my_env --vla my_vla --controller api --plugin my_plugin
```

完整示例：`tests/plugins/toy.py`

## 目录

```
rsi-framework/
├── run_original.sh                                        原库一键运行
├── rsi_framework/
│   ├── environments/ robots/ vla_backends/ tools/        环境 · 机器人 · VLA · 工具
│   ├── llm_backends/ policies/ protocols/                决策来源 · 决策循环 · 执行规则
│   ├── memory/ data/ trainers/ evaluation/ releases/     经验 · 数据 · 训练 · 评估 · 发布
│   ├── compat/                                           原库兼容
│   └── core/                                             注册 · 配置 · 装配 · 运行器
├── configs/                                              默认参数与组合
├── pyrua-ewam-agent/                                     原库（原样保留）
└── tests/
```

## 其他命令

```bash
python -m rsi_framework presets                           # 原库组合
python -m rsi_framework blocks                            # 已注册插件
python -m rsi_framework round --plan plan.json --out runs/round1   # 一轮改进
python -m unittest discover -s tests -t .                 # 测试
```
