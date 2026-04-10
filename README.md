# nanoteam

多 Agent 协作系统，把一个高层目标拆解成任务图，由专门的 Agent 角色分工执行。基于 Claude Code CLI，零外部依赖。

## 工作原理

```
你（甲方）──目标──▶ Lead Agent ──规划──▶ 任务图
                        │
                        ▼
                  Worker Agents ──执行──▶ 代码变更
                        │
                        ▼
                  Lead Agent ──Review──▶ 接受/驳回/重新规划
                        │
                        ▼
                  Checkpoint ──▶ 你检查进度、提新需求
```

- **Lead Agent**（默认 Opus）：规划任务、分配角色、审查结果、处理失败，不写代码
- **Worker Agent**（默认 Sonnet）：按 spec 执行具体编码任务，只看到自己需要的上下文
- **角色动态创建**：Lead 根据目标自行决定需要几个什么样的角色
- **文件系统通信**：所有状态存在 `.nanoteam/` 目录，无数据库、无消息队列

## 前置条件

- Python >= 3.12
- [uv](https://github.com/astral-sh/uv)
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) (`claude` 命令可用)

## 安装

```bash
cd nanoteam
uv pip install -e .
```

## 使用

### 启动一个新任务

```bash
nanoteam "构建一个基于 WebSocket 的实时聊天后端，支持多房间和消息持久化"
```

Lead 会先规划任务图，在 `plan` checkpoint 暂停让你确认。按回车继续，或输入反馈调整计划。

### 指定参数

```bash
nanoteam "目标描述" \
  --lead-model claude-opus-4-6 \
  --worker-model claude-opus-4-6 \
  --lead-effort high \
  --worker-effort medium \
  --max-budget 20.0 \
  --root /path/to/project \
  --checkpoint plan,phase,finish
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lead-model` | `claude-opus-4-6` | Lead 用的模型 |
| `--worker-model` | `claude-opus-4-6` | Worker 默认模型 |
| `--lead-effort` | `high` | Lead 的 effort 级别（low/medium/high/max） |
| `--worker-effort` | `medium` | Worker 的 effort 级别（low/medium/high/max） |
| `--max-budget` | `10.0` | 总预算上限（USD） |
| `--root` | 当前目录 | 项目根目录 |
| `--checkpoint` | `plan,finish` | 暂停时机：`plan`/`phase`/`finish`/`none` |

### 查看状态

```bash
nanoteam --status
```

输出任务图概览：每个任务的状态、角色、已花费金额。

### 中断恢复

Ctrl-C 中断后：

```bash
# 正常恢复（从断点继续）
nanoteam --resume

# 跳过卡住的任务
nanoteam --resume --skip task-003

# 跳过多个任务
nanoteam --resume --skip task-003 task-005

# 从指定任务开始（之前的自动标记完成）
nanoteam --resume --from task-004
```

Cost 会自动持久化，resume 后不会丢失。

### AI 诊断

跑挂了不知道哪里出问题？

```bash
nanoteam --diagnose
```

会读取 `log.jsonl`、任务图状态、最近的 prompt/response，喂给 Claude 分析问题原因并给出建议。

## `.nanoteam/` 目录结构

```
.nanoteam/
├── goal.md                          # 原始目标
├── task_graph.json                  # 任务图（单一真相源）
├── decisions.md                     # 架构决策记录
├── log.jsonl                        # 结构化事件日志
├── turns/                           # 全局调用记录（planning, feedback）
│   ├── 001-plan-prompt.md
│   └── 001-plan-response.md
├── tasks/
│   └── task-001/
│       ├── spec.md                  # 任务规格
│       ├── context.md               # 上下文（依赖输出等）
│       ├── result.md                # 执行结果
│       └── turns/                   # 该任务的调用记录
│           ├── 002-execute-prompt.md
│           ├── 002-execute-response.md
│           ├── 003-review-prompt.md
│           └── 003-review-response.md
└── team/
    └── roles/
        └── backend-dev.md           # 角色定义
```

## Checkpoint 交互

在 checkpoint 暂停时，你可以：

- **按回车**：继续执行
- **输入反馈**：Lead 会据此调整计划（增删改任务、加新角色等）

例如输入 `"ASR 用 whisper，不要用 Google Speech API"` → Lead 会修改相关任务的 spec。

## 许可

MIT
