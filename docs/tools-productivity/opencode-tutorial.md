# opencode 使用教程

::: tip 使用原则
和快捷键速查一样，这篇当外部记忆用，不用背诵。先记住三条主线：`/help` 查所有命令、`ctrl+p` 打开命令面板可搜索任意动作、`esc` 随时打断。其余用到时再回来查表。
:::

opencode 是一个跑在终端里的 AI 编程助手。本文不涉及安装与登录，只覆盖日常开发中反复用到的那部分：会话管理、输入技巧、Plan/Build 工作流、项目规则与自动化。

## 1. 三种打开姿势

```bash
opencode                     # 当前目录启动 TUI
opencode /path/to/project    # 指定项目启动
opencode -c                  # 继续上一次会话（最常用的开工命令）
opencode run "总结这个仓库"   # 非交互一次性提问，适合脚本和快速问答
```

一次性任务不必进 TUI。`opencode run` 直接收答案，还可以指定模型和 agent：

```bash
opencode run -m anthropic/claude-sonnet-4-5 --agent plan "分析这个函数的复杂度"
```

## 2. 会话是第一公民

一个任务一个会话。任务做偏了就开新会话，不要在旧会话里硬掰方向——上下文越干净，模型表现越好。

| 操作 | 命令 | 快捷键 |
| --- | --- | --- |
| 新会话 | `/new`（别名 `/clear`） | `ctrl+x n` |
| 切换/恢复会话 | `/sessions`（别名 `/resume`） | `ctrl+x l` |
| 重命名会话 | — | `ctrl+r` |
| 中断当前生成 | — | `esc` |
| 压缩上下文 | `/compact`（别名 `/summarize`） | `ctrl+x c` |
| 导出会话为 Markdown | `/export` | `ctrl+x x` |

几点心得：

- 会话变长、响应变慢或变贵时用 `/compact`，它把历史压成摘要后继续；注意压缩会丢失细节，关键结论先让它写进文件或 `AGENTS.md`。
- 调试过程有存档价值时用 `/export`，导出的 Markdown 可以直接丢进笔记仓库。
- 子代理会产生子会话，`ctrl+x ↓` 进入子会话，`←` / `→` 在子会话间轮换，`↑` 回到主会话。

## 3. 撤销与重做：大胆让模型动手

```text
/undo   # 撤销最近一条消息：连带回滚它产生的所有文件改动
/redo   # 反悔刚才的撤销
```

撤销基于 Git 实现，所以项目**必须是 Git 仓库**。这个机制决定了使用心态：让模型大胆改，不满意就 `/undo`，比反复澄清提示词更快。

## 4. 输入框里的四个技巧

| 语法 | 作用 | 示例 |
| --- | --- | --- |
| `@文件` | 模糊搜索并注入文件内容 | `鉴权逻辑在哪 @packages/functions/src/api/index.ts` |
| `!命令` | 运行 shell，把输出注入对话 | `!git diff --stat` |
| `@子代理` | 手动召唤子代理 | `@explore 找到所有用到 UserController 的地方` |
| `$ARGUMENTS` | 仅用于自定义命令的参数占位 | 见第 8 节 |

长提示词不要在输入框里憋：`/editor`（`ctrl+x e`）会打开 `EDITOR` 环境变量指定的编辑器来写。建议在 shell 配置里加上：

```bash
export EDITOR="code --wait"   # VS Code；vim/nano 等同理
```

输入框本身是 Readline 风格：`ctrl+a` / `ctrl+e` 跳行首行尾，`ctrl+k` 删到行尾，`ctrl+w` 删一个词，`shift+enter` 或 `ctrl+j` 换行。

## 5. Leader 键与高频快捷键

opencode 的大量快捷键都带 leader 前缀（默认 `ctrl+x`），先按 leader 再按第二键，等待时间默认 2000 ms。例如「新会话」是 `ctrl+x` 然后 `n`。

| 动作 | 按键 |
| --- | --- |
| 命令面板（可搜索一切） | `ctrl+p` |
| 切换 Build / Plan 模式 | `tab` / `shift+tab` |
| 打开模型列表 | `ctrl+x m` |
| 打开主题列表 | `ctrl+x t` |
| 切换模型推理档位（thinking 强度） | `ctrl+t` |
| 查看帮助 | `/help` |

觉得不顺手就改：快捷键在 `tui.json` 的 `keybinds` 里覆盖，只需写想改的那几条。

```json
{
  "$schema": "https://opencode.ai/tui.json",
  "keybinds": { "leader": "ctrl+s" }
}
```

## 6. 先 Plan 后 Build：最值得养成的习惯

opencode 内置两个主 agent，用 `tab` 随时切换：

| Agent | 定位 | 默认权限 |
| --- | --- | --- |
| **Build** | 干活的，全工具开放 | 文件编辑、命令全部可用 |
| **Plan** | 想事的，只读为主 | 编辑和命令都会先征求同意 |

推荐的节奏：

1. 新需求先进 **Plan**，让它读代码、给方案，不产生任何改动；
2. 方案不满意就地反驳，反正它改不了文件；
3. 确认后 `tab` 切到 **Build**，一句「按刚才的方案执行」。

子代理按需召唤：`@explore` 快速只读搜代码，`@general` 跑多步骤研究任务，`@scout` 查外部依赖的文档与源码。搜索类问题丢给子代理，能避免主会话上下文被搜索结果灌满。

## 7. 让它更懂你的项目：AGENTS.md

在 TUI 里跑一次：

```text
/init
```

它会扫描仓库，生成一份项目规则文件 `AGENTS.md`（构建/测试命令、目录结构、项目约定等），之后每个会话自动携带。已有 `AGENTS.md` 时 `/init` 是就地改进而非覆盖。

两层配置：

- **项目级**：仓库根目录的 `AGENTS.md`，提交到 Git，团队共享；
- **全局级**：`~/.config/opencode/AGENTS.md`，放个人偏好，比如「回答用中文」「提交信息用英文」。

规则文件会随对话持续生效，是最省力的「调教」手段：凡是重复纠正过模型两次以上的事，就该写进去。

## 8. 自定义命令：把重复动作固化

把常用提示词做成 `.opencode/commands/*.md`，文件名即命令名。项目级放 `.opencode/commands/`，全局放 `~/.config/opencode/commands/`。

`.opencode/commands/review.md`：

```markdown
---
description: 按团队规范审查未提交的改动
agent: plan
---
以下是当前未提交的改动：

!`git diff`

按安全性、边界条件、可维护性三个维度审查，只报告确定的问题，给出修改建议但不要改代码。
```

之后在会话里输入 `/review` 即可。两个常用占位符：

- `$ARGUMENTS` 接收参数：`/review src/api.ts`；
- `` !`命令` `` 注入 shell 输出，`` @路径 `` 注入文件内容。

再配一个测试命令，日常就够用了：

```markdown
---
description: 跑测试并修复失败项
agent: build
---
运行完整测试套件，聚焦失败的用例，定位原因并修复。修改后重新运行验证。
```

## 9. 权限：把安全感配置出来

敏感操作可以在 `opencode.json` 里设为 `ask`（每次确认）、`allow`（放行）或 `deny`（禁止），bash 支持通配符：

```json
{
  "$schema": "https://opencode.ai/config.json",
  "permission": {
    "edit": "allow",
    "bash": {
      "*": "ask",
      "git status*": "allow",
      "git push": "ask",
      "rm *": "deny"
    }
  }
}
```

不想逐条配置的最简方案：平时用 **Plan** 模式（天然全 ask），只在执行时切 **Build**。

## 10. 命令行自动化与随手查

```bash
opencode models                  # 查可用模型的准确 ID，配置 -m 时用
opencode models --refresh        # 刷新模型列表缓存
opencode session list            # 列出历史会话
opencode stats                   # 查看 token 用量与花费统计
opencode pr 42                   # 拉取并检出 GitHub PR 分支，然后启动 opencode
opencode upgrade                 # 自更新
```

`opencode run` 与 `-c` 组合可以在脚本里做持续对话，`--format json` 可以拿到结构化输出，适合接进自己的流水线。

## 11. 日常节奏小结

1. **开工**：`opencode -c` 找回昨天的会话，或 `/sessions` 挑一个；
2. **新需求**：`tab` 切 Plan，先要方案再动手；
3. **执行**：`tab` 切 Build，放它跑，`esc` 随时叫停；
4. **不满意**：`/undo` 回滚，换个说法重来；
5. **提交前**：`/review` 让 Plan 模式把关；
6. **收工**：`/export` 存档有价值的会话，提交 `AGENTS.md` 与自定义命令，让下次更快。
