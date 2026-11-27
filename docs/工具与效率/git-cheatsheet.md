# Git 实用速查

Git 的命令很多，但日常真正用到的只有二十来个。本篇按"心智模型 → 每日循环 → 撤销 → 分支协作 → 急救"组织，每条命令给最常用形态和一句话说明，遇到问题时按场景查。

::: tip 一句话心智模型
工作区（你正在改的文件）→ 暂存区（`git add` 后待提交的快照）→ 本地仓库（`git commit` 后的历史）→ 远程仓库（`git push`）。
**提交是快照，不是差异**；分支只是一个指向提交的指针，所以建分支、切分支几乎零成本。
:::

## 1. 每日循环：改、看、提、推

| 动作 | 命令 | 说明 |
| --- | --- | --- |
| 看当前状态 | `git status` | 最该先敲的命令，它会告诉你下一步能做什么 |
| 看未暂存的改动 | `git diff` | 工作区 vs 暂存区 |
| 看已暂存的改动 | `git diff --staged` | 暂存区 vs 最后一次提交 |
| 暂存单个文件 | `git add path/to/file` | 比 `git add .` 更可控 |
| 暂存文件中的部分改动 | `git add -p` | 交互式挑选 hunk，提交粒度更干净 |
| 提交 | `git commit -m "feat: 说明"` | 提交信息写"做了什么"，不写"改了文件" |
| 修改上一条提交 | `git commit --amend` | 只用于尚未推送的提交 |
| 拉取并变基 | `git pull --rebase` | 避免无意义的 merge 提交 |
| 推送 | `git push` | 首次推送新分支：`git push -u origin 分支名` |

改动提交后才发现漏了文件：`git add 漏掉的文件 && git commit --amend --no-edit`，提交信息保持不变。

## 2. 查看历史与差异

| 目的 | 命令 |
| --- | --- |
| 简洁历史 | `git log --oneline` |
| 图形化看分支 | `git log --oneline --graph --all` |
| 最近 5 条带改动统计 | `git log -5 --stat` |
| 某文件的历史 | `git log --oneline -- path/to/file` |
| 某文件每次的具体改动 | `git log -p -- path/to/file` |
| 谁改的、哪次改的 | `git blame path/to/file` |
| 按内容搜索提交 | `git log -S"关键字" --oneline` |
| 查看某次提交 | `git show <commit>` |
| 查看某文件某版本 | `git show <commit>:path/to/file` |

`git log -S` 搜的是"哪个提交增删了这段字符串"（pickaxe），比 grep 提交信息可靠得多，定位"这行是谁删的"非常好用。

## 3. 撤销：按"改到哪一层"选命令

这是最常问也最容易搞错的部分。先问自己：**改动有没有 commit？commit 有没有 push？**

### 3.1 还没提交

| 目的 | 命令 |
| --- | --- |
| 丢弃工作区某文件的改动 | `git restore path/to/file` |
| 取消暂存（改动保留） | `git restore --staged path/to/file` |
| 取消暂存并丢弃改动 | `git restore --staged --worktree path/to/file` |
| 丢弃所有未提交改动（危险） | `git reset --hard` |
| 删除未跟踪文件（先预览） | `git clean -nd`，确认后 `git clean -fd` |

`git clean` 不会删除被 `.gitignore` 忽略的文件，只有加 `-x` 才会。删之前一定先跑 `-n`（dry run）。

### 3.2 已经提交，还没推送

| 目的 | 命令 | 效果 |
| --- | --- | --- |
| 撤销提交，改动回到暂存区 | `git reset --soft HEAD~1` | 最安全，适合"提交早了" |
| 撤销提交，改动回到工作区 | `git reset HEAD~1` | 默认 `--mixed` |
| 撤销提交并丢弃改动 | `git reset --hard HEAD~1` | 危险，改动不可恢复（除 reflog） |
| 只改提交信息 | `git commit --amend` | 生成新提交替换旧的 |

### 3.3 已经推送

不要 `reset` 后强推共享分支。正确做法是**追加一个反向提交**：

```bash
git revert <commit>       # 生成一个抵消该提交的新提交
git revert -n <commit>    # 只改工作区不提交，方便合并多个 revert
```

只有自己的分支、且确认没人在上面工作时，才用 `git push --force-with-lease`（比 `--force` 安全，别人推过会拒绝）。

### 3.4 reflog：误操作的后悔药

`reset --hard` 之后、分支误删之后，先别慌：

```bash
git reflog                # 本地 HEAD 的移动记录，保留约 90 天
git reset --hard HEAD@{3} # 回到 reflog 里的某个位置
```

reflog 只记录本地操作，且未推送的提交被 gc 后仍可能丢失，但绝大多数误操作都能靠它救回来。

## 4. 分支

| 目的 | 命令 |
| --- | --- |
| 查看本地 / 全部分支 | `git branch` / `git branch -a` |
| 新建并切换 | `git switch -c feature/x` |
| 切换分支 | `git switch main` |
| 切回上一个分支 | `git switch -` |
| 重命名当前分支 | `git branch -m 新名字` |
| 删除已合并分支 | `git branch -d feature/x` |
| 强制删除未合并分支 | `git branch -D feature/x` |
| 删除远程分支 | `git push origin --delete feature/x` |

`switch` 与 `checkout` 的区别：`switch` 只切分支、`restore` 只还原文件，语义更清晰；`checkout` 两个都能干，是老命令。

## 5. 合并、变基与拣选

| 目的 | 命令 | 适用场景 |
| --- | --- | --- |
| 合并分支 | `git merge feature/x` | 保留真实分叉历史，团队主线常用 |
| 快进合并 | `git merge --ff-only feature/x` | 要求线性，不能快进就报错 |
| 变基到主分支 | `git rebase main` | 把自己的提交"搬"到最新 main 之上，历史更线性 |
| 交互式整理提交 | `git rebase -i HEAD~3` | 合并、改写、排序、删除最近 3 个提交 |
| 拣选单个提交 | `git cherry-pick <commit>` | 把别的分支某个提交复制过来 |
| 中止合并 / 变基 | `git merge --abort` / `git rebase --abort` | 冲突处理不下去时的退路 |

::: warning 变基的铁律
**不要对已经推送、且别人可能基于它工作的分支做 rebase**。变基会重写提交哈希，别人的历史会对不上。个人分支可以随便 rebase，共享分支用 merge。
:::

冲突处理流程：`git status` 看冲突文件 → 编辑并删掉 `<<<<<<<` `=======` `>>>>>>>` 标记 → `git add` 标记已解决 → `git rebase --continue`（或 `git commit` 完成 merge）。

## 6. 暂存现场：stash

| 目的 | 命令 |
| --- | --- |
| 暂存当前改动 | `git stash` |
| 连未跟踪文件一起暂存 | `git stash -u` |
| 带说明 | `git stash push -m "调试到一半"` |
| 查看 stash 列表 | `git stash list` |
| 恢复最近一次并删除记录 | `git stash pop` |
| 恢复但不删除记录 | `git stash apply stash@{1}` |
| 查看某次 stash 内容 | `git stash show -p stash@{0}` |
| 丢弃某次 stash | `git stash drop stash@{0}` |

典型场景：正在改 A 分支，临时要切去修 B 分支的紧急 bug——`git stash -u`，修完切回来 `git stash pop`。

## 7. 远程协作

| 目的 | 命令 |
| --- | --- |
| 查看远程地址 | `git remote -v` |
| 添加远程 | `git remote add origin <url>` |
| 拉取但不合并 | `git fetch --all --prune` |
| 拉取并变基 | `git pull --rebase` |
| 推送新分支并建立跟踪 | `git push -u origin feature/x` |
| 查看本地与远程差异 | `git log --oneline HEAD..origin/main` |
| 修改远程地址 | `git remote set-url origin <new-url>` |

`fetch` 只更新远程跟踪分支（`origin/main`），不动你的工作区；`pull` = `fetch` + `merge`（或 `rebase`）。排查"为什么我的分支和远程不一样"时，先 `fetch` 再对比，比直接 `pull` 安全。

## 8. 配置与别名

首次使用至少配好身份和换行：

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
git config --global pull.rebase true
git config --global core.autocrlf input   # macOS/Linux；Windows 用 true
```

常用别名（写进 `~/.gitconfig` 的 `[alias]` 段）：

```bash
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.lg "log --oneline --graph --all --decorate"
git config --global alias.last "log -1 --stat"
git config --global alias.unstage "restore --staged"
```

`git lg` 基本可以替代大部分 `git log` 场景。查看所有配置来源：`git config --list --show-origin`。

## 9. 提交信息与忽略规则

提交信息建议遵循 Conventional Commits，一眼能看出改动性质：

```text
feat: 新增 xxx 功能
fix: 修复 xxx 崩溃
docs: 更新 README
refactor: 抽出公共函数
test: 补充单元测试
chore: 升级依赖
```

`.gitignore` 只对**未跟踪**文件生效。已经被跟踪的文件，加了规则也不会消失，需要先移出索引：

```bash
git rm --cached path/to/file     # 保留本地文件，停止跟踪
git rm -r --cached build/        # 整个目录
```

## 10. 急救清单

| 症状 | 处理 |
| --- | --- |
| 提交到错误分支 | `git switch 正确分支 && git cherry-pick <commit>`，再回原分支 `git reset --hard HEAD~1` |
| 忘了拉最新就提交 | `git pull --rebase`，有冲突按提示解决 |
| 合并乱了想重来 | `git merge --abort` 或 `git reset --hard ORIG_HEAD` |
| 误删分支 | `git reflog` 找到哈希，`git branch 分支名 <hash>` |
| 误 `reset --hard` | `git reflog` 找到之前 HEAD，`git reset --hard HEAD@{n}` |
| 推送被拒绝（non-fast-forward） | 先 `git pull --rebase`，再 `git push`；不要直接 `--force` |
| 大文件误提交 | 用 `git filter-repo` 清理历史，并加入 `.gitignore` |
| 想知道某行何时引入 | `git blame -L 10,20 path/to/file` |

## 11. 危险命令红榜

执行前必须确认目标分支、是否有未推送提交、是否是共享分支：

- `git reset --hard`：丢弃工作区与暂存区改动；
- `git clean -fd`：删除未跟踪文件/目录，无法从 Git 恢复；
- `git push --force`：覆盖远程历史，优先用 `--force-with-lease`；
- `git rebase` 共享分支：重写他人依赖的历史；
- `git filter-repo` / `filter-branch`：重写全部历史，团队需同步重新克隆。

::: tip 保命习惯
1. 动手前 `git status` + `git log --oneline -5` 看清现状；
2. 破坏性命令先加 `-n` / `--dry-run` 预览；
3. 不确定时先 `git stash` 或新建一个备份分支 `git branch backup/xxx`；
4. 共享分支上"只追加，不改写"。
:::

## 12. 学习路径与参考

- `git help <命令>` 或 `git <命令> --help` 看官方手册，`git help -a` 列出全部命令；
- [Pro Git（中文版）](https://git-scm.com/book/zh/v2)：最系统的免费教材，前 3 章足够日常使用；
- [Learn Git Branching](https://learngitbranching.js.org/?locale=zh_CN)：可视化交互练习分支、rebase、cherry-pick；
- 配合 [Linux 服务器常用命令速查](linux-server-commands.md) 在远程仓库环境中使用。
