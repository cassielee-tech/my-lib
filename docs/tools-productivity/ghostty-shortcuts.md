# Ghostty 快捷键速查

> 适用于 macOS 版 Ghostty。Ghostty 的默认按键会随平台和版本变化，也可能被个人配置覆盖；遇到不一致时，以本机输出和菜单栏显示为准。

::: tip 使用建议
不要背完整张表。遇到任务时搜索功能名，每周只挑 1～2 个高频快捷键练习。建议先掌握：新建标签、切换标签、分屏、切换分屏和清屏。
:::

- [ ] 今天只选择一个新快捷键
- [ ] 实际使用三次
- [ ] 不常用的继续留在速查表，不占记忆

## 1. 先查看本机真实按键

查看当前版本的全部默认快捷键：

```bash
ghostty +list-keybinds --default
```

查看合并个人配置后真正生效的快捷键：

```bash
ghostty +list-keybinds
```

查看 Ghostty 支持的全部动作：

```bash
ghostty +list-actions
```

这三条命令比网上的静态表格更可靠，因为 Ghostty 的 macOS 与 Linux 默认值不同，版本升级也可能增加动作。

## 2. 窗口与标签页

| 功能 | macOS 快捷键 |
| --- | --- |
| 新建窗口 | `⌘ N` |
| 新建标签页 | `⌘ T` |
| 关闭当前 Surface / 标签 | `⌘ W` |
| 退出 Ghostty | `⌘ Q` |
| 下一个标签页 | `⌘ ⇧ ]` |
| 上一个标签页 | `⌘ ⇧ [` |
| 跳到第 1～8 个标签页 | `⌘ 1` ～ `⌘ 8` |
| 跳到最后一个标签页 | `⌘ 9` |
| 切换全屏 | `⌃ ⌘ F` |

Ghostty 使用 **Surface** 表示一个终端表面。一个标签页中可以有多个分屏 Surface，因此 `close_surface` 会关闭当前聚焦分屏；只剩一个分屏时，看起来就是关闭标签页。

## 3. 分屏

| 功能 | macOS 默认快捷键 |
| --- | --- |
| 向右分屏 | `⌘ D` |
| 向下分屏 | `⌘ ⇧ D` |

分屏后的焦点移动、调整大小和放大当前分屏，在不同 Ghostty 版本或个人配置中更容易出现差异。先用下面命令搜索本机绑定：

```bash
ghostty +list-keybinds | grep -E 'goto_split|resize_split|toggle_split_zoom|equalize_splits'
```

如果希望使用类似 Vim 的方向键，可以在 Ghostty 配置中加入：

```ini
keybind = opt+h=goto_split:left
keybind = opt+j=goto_split:down
keybind = opt+k=goto_split:up
keybind = opt+l=goto_split:right
keybind = opt+enter=new_split:auto
keybind = opt+shift+enter=toggle_split_zoom
```

`new_split:auto` 会根据当前区域宽高自动选择分屏方向。自定义键位发生冲突时，后出现的同一 Trigger 会覆盖先前绑定。

## 4. 复制、粘贴与字体

| 功能 | 快捷键 |
| --- | --- |
| 复制选中内容 | `⌘ C` |
| 粘贴 | `⌘ V` |
| 增大字体 | `⌘ +` |
| 减小字体 | `⌘ -` |
| 恢复默认字体大小 | `⌘ 0` |
| 清理可见终端屏幕 | `⌘ K` |

终端中的 `⌃ C` 通常是向前台进程发送中断信号，不是复制。macOS 下应使用 `⌘ C` 复制已选文本。

## 5. 配置

| 功能 | 快捷键 |
| --- | --- |
| 打开配置文件 | `⌘ ,` |
| 重新加载配置 | `⌘ ⇧ ,` |

Ghostty 配置文件通常位于：

```text
~/.config/ghostty/config
```

自定义按键的格式为：

```ini
keybind = 触发按键=动作
```

例如：

```ini
keybind = cmd+shift+r=reload_config
keybind = global:cmd+backquote=toggle_quick_terminal
```

`global:` 表示 Ghostty 不在前台时也能触发；在 macOS 上需要为 Ghostty 授予“辅助功能”权限。`cmd`、`command` 和 `super` 是同一修饰键的别名。

## 6. Shell 行编辑快捷键

下面这些通常由 Zsh/Bash 的行编辑器或终端中的程序处理，**不是 Ghostty 独有快捷键**：

| 功能 | 常见按键 |
| --- | --- |
| 移到行首 / 行尾 | `⌃ A` / `⌃ E` |
| 向前 / 向后移动一个字符 | `⌃ F` / `⌃ B` |
| 删除光标前一个字符 | `⌃ H` |
| 删除光标前一个单词 | `⌃ W` |
| 删除光标到行首 | `⌃ U` |
| 删除光标到行尾 | `⌃ K` |
| 清屏 | `⌃ L` |
| 搜索历史命令 | `⌃ R` |
| 中断当前进程 | `⌃ C` |
| 发送 EOF / 空行时退出 Shell | `⌃ D` |
| 暂停前台进程 | `⌃ Z` |

Shell 开启 Vim 编辑模式、运行 Tmux、Neovim 或其他 TUI 程序后，按键含义可能完全改变。排查冲突时应依次判断：按键被 macOS、Ghostty、Tmux，还是终端内程序消费了。

## 7. 快捷键配置语法

Ghostty 支持组合键、按键序列和触发前缀：

```ini
# 组合键
keybind = ctrl+shift+t=new_tab

# 两段按键序列：先按 ctrl+a，再单独按 c
keybind = ctrl+a>c=new_tab

# 只有动作可以执行时才消费按键
keybind = performable:ctrl+c=copy_to_clipboard

# 触发动作后，仍把按键传给终端程序
keybind = unconsumed:ctrl+a=reload_config
```

常用前缀：

- `global:`：应用不在前台也可触发；
- `all:`：对所有终端 Surface 执行动作；
- `performable:`：动作当前可执行时才消费输入；
- `unconsumed:`：执行动作后继续把输入发送给终端程序。

## 8. 参考资料

- [Ghostty 官方文档：Keybindings](https://ghostty.org/docs/config/keybind)
- [Ghostty 官方文档：Action Reference](https://ghostty.org/docs/config/keybind/reference)
- [Ghostty 官方文档：Configuration](https://ghostty.org/docs/config)
