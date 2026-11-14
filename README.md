# CassieLee 的知识库

一个基于 VitePress 搭建的个人知识博客。核心内容是 **HCCL 开发主线**——一门面向 CANN 集合通信算子开发岗位的六站课程，其余栏目为主线之外的补充。

## 内容目录

- `docs/model-systems`：模型与系统（上篇·模型全景 + 下篇·训练与推理系统）
- `docs/execution-parallel`：执行与并行（上篇·单卡执行系统 + 下篇·并行策略）
- `docs/comm-ascend`：通信与昇腾（上篇·集合通信 + 下篇·昇腾与 HCCL，含双仓源码课程）
- `docs/practice`：实战复盘（独立专栏，含写作模板）
- `docs/now.md`：Now 页面（当前在做什么）
- `docs/index.md`：首页（含课程总览与专栏导航）
- `docs/backend`：后端知识
- `docs/tools-productivity`：工具与效率
- `docs/essays`：个人杂谈（含 AI Agent 文章与实用提示词收集）

> 历史：六专栏（model / systems / device / parallel / collective / ascend）已合并为上述三大专栏，旧路径由主题脚本自动重定向。

## 本地运行

```bash
npm install
npm run docs:dev
```

构建静态站点：

```bash
npm run docs:build
```

## License

MIT
