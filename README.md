# CassieLee 的知识库

一个基于 VitePress 搭建的个人知识博客。核心内容是 **HCCL 开发主线**——一门面向 CANN 集合通信算子开发岗位的六站课程，其余栏目为主线之外的补充。

## 内容目录

- `docs/model`：模型全景（HCCL 开发主线·专栏一）
- `docs/systems`：训练与推理系统（专栏二）
- `docs/device`：单卡执行系统（专栏三）
- `docs/parallel`：并行策略（专栏四）
- `docs/collective`：集合通信·岗位核心（专栏五）
- `docs/ascend`：昇腾与 HCCL（专栏六，含 HCCL 源码专题）
- `docs/roadmap.md`：课程总览（六专栏导航）
- `docs/ai-agent`：AI Agent
- `docs/backend`：后端知识
- `docs/tools-productivity`：工具与效率
- `docs/essays`：个人杂谈

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
