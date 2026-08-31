import { defineConfig } from 'vitepress'

export default defineConfig({
  lang: 'zh-CN',
  title: 'CassieLee 的知识库',
  description: '记录大模型、AI Infra、AI Agent、后端与个人思考',
  base: '/my-lib/',
  cleanUrls: true,
  markdown: {
    math: true
  },

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '学习路线', link: '/roadmap' },
      { text: '大模型基础', link: '/llm-foundations/' },
      { text: 'AI Infra', link: '/ai-infra/' },
      { text: 'AI Agent', link: '/ai-agent/' },
      { text: '后端知识', link: '/backend/' },
      { text: '个人杂谈', link: '/essays/' }
    ],

    sidebar: {
      '/backend/': [{ text: '后端知识', items: [{ text: '目录', link: '/backend/' }] }],
      '/llm-foundations/': [
        {
          text: '大模型基础',
          items: [
            { text: '目录', link: '/llm-foundations/' },
            { text: '01｜全景：从模型到集合通信', link: '/llm-foundations/01-landscape' },
            { text: '02｜张量、计算图与反向传播', link: '/llm-foundations/02-tensor-autograd' },
            { text: '03｜文本如何变成模型输入', link: '/llm-foundations/03-tokenization-embedding' },
            { text: '04｜Self-Attention 与 QKV', link: '/llm-foundations/04-self-attention' },
            { text: '05｜完整 Decoder Block', link: '/llm-foundations/05-decoder-block' },
            { text: '06｜RoPE 与长上下文', link: '/llm-foundations/06-rope-long-context' }
          ]
        }
      ],
      '/ai-infra/': [{ text: 'AI Infra', items: [{ text: '目录', link: '/ai-infra/' }] }],
      '/ai-agent/': [{ text: 'AI Agent', items: [{ text: '目录', link: '/ai-agent/' }] }],
      '/essays/': [{ text: '个人杂谈', items: [{ text: '目录', link: '/essays/' }] }]
    },

    search: { provider: 'local' },
    outline: { label: '本页目录', level: [2, 3] },
    docFooter: { prev: '上一篇', next: '下一篇' },
    lastUpdated: { text: '最后更新于' },
    returnToTopLabel: '返回顶部',
    sidebarMenuLabel: '目录',
    darkModeSwitchLabel: '主题',
    socialLinks: [{ icon: 'github', link: 'https://github.com/cassielee-tech' }]
  }
})
