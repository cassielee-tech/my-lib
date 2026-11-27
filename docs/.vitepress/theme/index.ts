import DefaultTheme from 'vitepress/theme'
import Layout from './Layout.vue'
import './custom.css'

const BASE = '/my-lib/'

/**
 * 旧专栏 URL → 新合并专栏 URL。
 * 六专栏合并为三专栏后，外部引用的旧链接由 GitHub Pages 的 404.html
 * 加载本主题脚本，在此做客户端重定向。
 */
const REDIRECTS: Array<[string, string]> = [
  // 2026-11 目录中文化：英文路径 → 中文路径（前缀规则，具体在前）
  ['/comm-ascend/reference/debug/cases', '/通信与昇腾/参考手册/排障手册/案例'],
  ['/comm-ascend/reference/debug', '/通信与昇腾/参考手册/排障手册'],
  ['/comm-ascend/reference', '/通信与昇腾/参考手册'],
  ['/comm-ascend/hccl-hcomm', '/通信与昇腾/源码课程/'],
  ['/comm-ascend', '/通信与昇腾'],
  ['/model-systems', '/模型与系统'],
  ['/execution-parallel', '/执行与并行'],
  ['/practice', '/实战复盘'],
  ['/backend', '/后端知识'],
  ['/tools-productivity', '/工具与效率'],
  ['/essays', '/个人杂谈'],
  // 成对专栏的速通拆为上篇/下篇（须在前缀规则之前）
  ['/model/00-速通', '/模型与系统/00-速通-上篇'],
  ['/systems/00-速通', '/模型与系统/00-速通-下篇'],
  ['/device/00-速通', '/执行与并行/00-速通-上篇'],
  ['/parallel/00-速通', '/执行与并行/00-速通-下篇'],
  // practice 独立成 /实战复盘/（须在 /ascend 之前）
  ['/ascend/practice', '/实战复盘'],
  // 六专栏 → 三专栏
  ['/model', '/模型与系统'],
  ['/systems', '/模型与系统'],
  ['/device', '/执行与并行'],
  ['/parallel', '/执行与并行'],
  ['/collective', '/通信与昇腾'],
  ['/ascend', '/通信与昇腾']
]

function redirectFor(pathname: string): string | null {
  try {
    let p = decodeURIComponent(pathname)
    if (!p.startsWith(BASE)) return null
    p = p.slice(BASE.length - 1)          // 保留开头的 /
    p = p.replace(/\.html$/, '').replace(/\/$/, '')
    for (const [old, nu] of REDIRECTS) {
      if (p === old || p.startsWith(old + '/')) {
        const rest = p.slice(old.length)
        // 旧专栏目录页已合并进新专栏目录
        if (rest === '' || rest === '/index') return nu + '/'
        return nu + rest
      }
    }
  } catch {
    /* decode 失败则不重定向 */
  }
  return null
}

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp() {
    if (typeof window === 'undefined') return
    const to = redirectFor(window.location.pathname)
    if (to) {
      window.location.replace(BASE.slice(0, -1) + to + window.location.hash)
    }
  }
}
