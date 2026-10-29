<script setup lang="ts">
import { useData } from 'vitepress'
import { computed, ref } from 'vue'

const { page, frontmatter, theme } = useData()

const SIDEBAR_KEY = 'vp-sidebar-collapsed'
const ASIDE_KEY = 'vp-aside-collapsed'

// 与 VitePress 内部 useSidebar 的判断保持一致（仅用公开 API 复现）
function matchSidebar(relativePath: string): unknown[] {
  const config = theme.value?.sidebar
  if (Array.isArray(config)) return config
  if (!config) return []
  const path = relativePath.startsWith('/') ? relativePath : `/${relativePath}`
  const dir = Object.keys(config)
    .sort((a, b) => b.split('/').length - a.split('/').length)
    .find((k) => path.startsWith(k.startsWith('/') ? k : `/${k}`))
  return dir ? config[dir] : []
}

const isDocPage = computed(
  () =>
    !page.value.isNotFound &&
    (frontmatter.value.layout == null || frontmatter.value.layout === 'doc')
)

const hasSidebar = computed(
  () =>
    isDocPage.value &&
    frontmatter.value.sidebar !== false &&
    matchSidebar(page.value.relativePath).length > 0
)

const hasAside = computed(() => {
  if (!isDocPage.value) return false
  if (frontmatter.value.aside != null) return !!frontmatter.value.aside
  return theme.value?.aside !== false
})

const sidebarCollapsed = ref(false)
const asideCollapsed = ref(false)

if (typeof window !== 'undefined') {
  sidebarCollapsed.value = localStorage.getItem(SIDEBAR_KEY) === '1'
  asideCollapsed.value = localStorage.getItem(ASIDE_KEY) === '1'
  applyClasses()
}

function applyClasses() {
  const el = document.documentElement
  el.classList.toggle('sidebar-collapsed', sidebarCollapsed.value)
  el.classList.toggle('aside-collapsed', asideCollapsed.value)
}

function toggleSidebar() {
  sidebarCollapsed.value = !sidebarCollapsed.value
  localStorage.setItem(SIDEBAR_KEY, sidebarCollapsed.value ? '1' : '0')
  applyClasses()
}

function toggleAside() {
  asideCollapsed.value = !asideCollapsed.value
  localStorage.setItem(ASIDE_KEY, asideCollapsed.value ? '1' : '0')
  applyClasses()
}
</script>

<template>
  <button
    v-if="hasSidebar"
    type="button"
    class="panel-toggle panel-toggle--sidebar"
    :title="sidebarCollapsed ? '展开目录' : '收起目录'"
    :aria-label="sidebarCollapsed ? '展开目录' : '收起目录'"
    :aria-expanded="!sidebarCollapsed"
    @click="toggleSidebar"
  >
    <svg
      v-if="sidebarCollapsed"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2.5"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <polyline points="9 18 15 12 9 6" />
    </svg>
    <svg
      v-else
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2.5"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <polyline points="15 18 9 12 15 6" />
    </svg>
  </button>

  <button
    v-if="hasAside"
    type="button"
    class="panel-toggle panel-toggle--outline"
    :title="asideCollapsed ? '展开大纲' : '收起大纲'"
    :aria-label="asideCollapsed ? '展开大纲' : '收起大纲'"
    :aria-expanded="!asideCollapsed"
    @click="toggleAside"
  >
    <svg
      v-if="asideCollapsed"
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2.5"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <polyline points="15 18 9 12 15 6" />
    </svg>
    <svg
      v-else
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2.5"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <polyline points="9 18 15 12 9 6" />
    </svg>
  </button>
</template>
