<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import FocusTimer from './FocusTimer.vue'
import PanelToggles from './PanelToggles.vue'

const { Layout } = DefaultTheme
</script>

<template>
  <Layout>
    <template #layout-bottom>
      <PanelToggles />
      <FocusTimer />
    </template>
  </Layout>
</template>
