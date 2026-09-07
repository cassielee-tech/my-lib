<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'
import FocusTimer from './FocusTimer.vue'

const { Layout } = DefaultTheme
</script>

<template>
  <Layout>
    <template #layout-bottom>
      <FocusTimer />
    </template>
  </Layout>
</template>
