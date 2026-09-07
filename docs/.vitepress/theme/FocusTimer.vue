<script setup lang="ts">
import { computed, onBeforeUnmount, ref } from 'vue'

const SESSION_SECONDS = 15 * 60
const remaining = ref(SESSION_SECONDS)
const running = ref(false)
const completed = ref(false)

let deadline = 0
let timer: ReturnType<typeof setInterval> | undefined

const timeText = computed(() => {
  const minutes = Math.floor(remaining.value / 60)
  const seconds = remaining.value % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
})

function updateRemaining() {
  remaining.value = Math.max(0, Math.ceil((deadline - Date.now()) / 1000))
  if (remaining.value === 0) {
    running.value = false
    completed.value = true
    stopTimer()
  }
}

function stopTimer() {
  if (timer) clearInterval(timer)
  timer = undefined
}

function toggleTimer() {
  if (running.value) {
    updateRemaining()
    running.value = false
    stopTimer()
    return
  }

  if (remaining.value === 0) remaining.value = SESSION_SECONDS
  completed.value = false
  deadline = Date.now() + remaining.value * 1000
  running.value = true
  timer = setInterval(updateRemaining, 250)
}

function resetTimer() {
  stopTimer()
  running.value = false
  completed.value = false
  remaining.value = SESSION_SECONDS
}

onBeforeUnmount(stopTimer)
</script>

<template>
  <aside class="focus-timer" aria-label="15 分钟倒计时">
    <div class="focus-timer__display" aria-live="polite">
      <span class="focus-timer__label">{{ completed ? '本单元完成' : '专注计时' }}</span>
      <time :datetime="`PT${remaining}S`">{{ timeText }}</time>
    </div>
    <div class="focus-timer__actions">
      <button type="button" @click="toggleTimer">
        {{ running ? '暂停' : completed ? '再来一次' : remaining < SESSION_SECONDS ? '继续' : '开始' }}
      </button>
      <button type="button" class="secondary" @click="resetTimer">重置</button>
    </div>
  </aside>
</template>
