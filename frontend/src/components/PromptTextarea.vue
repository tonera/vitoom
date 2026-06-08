<template>
  <div class="relative w-full">
    <div class="indicator w-full relative">
      <textarea
        :placeholder="placeholderText"
        :rows="rows"
        :class="[
          'w-full px-3 py-2 rounded-lg border bg-white text-gray-950 placeholder-gray-400 resize-y focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all [.dark_&]:bg-gray-700 [.dark_&]:text-white [.dark_&]:placeholder-gray-400',
          isHighlight ? 'ring-2 ring-offset-2 ring-indigo-500 shadow-lg z-10 duration-200' : (preferSlowFade ? 'duration-700 border-gray-300 [.dark_&]:border-gray-600' : 'duration-150 border-gray-300 [.dark_&]:border-gray-600')
        ]"
        style="transition-property: border-color, background-color, ring, shadow, outline;"
        :value="modelValue"
        @input="onInput"
        @paste="onInput"
        :maxlength="1024"
      ></textarea>
      <button
        v-if="history && history.length > 0"
        @click="cycleHistory"
        class="absolute top-2 right-2 p-1.5 text-gray-500 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors [.dark_&]:text-gray-400 [.dark_&]:hover:text-gray-200 [.dark_&]:hover:bg-gray-600"
        :title="t('components.promptTextarea.cycleHistory')"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      </button>
    </div>
    <p v-if="showCharCount" class="mt-1 text-xs text-gray-500 text-right [.dark_&]:text-gray-400">
      {{ modelValue.length }}/1024
    </p>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onBeforeUnmount } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = withDefaults(defineProps<{
  modelValue: string
  placeholder?: string
  rows?: number
  history?: string[]
  showCharCount?: boolean
}>(), {
  modelValue: '',
  placeholder: undefined,
  rows: 4,
  history: () => [],
  showCharCount: false,
})

const placeholderText = computed(() => props.placeholder ?? t('components.promptTextarea.defaultPlaceholder'))

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

const lastPromptIndex = ref(-1)

// 供循环使用的历史列表：当存在历史项时，在首位加入空字符串
const historyList = computed(() => {
  const base = Array.isArray(props.history) ? props.history : []
  if (base.length > 0) {
    return ['', ...base]
  }
  return base
})

// 当历史记录发生变化时，重置索引
watch(() => props.history, () => {
  lastPromptIndex.value = -1
}, { deep: true })

// 高亮相框效果：对外暴露方法 flashFrame(duration, opts)
const isHighlight = ref(false)
let highlightTimer: ReturnType<typeof setTimeout> | null = null
const preferSlowFade = ref(false)

function flashFrame(duration = 3000, opts: { slowFade?: boolean } = {}) {
  const slow = !!opts?.slowFade
  preferSlowFade.value = !!slow
  if (highlightTimer) {
    clearTimeout(highlightTimer)
    highlightTimer = null
  }
  // 重启一次高亮，确保"闪烁"生效
  if (isHighlight.value) {
    isHighlight.value = false
    requestAnimationFrame(() => {
      isHighlight.value = true
    })
  } else {
    isHighlight.value = true
  }
  highlightTimer = setTimeout(() => {
    isHighlight.value = false
    highlightTimer = null
    // 慢速淡出结束后重置慢速偏好，避免影响后续闪烁
    if (preferSlowFade.value) {
      setTimeout(() => { preferSlowFade.value = false }, 800)
    }
  }, Math.max(0, duration))
}

defineExpose({ flashFrame })

onBeforeUnmount(() => {
  if (highlightTimer) {
    clearTimeout(highlightTimer)
    highlightTimer = null
  }
})

function onInput(e: Event) {
  const target = e.target as HTMLTextAreaElement
  const str = target?.value ?? ''
  const result = str.slice(0, 1024)
  emit('update:modelValue', result)
}

function cycleHistory() {
  const list = Array.isArray(historyList.value) ? historyList.value : []
  const len = list.length
  if (len === 0) return
  if (lastPromptIndex.value < 0 || lastPromptIndex.value >= len) {
    lastPromptIndex.value = 0 // 首次从最新一条开始（history 使用 unshift 存储）
  } else {
    lastPromptIndex.value = (lastPromptIndex.value + 1) % len // 依次 p3 -> p2 -> p1
  }
  const nextPrompt = String(list[lastPromptIndex.value] || '')
  emit('update:modelValue', nextPrompt)
}
</script>

