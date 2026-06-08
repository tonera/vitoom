<template>
  <article class="min-w-0 overflow-hidden rounded-2xl border border-gray-200 bg-white p-4 shadow-sm [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
    <div class="flex items-start justify-between gap-3">
      <div class="min-w-0">
        <div class="flex items-center gap-2">
          <span class="rounded-full px-2 py-0.5 text-xs font-medium" :class="badgeClass">{{ badgeText }}</span>
          <span class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ result.createdAt }}</span>
        </div>
        <h3 class="mt-2 truncate text-sm font-semibold text-gray-950 [.dark_&]:text-white" :title="result.title">
          {{ result.title }}
        </h3>
        <p v-if="result.subtitle" class="mt-1 truncate text-xs text-gray-500 [.dark_&]:text-gray-400" :title="result.subtitle">
          {{ result.subtitle }}
        </p>
      </div>
      <button
        v-if="result.url"
        type="button"
        class="shrink-0 rounded-xl border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-700 transition hover:bg-gray-50 [.dark_&]:border-gray-700 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800 cursor-pointer"
        @click="$emit('download', result)"
      >
        {{ t('audio.resultCard.download') }}
      </button>
    </div>

    <div class="mt-4 min-w-0">
      <div v-if="result.status === 'pending'" class="flex min-h-24 flex-col items-center justify-center gap-3 rounded-xl bg-gray-50 px-6 text-sm text-gray-500 [.dark_&]:bg-gray-950/60 [.dark_&]:text-gray-400">
        <span>{{ pendingText }}</span>
        <div v-if="progressPercent !== undefined" class="h-2 w-full max-w-sm overflow-hidden rounded-full bg-gray-200 [.dark_&]:bg-gray-800">
          <div class="h-full rounded-full bg-amber-500 transition-all duration-300" :style="{ width: `${progressPercent}%` }" />
        </div>
      </div>

      <audio v-else-if="result.kind === 'audio' && result.url" class="w-full" :src="result.url" controls preload="metadata" />

      <div v-else-if="result.kind === 'text'" class="min-w-0 overflow-hidden rounded-xl bg-gray-50 p-3 [.dark_&]:bg-gray-950/60">
        <pre v-if="result.text" class="max-h-56 max-w-full overflow-auto whitespace-pre-wrap break-all text-sm leading-6 text-gray-800 [.dark_&]:text-gray-100">{{ result.text }}</pre>
        <div v-else class="flex items-center justify-between gap-3">
          <p class="text-sm text-gray-500 [.dark_&]:text-gray-400">{{ t('audio.resultCard.asrReady') }}</p>
          <button
            v-if="result.url"
            type="button"
            class="rounded-xl bg-gray-950 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-gray-800 [.dark_&]:bg-white [.dark_&]:text-gray-950 [.dark_&]:hover:bg-gray-200 cursor-pointer"
            @click="$emit('loadText', result)"
          >
            {{ t('audio.resultCard.readText') }}
          </button>
        </div>
      </div>

      <div v-else class="rounded-xl bg-gray-50 p-3 text-sm text-gray-500 [.dark_&]:bg-gray-950/60 [.dark_&]:text-gray-400">
        {{ t('audio.resultCard.noPreview') }}
      </div>
    </div>

    <p v-if="result.prompt" class="mt-3 line-clamp-2 text-xs leading-5 text-gray-500 [.dark_&]:text-gray-400">
      {{ result.prompt }}
    </p>
  </article>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

export type AudioResultKind = 'audio' | 'text'
export type AudioResultStatus = 'pending' | 'completed' | 'failed'

export interface AudioResultItem {
  key: string
  runId?: string
  kind: AudioResultKind
  status: AudioResultStatus
  progress?: number
  title: string
  subtitle?: string
  prompt?: string
  url?: string
  text?: string
  fileName?: string
  mimeType?: string
  createdAt: string
}

const props = defineProps<{
  result: AudioResultItem
}>()

defineEmits<{
  download: [result: AudioResultItem]
  loadText: [result: AudioResultItem]
}>()

const { t } = useI18n()

const badgeText = computed(() => {
  if (props.result.status === 'pending') return t('audio.resultCard.statusPending')
  if (props.result.status === 'failed') return t('audio.resultCard.statusFailed')
  return props.result.kind === 'audio' ? t('audio.resultCard.kindAudio') : t('audio.resultCard.kindText')
})

const badgeClass = computed(() => {
  if (props.result.status === 'pending') return 'bg-amber-100 text-amber-700 [.dark_&]:bg-amber-950/40 [.dark_&]:text-amber-300'
  if (props.result.status === 'failed') return 'bg-red-100 text-red-700 [.dark_&]:bg-red-950/40 [.dark_&]:text-red-300'
  if (props.result.kind === 'audio') return 'bg-indigo-100 text-indigo-700 [.dark_&]:bg-indigo-950/40 [.dark_&]:text-indigo-300'
  return 'bg-emerald-100 text-emerald-700 [.dark_&]:bg-emerald-950/40 [.dark_&]:text-emerald-300'
})

const progressPercent = computed(() => {
  const value = props.result.progress
  if (typeof value !== 'number' || !Number.isFinite(value)) return undefined
  return Math.max(0, Math.min(100, Math.round(value)))
})

const pendingText = computed(() => {
  if (progressPercent.value === undefined) return t('audio.resultCard.processing')
  return t('audio.resultCard.processingWithProgress', { progress: progressPercent.value })
})
</script>
