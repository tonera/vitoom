<template>
  <div class="vt-card overflow-hidden flex flex-col h-full">
    <div
      class="relative h-96 border-b border-gray-200 bg-gray-100 sm:h-72 lg:h-80 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-900/30"
    >
      <div class="absolute left-3 top-3 z-10 w-2.5 h-2.5 rounded-full" :class="model.service_status === 'active' ? 'bg-green-500' : 'bg-gray-600'" />

      <div class="absolute right-3 top-3 z-10 flex items-center gap-2">
        <button
          v-if="showAdvancedConfig"
          type="button"
          class="px-2.5 py-1 rounded-lg text-xs bg-indigo-600/90 text-white hover:bg-indigo-500 cursor-pointer"
          @click="emit('advancedConfig', model)"
        >
          {{ t('models.advancedConfig') }}
        </button>

        <button type="button" class="px-2.5 py-1 rounded-lg text-xs bg-indigo-600/90 text-white hover:bg-indigo-500 cursor-pointer" @click="emit('edit', model)">
          {{ t('models.edit') }}
        </button>
        <button type="button" class="px-2.5 py-1 rounded-lg text-xs bg-rose-600/90 text-white hover:bg-rose-500 cursor-pointer" @click="emit('delete', model)">
          {{ t('common.delete') }}
        </button>
      </div>

      <img
        v-if="model.thumb"
        :src="resolveBackendPublicUrl(model.thumb)"
        class="absolute inset-0 z-0 w-full h-full object-cover block"
        :alt="model.name"
        loading="lazy"
        referrerpolicy="no-referrer"
      />
      <div
        v-else
        class="flex h-full w-full items-center justify-center text-gray-400 [.dark_&]:text-gray-500"
      >
        <div
          class="flex h-16 w-16 items-center justify-center rounded-2xl border border-gray-200 bg-white text-lg font-semibold text-gray-600 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/40 [.dark_&]:text-gray-300"
        >
          {{ abbr(model.name || model.id) }}
        </div>
      </div>
    </div>

    <div class="p-4 flex-1 flex flex-col gap-2">
      <div class="flex items-center justify-between gap-3">
        <div
          class="min-w-0 truncate text-sm font-semibold text-gray-900 [.dark_&]:text-white"
          :title="model.name"
        >
          {{ model.name }}
        </div>
        <button
          type="button"
          class="relative inline-flex h-6 w-11 shrink-0 cursor-pointer items-center rounded-full border transition-colors"
          :class="model.service_status === 'active' ? 'border-indigo-500/40 bg-indigo-600' : 'border-gray-300 bg-gray-200 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/60'"
          :disabled="busy"
          :title="model.service_status === 'active' ? t('models.deactivate') : t('models.activate')"
          @click="emit('toggleActive', model)"
        >
          <span class="inline-block h-5 w-5 transform rounded-full bg-white transition-transform" :class="model.service_status === 'active' ? 'translate-x-5' : 'translate-x-1'" />
        </button>
      </div>
      <div class="flex flex-wrap gap-2">
        <span
          v-if="model.family"
          class="inline-flex items-center rounded-md border border-indigo-200/90 bg-indigo-50 px-2 py-0.5 text-[11px] leading-none text-indigo-800 [.dark_&]:border-indigo-500/20 [.dark_&]:bg-indigo-500/15 [.dark_&]:text-indigo-200"
        >
          {{ model.family }}
        </span>
        <span
          v-if="model.asset_type"
          class="inline-flex items-center rounded-md border border-gray-200 bg-gray-100 px-2 py-0.5 text-[11px] leading-none text-gray-800 [.dark_&]:border-white/10 [.dark_&]:bg-gray-900/30 [.dark_&]:text-gray-200"
        >
          {{ model.asset_type }}
        </span>
        <span
          v-if="model.storage_mode"
          class="inline-flex items-center rounded-md border border-gray-200 bg-gray-100 px-2 py-0.5 text-[11px] leading-none text-gray-800 [.dark_&]:border-white/10 [.dark_&]:bg-black/30 [.dark_&]:text-gray-200"
        >
          {{ storageValueToLabel(model.storage_mode) }}
        </span>
      </div>

      <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">
        {{ formatTime(model.created_at) }}
      </div>

      <div class="mt-auto flex items-center gap-3">
        <div class="min-w-0 flex-1 text-xs text-gray-600 [.dark_&]:text-gray-500" :title="downloadTitle">
          <button v-if="hasDownloadMeta && notDownloaded" type="button" class="w-full text-left cursor-pointer" @click="emit('openDownloadTerminal', model)">
            <div class="flex items-center gap-2">
              <span class="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] leading-none border shrink-0" :class="downloadStatusTagClass">
                {{ downloadLabel }}
              </span>
              <div
                class="h-2 min-w-0 flex-1 overflow-hidden rounded border border-gray-300 bg-gray-200 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/60"
              >
                <div v-if="downloadPercent !== null" class="h-full bg-indigo-600/80" :style="{ width: `${downloadPercent}%` }" />
                <div v-else class="h-full w-2/3 bg-indigo-600/50 animate-pulse" />
              </div>
            </div>
          </button>
          <div v-else class="flex items-center gap-2 min-w-0">
            <span
              v-if="hasDownloadMeta && downloadStatus !== 'completed'"
              class="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] leading-none border shrink-0"
              :class="downloadStatusTagClass"
            >
              {{ downloadLabel }}
            </span>
            <div
              class="cursor-pointer select-none truncate text-gray-600 [.dark_&]:text-gray-400"
              :title="t('models.openDownloadTerminal')"
              @dblclick.stop.prevent="emit('openDownloadTerminal', model)"
            >
              {{ model.load_name || '' }}
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { ModelRecord } from '../../../api/models'
import { abbr, downloadStatusLabel, extractDownloadPercent, formatTime, isNotDownloadCompleted, storageValueToLabel } from './modelListFormatters'

const { t } = useI18n()

const props = defineProps<{
  model: ModelRecord
  busy: boolean
  resolveBackendPublicUrl: (p: string) => string
}>()

const emit = defineEmits<{
  (e: 'advancedConfig', m: ModelRecord): void
  (e: 'edit', m: ModelRecord): void
  (e: 'delete', m: ModelRecord): void
  (e: 'toggleActive', m: ModelRecord): void
  (e: 'openDownloadTerminal', m: ModelRecord): void
}>()

const notDownloaded = computed(() => isNotDownloadCompleted(props.model as any))
const showAdvancedConfig = computed(() => {
  const modality = String(props.model?.modality || '').trim().toLowerCase()
  return modality === 'image' || modality === 'video'
})
const downloadPercent = computed(() => extractDownloadPercent(props.model as any))
const downloadLabel = computed(() => downloadStatusLabel(props.model as any))
const downloadStatus = computed(() => String((props.model as any)?.download_status || '').trim().toLowerCase() || 'pending')
const hasDownloadMeta = computed(() => {
  const m: any = props.model as any
  const st = String(m?.download_status || '').trim()
  const source = m?.source && typeof m.source === 'object' ? m.source : {}
  const rid = String(source?.repo_id || '').trim()
  const desc = String(m?.description || '')
  return Boolean(st || rid || (desc.includes('--- download ---') && desc.includes('--- /download ---')))
})

const downloadStatusTagClass = computed(() => {
  const st = downloadStatus.value
  if (st === 'completed') {
    return 'border border-emerald-200/80 bg-emerald-50 text-emerald-800 [.dark_&]:border-emerald-500/20 [.dark_&]:bg-emerald-500/15 [.dark_&]:text-emerald-200'
  }
  if (st === 'downloading') {
    return 'border border-indigo-200/80 bg-indigo-50 text-indigo-800 [.dark_&]:border-indigo-500/20 [.dark_&]:bg-indigo-500/15 [.dark_&]:text-indigo-200'
  }
  if (st === 'ending') {
    return 'border border-amber-200/80 bg-amber-50 text-amber-800 [.dark_&]:border-amber-500/20 [.dark_&]:bg-amber-500/15 [.dark_&]:text-amber-200'
  }
  if (st === 'failed') {
    return 'border border-rose-200/80 bg-rose-50 text-rose-800 [.dark_&]:border-rose-500/20 [.dark_&]:bg-rose-500/15 [.dark_&]:text-rose-200'
  }
  if (st === 'canceled' || st === 'cancelled') {
    return 'border border-gray-200 bg-gray-100 text-gray-700 [.dark_&]:border-white/10 [.dark_&]:bg-gray-900/30 [.dark_&]:text-gray-200'
  }
  return 'border border-sky-200/80 bg-sky-50 text-sky-800 [.dark_&]:border-sky-500/20 [.dark_&]:bg-sky-500/15 [.dark_&]:text-sky-200'
})

const downloadTitle = computed(() => {
  const p = props.model.load_name || ''
  if (!hasDownloadMeta.value) return p
  if (downloadStatus.value === 'completed') return p
  if (notDownloaded.value) return downloadLabel.value
  return [downloadLabel.value, p].filter(Boolean).join(' · ')
})
</script>

