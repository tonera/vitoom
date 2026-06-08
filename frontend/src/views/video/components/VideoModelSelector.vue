<template>
  <div class="w-full">
    <button
      type="button"
      class="group flex w-full cursor-pointer items-center gap-3 rounded-lg border border-gray-200 bg-white px-3 py-2 text-left transition-colors hover:border-gray-300 hover:bg-gray-50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/35 [.dark_&]:hover:border-gray-600/80 [.dark_&]:hover:bg-gray-800/55"
      @click="showDialog = true"
    >
      <div
        class="h-10 w-10 shrink-0 overflow-hidden rounded-lg border border-gray-200 bg-gray-100 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/50"
      >
        <img v-if="selectedThumb" :src="selectedThumb" class="h-full w-full object-cover" :alt="selectedTitle" />
        <div
          v-else
          class="flex h-full w-full items-center justify-center text-xs text-gray-500 [.dark_&]:text-gray-400"
        >
          {{ selectedTitle.slice(0, 2) || t('video.modelSelector.defaultLabel') }}
        </div>
      </div>
      <div class="min-w-0 flex-1">
        <div class="truncate text-sm font-medium text-gray-900 [.dark_&]:text-white">
          {{ selectedTitle }}
        </div>
        <div
          v-if="selectedSubtitle"
          class="mt-0.5 truncate text-xs text-gray-500 [.dark_&]:text-gray-400"
        >
          {{ selectedSubtitle }}
        </div>
        <div v-if="selectedBadges.length > 0" class="mt-1.5 flex flex-wrap gap-1">
          <span
            v-for="badge in selectedBadges.slice(0, 4)"
            :key="badge"
            class="inline-flex items-center rounded-md border border-indigo-200/90 bg-indigo-50 px-2 py-0.5 text-[10px] leading-none text-indigo-800 [.dark_&]:border-indigo-500/20 [.dark_&]:bg-indigo-500/15 [.dark_&]:text-indigo-100"
          >
            {{ badge }}
          </span>
        </div>
      </div>
      <button
        v-if="allowClear && modelValue"
        type="button"
        class="shrink-0 cursor-pointer rounded-lg p-1.5 text-gray-400 transition-colors hover:bg-gray-100 hover:text-gray-800 [.dark_&]:hover:bg-gray-700/60 [.dark_&]:hover:text-white"
        :title="t('common.clear')"
        @click.stop="emit('update:modelValue', '')"
      >
        <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
      <div class="shrink-0 text-gray-500 [.dark_&]:text-gray-400">
        <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
        </svg>
      </div>
    </button>

    <Teleport to="body">
      <div
        v-if="showDialog"
        class="fixed inset-0 z-9999 bg-black/60 flex items-center justify-center p-4"
        @click.self="showDialog = false"
      >
        <div class="vt-card w-full max-w-5xl h-[80vh] overflow-hidden flex flex-col">
          <div
            class="flex items-center justify-between gap-4 border-b border-gray-200 px-5 py-4 [.dark_&]:border-gray-700/70"
          >
            <div class="min-w-0">
              <div class="truncate text-lg font-semibold text-gray-900 [.dark_&]:text-white">{{ t('video.modelSelector.title') }}</div>
              <div class="truncate text-xs text-gray-500 [.dark_&]:text-gray-400">
                {{ t('video.modelSelector.compatibleCount', { task: displayTaskLabel, count: models.length }) }}
              </div>
            </div>
            <div class="flex items-center gap-2">
              <input
                v-model="keyword"
                type="text"
                :placeholder="t('video.modelSelector.searchPlaceholder')"
                class="w-64 max-w-[40vw] rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/60 [.dark_&]:text-white [.dark_&]:placeholder:text-gray-500"
              />
              <button
                type="button"
                class="cursor-pointer rounded-lg p-2 text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-900 [.dark_&]:text-gray-400 [.dark_&]:hover:bg-gray-800/60 [.dark_&]:hover:text-white"
                @click="showDialog = false"
                :aria-label="t('common.close')"
              >
                <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>

          <div class="min-h-0 flex-1 overflow-y-auto p-4 vt-scroll">
            <div v-if="loading" class="text-sm text-gray-500 [.dark_&]:text-gray-400">{{ t('common.modelsLoading') }}</div>

            <div
              v-else-if="filteredModels.length === 0"
              class="flex h-full items-center justify-center text-center"
            >
              <div>
                <div class="text-base font-medium text-gray-800 [.dark_&]:text-gray-200">{{ t('video.modelSelector.noModels') }}</div>
                <div class="mt-1 text-sm text-gray-500 [.dark_&]:text-gray-500">
                  {{ keyword ? t('video.modelSelector.tryShorterKeyword') : displayEmptyText }}
                </div>
              </div>
            </div>

            <div v-else class="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
              <button
                v-for="model in filteredModels"
                :key="model.value"
                type="button"
                class="group relative cursor-pointer overflow-hidden rounded-xl border border-gray-200 bg-white/90 text-left transition-all hover:border-gray-300 hover:bg-gray-50/95 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/35 [.dark_&]:hover:border-gray-600/80 [.dark_&]:hover:bg-gray-800/55"
                :class="
                  model.value === modelValue
                    ? 'border-indigo-500/70 ring-2 ring-indigo-500/60'
                    : ''
                "
                @click="handleSelect(model.value)"
              >
                <div class="flex gap-3 p-3">
                  <div
                    class="h-[72px] w-[72px] shrink-0 overflow-hidden rounded-lg border border-gray-200 bg-gray-100 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/50"
                  >
                    <img
                      v-if="resolveThumb(model.thumb)"
                      :src="resolveThumb(model.thumb)"
                      class="h-full w-full object-cover"
                      :alt="model.label"
                    />
                    <div
                      v-else
                      class="flex h-full w-full items-center justify-center text-xs text-gray-500 [.dark_&]:text-gray-400"
                    >
                      {{ model.label.slice(0, 2) }}
                    </div>
                  </div>
                  <div class="min-w-0 flex-1">
                    <div class="flex min-w-0 items-center gap-2">
                      <div class="truncate text-sm font-semibold text-gray-900 [.dark_&]:text-white">
                        {{ model.label }}
                      </div>
                      <span
                        v-if="model.family"
                        class="inline-flex shrink-0 items-center rounded-md border border-gray-200 bg-gray-100 px-2 py-0.5 text-[10px] leading-none text-gray-700 [.dark_&]:border-white/10 [.dark_&]:bg-black/40 [.dark_&]:text-gray-100"
                      >
                        {{ model.family }}
                      </span>
                    </div>
                    <div
                      v-if="model.load_name"
                      class="mt-1 truncate text-xs text-gray-500 [.dark_&]:text-gray-400"
                    >
                      {{ model.load_name }}
                    </div>
                    <div class="mt-2 flex flex-wrap gap-1">
                      <span
                        v-for="badge in getBadges(model)"
                        :key="badge"
                        class="inline-flex items-center rounded-md border border-indigo-200/90 bg-indigo-50 px-2 py-0.5 text-[10px] leading-none text-indigo-800 [.dark_&]:border-indigo-500/20 [.dark_&]:bg-indigo-500/15 [.dark_&]:text-indigo-100"
                      >
                        {{ badge }}
                      </span>
                    </div>
                  </div>
                </div>
              </button>
            </div>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import type { UiModelOption } from '../../../composables/useCheckpointModels'
import type { VideoModelProfile } from '../videoTaskProfiles'

const { t } = useI18n()

const props = withDefaults(
  defineProps<{
    modelValue?: string
    models: UiModelOption[]
    loading?: boolean
    taskLabel?: string
    emptyText?: string
    allowClear?: boolean
  }>(),
  {
    modelValue: '',
    loading: false,
    allowClear: true,
  }
)

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
}>()

const showDialog = ref(false)
const keyword = ref('')

const displayTaskLabel = computed(() => props.taskLabel ?? t('video.modelSelector.currentMode'))
const displayEmptyText = computed(() => props.emptyText ?? t('video.modelSelector.noModelsForMode'))

const selectedModel = computed(() => props.models.find((item) => item.value === props.modelValue))
const selectedTitle = computed(() => selectedModel.value?.label || t('video.modelSelector.selectModel'))
const selectedSubtitle = computed(() => selectedModel.value?.load_name || selectedModel.value?.name || '')
const selectedThumb = computed(() => resolveThumb(selectedModel.value?.thumb))
const selectedBadges = computed(() => getBadges(selectedModel.value).slice(0, 4))

const filteredModels = computed(() => {
  const kw = keyword.value.trim().toLowerCase()
  if (!kw) return props.models
  return props.models.filter((model) => {
    const haystack = [model.label, model.name, model.load_name, model.family]
      .map((item) => String(item || '').toLowerCase())
      .join(' ')
    return haystack.includes(kw)
  })
})

function resolveThumb(url?: string): string {
  const raw = String(url || '').trim()
  if (!raw) return ''
  if (/^(https?:)?\/\//i.test(raw) || raw.startsWith('/')) return raw
  return `/outputs/${raw.replace(/^\/+/, '')}`
}

function getBadges(model?: UiModelOption): string[] {
  const profile = model?.video_profile as VideoModelProfile | undefined
  if (!profile) return []
  const taskBadges = Array.isArray(profile.labels) ? profile.labels : []
  const resolutionBadges = Array.isArray(profile.resolution_badges) ? profile.resolution_badges : []
  return [...taskBadges, ...resolutionBadges]
}

function handleSelect(value: string) {
  emit('update:modelValue', value)
  showDialog.value = false
}
</script>
