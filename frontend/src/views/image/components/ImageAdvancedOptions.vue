<template>
  <div class="space-y-4">
    <div class="rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-2 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35">
      <div class="relative group inline-block ">
        <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">Seed</label>
      </div>
      <div class="flex items-center gap-2">
        <div class="inline-flex rounded-lg overflow-hidden border border-gray-200 bg-white [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-900/40">
          <button
            type="button"
            class="px-3 py-1.5 text-sm transition-colors cursor-pointer"
            :class="seedMode === 'random' ? 'bg-gray-900 text-white [.dark_&]:bg-gray-700/60' : 'text-gray-600 hover:text-gray-950 [.dark_&]:text-gray-300 [.dark_&]:hover:text-white'"
            @click="setSeedMode('random')"
          >
            Random
          </button>
          <button
            type="button"
            class="px-3 py-1.5 text-sm transition-colors cursor-pointer"
            :class="seedMode === 'custom' ? 'bg-gray-900 text-white [.dark_&]:bg-gray-700/60' : 'text-gray-600 hover:text-gray-950 [.dark_&]:text-gray-300 [.dark_&]:hover:text-white'"
            @click="setSeedMode('custom')"
          >
            Custom
          </button>
        </div>

        <input
          class="flex-1 px-3 py-2 rounded-lg bg-white border border-gray-200 text-sm text-gray-950 focus:outline-none focus:ring-2 focus:ring-indigo-500/60 disabled:opacity-50 [.dark_&]:bg-gray-900/60 [.dark_&]:border-gray-700/60 [.dark_&]:text-gray-100"
          type="number"
          :min="1"
          :step="1"
          :disabled="seedMode !== 'custom'"
          :value="seed"
          :placeholder="t('image.seedPlaceholder')"
          @input="onSeedInput"
        />
      </div>
    </div>

    <div class="rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-2 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35">
      <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('image.options') }}</label>
      <div class="rounded-xl border border-gray-200 bg-white divide-y divide-gray-200 overflow-hidden [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-900/30 [.dark_&]:divide-gray-700/50">
        <label class="flex items-center justify-between gap-4 px-3 py-2.5 hover:bg-gray-50 transition-colors cursor-pointer [.dark_&]:hover:bg-white/5">
          <div class="min-w-0">
            <div class="text-sm font-medium text-gray-800 [.dark_&]:text-gray-200">{{ t('image.removeBg') }}</div>
            <div class="text-xs text-gray-500 mt-0.5">{{ t('image.removeBgHelp') }}</div>
          </div>
          <button
            type="button"
            class="shrink-0 relative inline-flex h-6 w-11 items-center rounded-full transition-colors border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/60 [.dark_&]:border-gray-700/60"
            :class="removeBg ? 'bg-indigo-500/70' : 'bg-gray-200 [.dark_&]:bg-gray-900/40'"
            @click.prevent="emit('update:removeBg', !removeBg)"
          >
            <span
              class="inline-block h-5 w-5 transform rounded-full bg-white transition-transform"
              :class="removeBg ? 'translate-x-5' : 'translate-x-1'"
            />
          </button>
        </label>

        <label class="flex items-center justify-between gap-4 px-3 py-2.5 hover:bg-gray-50 transition-colors cursor-pointer [.dark_&]:hover:bg-white/5">
          <div class="min-w-0">
            <div class="text-sm font-medium text-gray-800 [.dark_&]:text-gray-200">{{ t('image.faceEnhance') }}</div>
            <div class="text-xs text-gray-500 mt-0.5">{{ t('image.faceEnhanceHelp') }}</div>
          </div>
          <button
            type="button"
            class="shrink-0 relative inline-flex h-6 w-11 items-center rounded-full transition-colors border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/60 [.dark_&]:border-gray-700/60"
            :class="faceEnhance ? 'bg-indigo-500/70' : 'bg-gray-200 [.dark_&]:bg-gray-900/40'"
            @click.prevent="emit('update:faceEnhance', !faceEnhance)"
          >
            <span
              class="inline-block h-5 w-5 transform rounded-full bg-white transition-transform"
              :class="faceEnhance ? 'translate-x-5' : 'translate-x-1'"
            />
          </button>
        </label>

        <label class="flex items-center justify-between gap-4 px-3 py-2.5 hover:bg-gray-50 transition-colors cursor-pointer [.dark_&]:hover:bg-white/5">
          <div class="min-w-0">
            <div class="text-sm font-medium text-gray-800 [.dark_&]:text-gray-200">{{ t('image.lowVram') }}</div>
            <div class="text-xs text-gray-500 mt-0.5">{{ t('image.lowVramHelp') }}</div>
          </div>
          <button
            type="button"
            class="shrink-0 relative inline-flex h-6 w-11 items-center rounded-full transition-colors border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/60 [.dark_&]:border-gray-700/60"
            :class="lowVram ? 'bg-indigo-500/70' : 'bg-gray-200 [.dark_&]:bg-gray-900/40'"
            @click.prevent="emit('update:lowVram', !lowVram)"
          >
            <span
              class="inline-block h-5 w-5 transform rounded-full bg-white transition-transform"
              :class="lowVram ? 'translate-x-5' : 'translate-x-1'"
            />
          </button>
        </label>

        <label class="flex items-center justify-between gap-4 px-3 py-2.5 hover:bg-gray-50 transition-colors cursor-pointer [.dark_&]:hover:bg-white/5">
          <div class="min-w-0">
            <div class="text-sm font-medium text-gray-800 [.dark_&]:text-gray-200">{{ t('image.fastMode') }}</div>
            <div class="text-xs text-gray-500 mt-0.5">{{ t('image.fastModeHelp') }}</div>
          </div>
          <button
            type="button"
            class="shrink-0 relative inline-flex h-6 w-11 items-center rounded-full transition-colors border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-500/60 [.dark_&]:border-gray-700/60"
            :class="fastModeOn ? 'bg-indigo-500/70' : 'bg-gray-200 [.dark_&]:bg-gray-900/40'"
            @click.prevent="emit('update:fastMode', !fastModeOn)"
          >
            <span
              class="inline-block h-5 w-5 transform rounded-full bg-white transition-transform"
              :class="fastModeOn ? 'translate-x-5' : 'translate-x-1'"
            />
          </button>
        </label>
      </div>
    </div>

    <div class="rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-2 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35">
      <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('image.negativePrompt') }}</label>
      <PromptTextarea
        :model-value="negativePrompt"
        :rows="3"
        :placeholder="t('image.negativePromptPlaceholder')"
        @update:model-value="emit('update:negativePrompt', $event)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import PromptTextarea from '../../../components/PromptTextarea.vue'

type SeedMode = 'random' | 'custom'

const { t } = useI18n()

const props = defineProps<{
  negativePrompt: string
  seed: number
  seedMode: SeedMode
  removeBg: boolean
  faceEnhance: boolean
  lowVram: boolean
  fastMode?: boolean
}>()

const fastModeOn = computed(() => props.fastMode !== false)

const emit = defineEmits<{
  (e: 'update:negativePrompt', v: string): void
  (e: 'update:seed', v: number): void
  (e: 'update:seedMode', v: SeedMode): void
  (e: 'update:removeBg', v: boolean): void
  (e: 'update:faceEnhance', v: boolean): void
  (e: 'update:lowVram', v: boolean): void
  (e: 'update:fastMode', v: boolean): void
}>()

const lastCustomSeed = ref<number>(Math.max(1, Number(props.seed || 1)))

watch(
  () => props.seed,
  (v) => {
    const n = Number(v)
    if (Number.isFinite(n) && n >= 1) lastCustomSeed.value = Math.floor(n)
  }
)

function clampNum(v: number, min: number, max: number) {
  if (!Number.isFinite(v)) return min
  return Math.min(max, Math.max(min, v))
}

function onSeedInput(e: Event) {
  const raw = Number((e.target as HTMLInputElement).value)
  const v = Math.floor(clampNum(raw, 1, 2147483647))
  lastCustomSeed.value = v
  emit('update:seed', v)
}

function setSeedMode(mode: SeedMode) {
  emit('update:seedMode', mode)
  if (mode === 'custom') {
    const v = props.seed && props.seed >= 1 ? props.seed : lastCustomSeed.value
    emit('update:seed', Math.floor(Math.max(1, Number(v || 1))))
  }
  if (mode === 'random') {
    emit('update:seed', -1)
  }
}
</script>
