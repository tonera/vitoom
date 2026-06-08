<template>
  <div class="space-y-4">
    <!-- CFG / Steps / Seed 一行显示 -->
    <div
      class="space-y-2 rounded-xl border border-gray-200 bg-gray-50/80 p-4 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35"
    >
      <div class="grid grid-cols-3 gap-3">
        <div class="space-y-2">
          <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('video.advancedOptions.cfgStrength') }}</label>
          <RangeInput
            :model-value="guidanceScale"
            label="CFG"
            :min="0"
            :max="20"
            :step="0.5"
            :clamp="true"
            :showTooltip="false"
            :round-to-step="true"
            @update:model-value="emit('update:guidanceScale', $event)"
          />
        </div>
        <div class="space-y-2">
          <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('video.advancedOptions.steps') }}</label>
          <RangeInput
            :model-value="numInferenceSteps"
            label="Steps"
            :min="1"
            :max="100"
            :step="1"
            :clamp="true"
            :showTooltip="false"
            :round-to-step="true"
            @update:model-value="emit('update:numInferenceSteps', $event)"
          />
        </div>
        <div class="space-y-2">
          <div class="flex items-center justify-between gap-2">
            <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('video.advancedOptions.seed') }}</label>
            <label
              class="inline-flex cursor-pointer select-none items-center gap-2 text-xs text-gray-600 [.dark_&]:text-gray-300"
            >
              <input
                type="checkbox"
                class="accent-indigo-600"
                :checked="seedMode === 'random'"
                @change="onToggleRandomSeed"
              />
              {{ t('video.advancedOptions.random') }}
            </label>
          </div>
          <RangeInput
            :model-value="seed"
            label="Seed"
            :min="seedMode === 'random' ? -1 : 0"
            :max="4294967295"
            :step="1"
            :clamp="true"
            :showTooltip="false"
            :round-to-step="true"
            :disabled="seedMode === 'random'"
            @update:model-value="
              (v) => {
                // 用户手动调整 seed 时，默认视为“固定 seed”
                emit('update:seedMode', 'custom')
                emit('update:seed', v)
              }
            "
          />
        </div>
      </div>
    </div>

    <!-- Negative prompt -->
    <div
      class="space-y-2 rounded-xl border border-gray-200 bg-gray-50/80 p-4 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35"
    >
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
import { ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import PromptTextarea from '../../../components/PromptTextarea.vue'
import RangeInput from '../../../components/RangeInput.vue'

const { t } = useI18n()

const props = defineProps<{
  negativePrompt: string
  seed: number
  seedMode: 'random' | 'custom'
  numInferenceSteps: number
  guidanceScale: number
}>()

const emit = defineEmits<{
  (e: 'update:negativePrompt', v: string): void
  (e: 'update:seed', v: number): void
  (e: 'update:seedMode', v: 'random' | 'custom'): void
  (e: 'update:numInferenceSteps', v: number): void
  (e: 'update:guidanceScale', v: number): void
}>()

const lastCustomSeed = ref<number>(props.seed >= 0 ? props.seed : 0)

watch(
  () => [props.seedMode, props.seed] as const,
  ([mode, seed]) => {
    if (mode === 'custom' && seed >= 0) lastCustomSeed.value = seed
  },
  { immediate: true }
)

function onToggleRandomSeed(e: Event) {
  const checked = (e.target as HTMLInputElement).checked
  if (checked) {
    // 选中“随机”：后端用 -1 表示随机 seed
    emit('update:seedMode', 'random')
    emit('update:seed', -1)
  } else {
    // 取消“随机”：恢复可编辑，并回填上一次的自定义 seed
    emit('update:seedMode', 'custom')
    emit('update:seed', lastCustomSeed.value)
  }
}
</script>
