<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="fixed inset-0 z-9999 bg-black/60 flex items-center justify-center p-4"
      @click.self="emitClose()"
    >
      <div class="vt-card w-full max-w-[720px] max-h-[min(82vh,900px)] overflow-y-auto vt-scroll p-5 rounded-2xl">
        <div class="flex items-start justify-between gap-4">
          <div class="min-w-0">
            <div class="text-lg font-semibold text-white truncate">{{ t('components.advancedConfig.title') }}</div>
            <div class="text-xs text-gray-400 mt-1 truncate">
              {{ headerDesc }}
            </div>
          </div>
          <button
            type="button"
            class="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800/60 transition-colors cursor-pointer"
            @click="emitClose()"
            :aria-label="t('common.close')"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div class="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="space-y-1 md:col-span-2">
            <div class="text-xs text-gray-400">{{ t('components.advancedConfig.vae') }}</div>
            <input
              v-model="vae"
              type="text"
              :placeholder="t('components.advancedConfig.vaePlaceholder')"
              class="w-full px-3 py-2 bg-gray-800/60 border border-gray-700/70 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/80"
            />
          </div>

          <div class="space-y-1">
            <div class="text-xs text-gray-400">{{ t('components.advancedConfig.scheduler') }}</div>
            <select
              v-model="schedulerName"
              class="w-full px-3 py-2 rounded-xl bg-gray-800/60 border border-gray-700/70 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-indigo-500/80"
            >
              <option value="">{{ t('components.advancedConfig.defaultOption') }}</option>
              <option v-for="s in samplerOptions" :key="`sam-${s}`" :value="s">{{ s }}</option>
            </select>
          </div>

          <div class="space-y-1">
            <div class="text-xs text-gray-400">{{ t('components.advancedConfig.guidanceScale') }}</div>
            <input
              v-model="guidanceScaleText"
              type="number"
              min="0"
              max="20"
              step="0.1"
              :placeholder="t('components.advancedConfig.defaultOption')"
              class="w-full px-3 py-2 bg-gray-800/60 border border-gray-700/70 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/80"
            />
            <div v-if="errors.guidance_scale" class="text-xs text-rose-300">{{ errors.guidance_scale }}</div>
          </div>

          <div class="space-y-1">
            <div class="text-xs text-gray-400">{{ t('components.advancedConfig.steps') }}</div>
            <input
              v-model="numStepsText"
              type="number"
              min="1"
              max="50"
              step="1"
              :placeholder="t('components.advancedConfig.defaultOption')"
              class="w-full px-3 py-2 bg-gray-800/60 border border-gray-700/70 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-indigo-500/80"
            />
            <div v-if="errors.num_inference_steps" class="text-xs text-rose-300">{{ errors.num_inference_steps }}</div>
          </div>

          <div class="md:col-span-2">
            <div class="flex items-center justify-between gap-3">
              <div class="text-xs text-gray-400">{{ t('components.advancedConfig.preview') }}</div>
              <button
                type="button"
                class="text-xs text-gray-400 hover:text-white cursor-pointer"
                @click="resetToInitial()"
              >
                {{ t('components.advancedConfig.resetToCurrent') }}
              </button>
            </div>
            <pre class="mt-2 text-xs font-mono p-3 rounded-xl bg-gray-900/40 border border-gray-700/70 text-gray-100 overflow-auto">{{ previewText }}</pre>
          </div>
        </div>

        <div class="mt-5 flex items-center justify-end gap-2">
          <button
            type="button"
            class="px-4 py-2 rounded-xl bg-gray-800/50 border border-gray-700/70 text-gray-200 hover:bg-gray-800/80 transition-colors cursor-pointer"
            @click="emitClose()"
          >
            {{ t('common.cancel') }}
          </button>
          <button
            type="button"
            class="px-4 py-2 rounded-xl bg-indigo-600 text-white hover:bg-indigo-500 transition-colors cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
            :disabled="saving"
            @click="submit()"
          >
            {{ saving ? t('components.advancedConfig.saving') : t('common.save') }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, reactive, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { samplerOptions } from '../config/samplers'

const { t } = useI18n()

type AnyCfg = Record<string, any>

const props = defineProps<{
  open: boolean
  saving?: boolean
  modelName?: string
  family?: string | null
  initialConfig?: any
}>()

const emit = defineEmits<{
  (e: 'update:open', v: boolean): void
  (e: 'submit', modelConfig: Record<string, any>): void
}>()

function safeParseConfig(input: any): AnyCfg {
  if (!input) return {}
  if (typeof input === 'object') return (input || {}) as AnyCfg
  if (typeof input === 'string') {
    try {
      const v = JSON.parse(input)
      if (v && typeof v === 'object') return v as AnyCfg
      return {}
    } catch {
      return {}
    }
  }
  return {}
}

const headerDesc = computed(() => {
  const name = String(props.modelName || '').trim()
  const mc = String(props.family || '').trim()
  const segs: string[] = []
  if (name) segs.push(name)
  if (mc) segs.push(t('components.advancedConfig.familyLabel', { family: mc }))
  return segs.length ? segs.join(' · ') : t('components.advancedConfig.editJson')
})

const vae = ref<string>('')
const schedulerName = ref<string>('')
const guidanceScaleText = ref<string | number>('')
const numStepsText = ref<string | number>('')

const errors = reactive<{ guidance_scale?: string; num_inference_steps?: string }>({})

function resetToInitial() {
  const cfg = safeParseConfig(props.initialConfig)

  vae.value = String((cfg as any).vae || '').trim()
  schedulerName.value = typeof (cfg as any).schedulerName === 'string' ? String((cfg as any).schedulerName) : ''

  if (Object.prototype.hasOwnProperty.call(cfg, 'guidance_scale')) {
    const v = Number((cfg as any).guidance_scale)
    guidanceScaleText.value = Number.isFinite(v) ? String(v) : ''
  } else {
    guidanceScaleText.value = ''
  }

  if (Object.prototype.hasOwnProperty.call(cfg, 'num_inference_steps')) {
    const v = Number((cfg as any).num_inference_steps)
    numStepsText.value = Number.isFinite(v) ? String(Math.floor(v)) : ''
  } else {
    numStepsText.value = ''
  }

  errors.guidance_scale = undefined
  errors.num_inference_steps = undefined
}

watch(
  () => [props.open, props.initialConfig] as const,
  ([open]) => {
    if (open) resetToInitial()
  }
)

function buildConfig(): Record<string, any> {
  const out: AnyCfg = { ...safeParseConfig(props.initialConfig) }

  const vvae = String(vae.value || '').trim()
  if (vvae) out.vae = vvae
  else delete out.vae

  const sch = String(schedulerName.value || '').trim()
  if (sch) out.schedulerName = sch
  else delete out.schedulerName

  const gsText = String(guidanceScaleText.value ?? '').trim()
  if (gsText) {
    out.guidance_scale = Number(gsText)
  } else {
    delete out.guidance_scale
  }

  const nsText = String(numStepsText.value ?? '').trim()
  if (nsText) {
    out.num_inference_steps = Number(nsText)
  } else {
    delete out.num_inference_steps
  }

  return out
}

const previewText = computed(() => {
  try {
    return JSON.stringify(buildConfig(), null, 2)
  } catch {
    return '{\n  "error": "failed to stringify"\n}'
  }
})

function validate(): boolean {
  errors.guidance_scale = undefined
  errors.num_inference_steps = undefined

  const gsText = String(guidanceScaleText.value ?? '').trim()
  if (gsText) {
    const gs = Number(gsText)
    if (!Number.isFinite(gs) || gs < 0 || gs > 20) {
      errors.guidance_scale = t('components.advancedConfig.guidanceScaleError')
    }
  }

  const nsRaw = String(numStepsText.value ?? '').trim()
  if (nsRaw) {
    const ns = Number(nsRaw)
    const isInt = Number.isFinite(ns) && Number.isInteger(ns)
    if (!isInt || ns < 1 || ns > 50) {
      errors.num_inference_steps = t('components.advancedConfig.stepsError')
    }
  }

  return !errors.guidance_scale && !errors.num_inference_steps
}

function emitClose() {
  if (props.saving) return
  emit('update:open', false)
}

function submit() {
  if (props.saving) return
  if (!validate()) return

  const cfg = buildConfig()
  if (Object.prototype.hasOwnProperty.call(cfg, 'guidance_scale')) {
    cfg.guidance_scale = Math.round(Number(cfg.guidance_scale) * 1000) / 1000
  }
  if (Object.prototype.hasOwnProperty.call(cfg, 'num_inference_steps')) {
    cfg.num_inference_steps = Math.floor(Number(cfg.num_inference_steps))
  }

  emit('submit', cfg)
}
</script>
