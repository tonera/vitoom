<template>
  <div class="w-full relative group " ref="containerRef">
    <!-- Tooltip (hover/focus only) -->
    <div
      v-if="label && showTooltip"
      class="pointer-events-none absolute -top-7 left-2 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity duration-150"
    >
      <div class="px-2 py-1 rounded-md bg-sky-50 text-sky-700 border border-sky-200 text-xs font-semibold [.dark_&]:bg-sky-500/15 [.dark_&]:text-sky-200 [.dark_&]:border-sky-500/25">
        {{ label }}
      </div>
    </div>

    <div class="relative">
      <input
        class="w-full px-3 py-2 pr-8 rounded-lg bg-white border border-gray-200 text-sm text-gray-950 focus:outline-none focus:ring-2 focus:ring-indigo-500/60 vt-no-spinner disabled:opacity-60 disabled:cursor-not-allowed [.dark_&]:bg-gray-900/50 [.dark_&]:border-white/10 [.dark_&]:text-gray-100"
        type="number"
        :min="min"
        :max="max"
        :step="step"
        :value="displayNumber"
        :disabled="disabled"
        @change="onTextChange"
      />

      <button
        ref="iconBtnRef"
        type="button"
        class="absolute right-1 top-1/2 -translate-y-1/2 p-1.5 rounded-md text-gray-500 hover:text-gray-950 hover:bg-gray-100 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-gray-400 [.dark_&]:text-gray-400 [.dark_&]:hover:text-white [.dark_&]:hover:bg-gray-800/60"
        :title="t('components.rangeInput.sliderAdjust')"
        :disabled="disabled"
        @click.stop="toggle"
      >
        <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">
          <path
            d="M233.4 406.6c12.5 12.5 32.8 12.5 45.3 0l192-192c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0L256 338.7 86.6 169.4c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3l192 192z"
          />
        </svg>
      </button>
    </div>

    <Teleport to="body">
      <div
        v-if="open"
        ref="popoverRef"
        class="fixed z-9999 p-2 rounded-xl border border-gray-200 bg-white/95 shadow-2xl backdrop-blur ring-1 ring-gray-200/60 [.dark_&]:border-white/10 [.dark_&]:bg-gray-900/95 [.dark_&]:ring-white/5"
        :style="popoverStyle"
        role="dialog"
        aria-modal="false"
        @mousedown.stop
      >
        <input
          type="range"
          class="w-32 cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
          :min="min"
          :max="max"
          :step="step"
          :value="displayNumber"
          :disabled="disabled"
          @input="onRangeInput"
          @change="onRangeInput"
        />
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = withDefaults(
  defineProps<{
    modelValue: number
    min: number
    max: number
    step?: number
    label?: string
    clamp?: boolean
    roundToStep?: boolean
    showTooltip?: boolean
    disabled?: boolean
  }>(),
  {
    showTooltip: true,
    disabled: false
  }
)

const emit = defineEmits<{
  (e: 'update:modelValue', v: number): void
}>()

const open = ref(false)
const containerRef = ref<HTMLElement | null>(null)
const iconBtnRef = ref<HTMLElement | null>(null)
const popoverRef = ref<HTMLElement | null>(null)
const popoverStyle = ref<Record<string, string>>({})
let raf = 0

const step = computed(() => (props.step === undefined ? 1 : Number(props.step)))
const displayNumber = computed(() => {
  const n = Number(props.modelValue)
  return Number.isFinite(n) ? n : props.min
})

function clampNum(v: number) {
  if (!props.clamp) return v
  return Math.min(props.max, Math.max(props.min, v))
}

function normalize(v: number) {
  let out = v
  out = clampNum(out)
  if (props.roundToStep && step.value > 0 && Number.isFinite(step.value)) {
    const s = step.value
    out = Math.round(out / s) * s
  }
  return out
}

function commit(v: number) {
  const n = normalize(v)
  emit('update:modelValue', n)
}

function onTextChange(e: Event) {
  const raw = Number((e.target as HTMLInputElement).value)
  if (!Number.isFinite(raw)) return
  commit(raw)
}

function onRangeInput(e: Event) {
  const raw = Number((e.target as HTMLInputElement).value)
  if (!Number.isFinite(raw)) return
  commit(raw)
}

function toggle() {
  if (props.disabled) return
  open.value = !open.value
  if (open.value) schedulePosition()
}

async function position() {
  await nextTick()
  const anchor = iconBtnRef.value || containerRef.value
  const pop = popoverRef.value
  if (!anchor || !pop || typeof window === 'undefined') return

  const padding = 12
  const gap = 8
  const rect = anchor.getBoundingClientRect()
  const popRect = pop.getBoundingClientRect()

  let left = rect.right - popRect.width
  left = Math.max(padding, Math.min(left, window.innerWidth - padding - popRect.width))

  let top = rect.bottom + gap
  if (top + popRect.height > window.innerHeight - padding) {
    top = rect.top - gap - popRect.height
  }
  top = Math.max(padding, Math.min(top, window.innerHeight - padding - popRect.height))

  popoverStyle.value = { left: `${left}px`, top: `${top}px` }
}

function schedulePosition() {
  if (typeof window === 'undefined') return
  if (raf) cancelAnimationFrame(raf)
  raf = window.requestAnimationFrame(() => {
    raf = 0
    void position()
  })
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape' && open.value) open.value = false
}

function onDocumentMouseDown(e: MouseEvent) {
  if (!open.value) return
  const el = containerRef.value
  const pop = popoverRef.value
  const t = e.target as Node | null
  if (!t) return
  if (el && el.contains(t)) return
  if (pop && pop.contains(t)) return
  open.value = false
}

watch(open, (v) => {
  if (typeof window === 'undefined') return
  if (v) {
    schedulePosition()
    window.addEventListener('keydown', onKeydown)
    window.addEventListener('resize', schedulePosition)
    window.addEventListener('scroll', schedulePosition, true)
    document.addEventListener('mousedown', onDocumentMouseDown, true)
  } else {
    window.removeEventListener('keydown', onKeydown)
    window.removeEventListener('resize', schedulePosition)
    window.removeEventListener('scroll', schedulePosition, true)
    document.removeEventListener('mousedown', onDocumentMouseDown, true)
  }
})

watch(
  () => props.disabled,
  (v) => {
    if (v) open.value = false
  }
)

onBeforeUnmount(() => {
  if (typeof window !== 'undefined') {
    window.removeEventListener('keydown', onKeydown)
    window.removeEventListener('resize', schedulePosition)
    window.removeEventListener('scroll', schedulePosition, true)
  }
  if (typeof document !== 'undefined') {
    document.removeEventListener('mousedown', onDocumentMouseDown, true)
  }
  if (raf) cancelAnimationFrame(raf)
})
</script>

<style scoped>
/* hide number spinners for a more TSRangeInput-like look */
.vt-no-spinner::-webkit-outer-spin-button,
.vt-no-spinner::-webkit-inner-spin-button {
  -webkit-appearance: none;
  margin: 0;
}
.vt-no-spinner[type="number"] {
  -moz-appearance: textfield;
  appearance: textfield;
}
</style>
