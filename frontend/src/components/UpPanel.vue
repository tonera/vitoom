<template>
  <div class="w-full relative group" ref="triggerRef">
    <!-- Tooltip (hover/focus only) -->
    <div
      v-if="label"
      class="pointer-events-none absolute -top-7 left-2 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity duration-150"
    >
      <div class="px-2 py-1 rounded-md bg-sky-50 text-sky-700 border border-sky-200 text-xs font-semibold [.dark_&]:bg-sky-500/15 [.dark_&]:text-sky-200 [.dark_&]:border-sky-500/25">
        {{ label }}
      </div>
    </div>
    <button
      type="button"
      class="w-full px-3 py-2 rounded-lg bg-white border border-gray-200 text-sm text-gray-950 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-indigo-500/60 transition-colors cursor-pointer flex items-center justify-between gap-2 [.dark_&]:bg-gray-900/50 [.dark_&]:border-white/10 [.dark_&]:text-gray-100 [.dark_&]:hover:bg-gray-900/70"
      :title="displayText"
      @click.stop="toggle"
    >
      <span class="truncate text-left">{{ displayText }}</span>
      <svg class="w-3.5 h-3.5 shrink-0 text-gray-400" fill="currentColor" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg">
        <path
          d="M233.4 406.6c12.5 12.5 32.8 12.5 45.3 0l192-192c12.5-12.5 12.5-32.8 0-45.3s-32.8-12.5-45.3 0L256 338.7 86.6 169.4c-12.5-12.5-32.8-12.5-45.3 0s-12.5 32.8 0 45.3l192 192z"
        />
      </svg>
    </button>

    <Teleport to="body">
      <div v-if="open" class="fixed inset-0 z-9998" @mousedown="open = false">
        <div
          ref="panelRef"
          class="fixed z-9999 w-[300px] max-w-[calc(100vw-24px)] p-2 rounded-2xl border border-gray-200 bg-white/95 shadow-2xl backdrop-blur ring-1 ring-gray-200/60 [.dark_&]:border-white/10 [.dark_&]:bg-gray-900/95 [.dark_&]:ring-white/5"
          :style="panelStyle"
          role="dialog"
          aria-modal="false"
          @mousedown.stop
        >
          <div class="px-2 py-1.5 flex items-center justify-between gap-2">
            <div class="text-xs font-semibold text-gray-800 [.dark_&]:text-gray-200">{{ t('components.upPanel.selectSampler') }}</div>
            <button
              type="button"
              class="p-1 rounded-md text-gray-500 hover:text-gray-950 hover:bg-gray-100 transition-colors cursor-pointer [.dark_&]:text-gray-400 [.dark_&]:hover:text-white [.dark_&]:hover:bg-gray-800/60"
              :title="t('common.close')"
              @click="open = false"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div class="border-t border-gray-200 [.dark_&]:border-white/10"></div>

          <div class="mt-2 max-h-[50vh] overflow-y-auto vt-scroll">
            <button
              type="button"
              class="w-full text-left px-3 py-2 rounded-xl text-xs hover:bg-white/5 transition-colors cursor-pointer"
              :class="!modelValue ? 'bg-indigo-50 text-indigo-700 border border-indigo-200 [.dark_&]:bg-indigo-500/10 [.dark_&]:text-indigo-100 [.dark_&]:border-indigo-500/20' : 'text-gray-700 [.dark_&]:text-gray-200'"
              @click="select('')"
            >
              {{ t('components.upPanel.defaultFollowModel') }}
            </button>
            <button
              v-for="s in options"
              :key="s"
              type="button"
              class="mt-1 w-full text-left px-3 py-2 rounded-xl text-xs hover:bg-gray-100 transition-colors cursor-pointer [.dark_&]:hover:bg-white/5"
              :class="modelValue === s ? 'bg-indigo-50 text-indigo-700 border border-indigo-200 [.dark_&]:bg-indigo-500/10 [.dark_&]:text-indigo-100 [.dark_&]:border-indigo-500/20' : 'text-gray-700 [.dark_&]:text-gray-200'"
              @click="select(s)"
            >
              {{ s }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = defineProps<{
  modelValue: string
  options: string[]
  label?: string
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: string): void
}>()

const open = ref(false)
const triggerRef = ref<HTMLElement | null>(null)
const panelRef = ref<HTMLElement | null>(null)
const panelStyle = ref<Record<string, string>>({})
let raf = 0

const displayText = computed(() => (props.modelValue ? props.modelValue : t('components.upPanel.defaultFollowModel')))

function toggle() {
  open.value = !open.value
  if (open.value) schedulePosition()
}

function select(v: string) {
  emit('update:modelValue', v)
  open.value = false
}

async function position() {
  await nextTick()
  const btn = triggerRef.value
  const pop = panelRef.value
  if (!btn || !pop || typeof window === 'undefined') return

  const padding = 12
  const gap = 10
  const rect = btn.getBoundingClientRect()
  const popRect = pop.getBoundingClientRect()

  // center align under trigger by default
  let left = rect.left + rect.width / 2 - popRect.width / 2
  left = Math.max(padding, Math.min(left, window.innerWidth - padding - popRect.width))

  // prefer up (like UpPanel); if not enough, place down
  let top = rect.top - gap - popRect.height
  if (top < padding) {
    top = rect.bottom + gap
  }
  top = Math.max(padding, Math.min(top, window.innerHeight - padding - popRect.height))

  panelStyle.value = { left: `${left}px`, top: `${top}px` }
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

watch(open, (v) => {
  if (typeof window === 'undefined') return
  if (v) {
    schedulePosition()
    window.addEventListener('keydown', onKeydown)
    window.addEventListener('resize', schedulePosition)
    window.addEventListener('scroll', schedulePosition, true)
  } else {
    window.removeEventListener('keydown', onKeydown)
    window.removeEventListener('resize', schedulePosition)
    window.removeEventListener('scroll', schedulePosition, true)
  }
})

onBeforeUnmount(() => {
  if (typeof window !== 'undefined') {
    window.removeEventListener('keydown', onKeydown)
    window.removeEventListener('resize', schedulePosition)
    window.removeEventListener('scroll', schedulePosition, true)
  }
  if (raf) cancelAnimationFrame(raf)
})
</script>
