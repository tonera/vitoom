<template>
  <div
    class="relative overflow-hidden rounded-xl border border-gray-700/60 bg-gray-800/40 aspect-2/3 group"
  >
    <!-- Image -->
    <button
      type="button"
      class="absolute inset-0 w-full h-full cursor-pointer"
      :title="title || t('components.media.viewLarge')"
      @click="requestOpenLightbox"
    >
      <img
        v-if="displayThumb"
        class="w-full h-full object-cover transition-transform duration-300 ease-out group-hover:scale-110"
        :src="displayThumb"
        :alt="title || (isVideo ? 'generated video' : 'generated image')"
        loading="lazy"
      />
      <div
        v-else
        class="w-full h-full bg-gray-900/30 flex items-center justify-center"
        aria-hidden="true"
      >
        <svg class="w-10 h-10 text-white/30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M8 12h8M12 8v8" />
        </svg>
      </div>
      <!-- Video play overlay -->
      <div
        v-if="isVideo"
        class="absolute inset-0 flex items-center justify-center"
        aria-hidden="true"
      >
        <div class="w-12 h-12 rounded-full bg-black/35 border border-white/15 backdrop-blur-sm flex items-center justify-center">
          <svg class="w-6 h-6 text-white" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z" />
          </svg>
        </div>
      </div>
      <div class="absolute inset-0 bg-black/0 group-hover:bg-black/15 transition-colors"></div>
    </button>

    <!-- Bottom-right actions -->
    <div class="absolute bottom-2 right-2 flex items-center gap-2 z-10">
      <button
        type="button"
        class="w-9 h-9 flex items-center justify-center text-white/55 hover:text-white cursor-pointer focus:outline-none transition-colors"
        :title="t('components.media.download')"
        @click.stop="download"
      >
        <!-- download icon -->
        <svg class="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
        </svg>
      </button>

      <button
        ref="infoBtnRef"
        type="button"
        class="w-9 h-9 flex items-center justify-center text-white/55 hover:text-white cursor-pointer focus:outline-none transition-colors"
        :title="t('components.media.moreInfo')"
        @click.stop="toggleInfo"
      >
        <!-- info icon -->
        <svg class="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
            d="M12 18h.01M12 10v5m0-11a9 9 0 110 18 9 9 0 010-18z" />
        </svg>
      </button>
    </div>

    <!-- Info popover (teleport to body; avoid being clipped by card's overflow-hidden) -->
    <Teleport to="body">
      <div
        v-if="showInfo"
        class="fixed inset-0 z-9998 bg-black/20"
        @click="showInfo = false"
      >
        <div
          ref="infoPopoverRef"
          class="fixed w-[400px] max-w-[calc(100vw-24px)] max-h-[min(70vh,560px)] overflow-y-auto vt-scroll p-4 rounded-2xl border border-white/10 bg-gray-900/95 shadow-2xl backdrop-blur ring-1 ring-white/5"
          :style="infoPopoverStyle"
          role="dialog"
          aria-modal="false"
          @click.stop
        >
          <div class="text-gray-100">
            <div class="flex items-start justify-between gap-3">
              <div class="min-w-0">
                <div class="flex flex-wrap items-center gap-2">
                  <div class="text-base font-semibold text-gray-100">Prompt</div>
                  <span
                    v-for="(b, i) in headerBadges"
                    :key="i"
                    class="text-[10px] font-semibold tracking-wide px-2 py-0.5 rounded-md bg-sky-900/35 border border-sky-500/15 text-sky-200"
                  >
                    {{ b }}
                  </span>
                </div>
                <div class="mt-2 text-sm leading-relaxed text-gray-300">
                  <div :class="promptExpanded ? 'whitespace-pre-wrap wrap-break-word' : 'max-h-24 overflow-hidden whitespace-pre-wrap wrap-break-word'">
                    {{ details?.prompt || '-' }}
                  </div>
                  <button
                    v-if="(details?.prompt || '').length > 120"
                    type="button"
                    class="mt-1 text-xs text-sky-200 hover:text-sky-100 underline underline-offset-2 cursor-pointer"
                    @click="promptExpanded = !promptExpanded"
                  >
                    {{ promptExpanded ? t('components.media.showLess') : t('components.media.showMore') }}
                  </button>
                </div>
              </div>

              <div class="shrink-0 flex items-center gap-2">
                <button
                  type="button"
                  class="w-8 h-8 rounded-lg border border-white/10 bg-black/20 hover:bg-black/30 text-white flex items-center justify-center cursor-pointer"
                  :title="copiedPrompt ? t('common.copied') : t('components.media.copyPrompt')"
                  @click="copyPrompt"
                >
                  <svg
                    v-if="copiedPrompt"
                    class="w-4 h-4 text-emerald-400"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                  </svg>
                  <svg
                    v-else
                    class="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M8 7V6a2 2 0 012-2h7a2 2 0 012 2v9a2 2 0 01-2 2h-1M8 7h7a2 2 0 012 2v7a2 2 0 01-2 2H10a2 2 0 01-2-2V7z"
                    />
                  </svg>
                </button>
                <button
                  type="button"
                  class="w-8 h-8 rounded-lg border border-white/10 bg-black/20 hover:bg-black/30 text-white flex items-center justify-center cursor-pointer"
                  :title="t('common.close')"
                  @click="showInfo = false"
                >
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            <div class="mt-4 border-t border-white/10"></div>

            <div v-if="hasNegative" class="mt-4">
              <div class="text-sm font-semibold text-gray-200">Negative prompt</div>
              <div class="mt-2 text-sm leading-relaxed text-gray-300">
                <div :class="negativeExpanded ? 'whitespace-pre-wrap wrap-break-word' : 'max-h-20 overflow-hidden whitespace-pre-wrap wrap-break-word'">
                  {{ details?.negativePrompt || '-' }}
                </div>
                <button
                  v-if="(details?.negativePrompt || '').length > 120"
                  type="button"
                  class="mt-1 text-xs text-sky-200 hover:text-sky-100 underline underline-offset-2 cursor-pointer"
                  @click="negativeExpanded = !negativeExpanded"
                >
                  {{ negativeExpanded ? t('components.media.showLess') : t('components.media.showMore') }}
                </button>
              </div>
            </div>

            <div v-if="hasNegative" class="mt-4 border-t border-white/10"></div>

            <div class="mt-4">
              <div class="text-sm font-semibold text-gray-200">Other metadata</div>
              <div class="mt-3 flex flex-wrap gap-2">
                <span
                  v-for="(chip, i) in metaChips"
                  :key="i"
                  class="inline-flex items-center gap-1 text-[11px] px-2 py-1 rounded-md bg-sky-900/35 border border-sky-500/15 text-sky-50"
                >
                  <span class="text-sky-200/80 font-semibold tracking-wide">{{ chip.k }}:</span>
                  <span class="font-semibold">{{ chip.v }}</span>
                </span>
              </div>
            </div>

            <div class="mt-4 border-t border-white/10"></div>

            <div class="mt-3 flex gap-2">
              <button
                type="button"
                class="flex-1 py-2.5 px-3 rounded-lg bg-sky-600/90 hover:bg-sky-600 text-white text-sm font-semibold cursor-pointer"
              @click="requestOpenLightbox"
              >
                {{ t('components.media.viewOriginal') }}
              </button>
              <button
                type="button"
                class="flex-1 py-2.5 px-3 rounded-lg bg-gray-800/80 hover:bg-gray-800 text-white text-sm font-semibold cursor-pointer"
                @click="download"
              >
                {{ t('components.media.downloadOriginal') }}
              </button>
            </div>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { downloadFile } from '../utils/upload'

const { t } = useI18n()

export interface GeneratedImageDetails {
  prompt?: string
  negativePrompt?: string
  cfgScale?: number | string
  steps?: number | string
  sampler?: string
  seed?: number | string
  modelName?: string
  family?: string
  width?: number | string
  height?: number | string
  taskId?: string
  fileName?: string
}

const props = defineProps<{
  thumbSrc: string
  originalSrc?: string
  title?: string
  downloadName?: string
  details?: GeneratedImageDetails
  badges?: string[]
  mediaType?: 'image' | 'video'
  posterSrc?: string
}>()

const emit = defineEmits<{
  (e: 'open'): void
}>()

const showInfo = ref(false)
const promptExpanded = ref(false)
const negativeExpanded = ref(false)

const isVideo = computed(() => props.mediaType === 'video')
const displayThumb = computed(() => props.posterSrc || props.thumbSrc)

const infoBtnRef = ref<HTMLElement | null>(null)
const infoPopoverRef = ref<HTMLElement | null>(null)
const infoPopoverStyle = ref<Record<string, string>>({})
let infoPopoverRaf = 0

const headerBadges = computed(() => props.badges ?? [])
const hasNegative = computed(() => Boolean((props.details?.negativePrompt || '').trim()))

const copiedPrompt = ref(false)
let copiedPromptTimer = 0

const metaChips = computed(() => {
  const d = props.details || {}
  const chips: Array<{ k: string; v: string }> = []
  if (d.cfgScale !== undefined) chips.push({ k: 'CFGSCALE', v: String(d.cfgScale) })
  if (d.steps !== undefined) chips.push({ k: 'STEPS', v: String(d.steps) })
  if (d.sampler) chips.push({ k: 'SAMPLER', v: String(d.sampler) })
  if (d.seed !== undefined) chips.push({ k: 'SEED', v: String(d.seed) })
  if (d.modelName) chips.push({ k: 'MODEL', v: String(d.modelName) })
  if (d.family) chips.push({ k: 'VERSION', v: String(d.family) })
  if (d.width !== undefined && d.height !== undefined) chips.push({ k: 'SIZE', v: `${d.width}×${d.height}` })
  // if (d.taskId) chips.push({ k: 'TASK', v: String(d.taskId) })
  // if (d.fileName) chips.push({ k: 'FILE', v: String(d.fileName) })
  return chips
})

async function copyText(text: string) {
  if (typeof document === 'undefined') return
  if (!text) return
  try {
    await navigator.clipboard.writeText(text)
    return
  } catch {
    // fallback below
  }

  const ta = document.createElement('textarea')
  ta.value = text
  ta.setAttribute('readonly', '')
  ta.style.position = 'fixed'
  ta.style.top = '0'
  ta.style.left = '0'
  ta.style.opacity = '0'
  document.body.appendChild(ta)
  ta.select()
  try {
    document.execCommand('copy')
  } finally {
    document.body.removeChild(ta)
  }
}

async function copyPrompt() {
  const text = props.details?.prompt || ''
  if (!text.trim()) return
  await copyText(text)
  copiedPrompt.value = true
  if (copiedPromptTimer) window.clearTimeout(copiedPromptTimer)
  copiedPromptTimer = window.setTimeout(() => (copiedPrompt.value = false), 1200)
}

async function positionInfoPopover() {
  if (typeof window === 'undefined') return
  await nextTick()
  const btn = infoBtnRef.value
  const pop = infoPopoverRef.value
  if (!btn || !pop) return

  const padding = 12
  const gap = 10
  const btnRect = btn.getBoundingClientRect()
  const popRect = pop.getBoundingClientRect()

  let left = btnRect.right - popRect.width
  left = Math.max(padding, Math.min(left, window.innerWidth - padding - popRect.width))

  let top = btnRect.bottom + gap
  if (top + popRect.height > window.innerHeight - padding) {
    top = btnRect.top - gap - popRect.height
  }
  top = Math.max(padding, Math.min(top, window.innerHeight - padding - popRect.height))

  infoPopoverStyle.value = { left: `${left}px`, top: `${top}px` }
}

function schedulePositionInfoPopover() {
  if (typeof window === 'undefined') return
  if (infoPopoverRaf) cancelAnimationFrame(infoPopoverRaf)
  infoPopoverRaf = window.requestAnimationFrame(() => {
    infoPopoverRaf = 0
    void positionInfoPopover()
  })
}

function toggleInfo() {
  showInfo.value = !showInfo.value
  if (showInfo.value) schedulePositionInfoPopover()
}

function requestOpenLightbox() {
  showInfo.value = false
  emit('open')
}

async function download() {
  const url = props.originalSrc || props.thumbSrc
  if (!url) return
  
  await downloadFile({
    url,
    filename: props.downloadName,
    mediaType: props.mediaType,
    title: props.title,
  })
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') {
    if (showInfo.value) showInfo.value = false
  }
}

watch(showInfo, (open) => {
  if (typeof window === 'undefined') return
  if (open) {
    schedulePositionInfoPopover()
    window.addEventListener('keydown', onKeydown)
    window.addEventListener('resize', schedulePositionInfoPopover)
    window.addEventListener('scroll', schedulePositionInfoPopover, true)
  } else {
    window.removeEventListener('keydown', onKeydown)
    window.removeEventListener('resize', schedulePositionInfoPopover)
    window.removeEventListener('scroll', schedulePositionInfoPopover, true)
  }
})

onBeforeUnmount(() => {
  if (typeof window !== 'undefined') {
    window.removeEventListener('keydown', onKeydown)
    window.removeEventListener('resize', schedulePositionInfoPopover)
    window.removeEventListener('scroll', schedulePositionInfoPopover, true)
  }
  if (infoPopoverRaf) cancelAnimationFrame(infoPopoverRaf)
  if (copiedPromptTimer) window.clearTimeout(copiedPromptTimer)
})
</script>


