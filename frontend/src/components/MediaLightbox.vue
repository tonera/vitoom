<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="fixed inset-0 z-10000 bg-black/85 backdrop-blur-sm flex items-center justify-center p-6"
      role="dialog"
      aria-modal="true"
      @click.self="emitClose"
    >
      <!-- Close -->
      <button
        type="button"
        class="absolute top-4 right-4 w-10 h-10 rounded-full bg-black/40 hover:bg-black/55 border border-white/10 text-white flex items-center justify-center cursor-pointer"
        :title="t('components.media.closeEsc')"
        @click="emitClose"
      >
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>

      <!-- Prev / Next -->
      <button
        type="button"
        class="absolute left-4 top-1/2 -translate-y-1/2 w-11 h-11 rounded-full bg-black/35 hover:bg-black/50 border border-white/10 text-white flex items-center justify-center cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
        :title="t('components.media.prevItem')"
        :disabled="!canPrev"
        @click.stop="prev"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
        </svg>
      </button>
      <button
        type="button"
        class="absolute right-4 top-1/2 -translate-y-1/2 w-11 h-11 rounded-full bg-black/35 hover:bg-black/50 border border-white/10 text-white flex items-center justify-center cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
        :title="t('components.media.nextItem')"
        :disabled="!canNext"
        @click.stop="next"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
        </svg>
      </button>

      <!-- Content -->
      <div class="w-full h-full flex items-center justify-center">
        <img
          v-if="activeItem?.type === 'image'"
          class="max-w-[95vw] max-h-[90vh] object-contain rounded-lg shadow-2xl"
          :src="activeItem.src"
          :alt="activeItem.title || 'image'"
        />
        <video
          v-else-if="activeItem?.type === 'video'"
          class="max-w-[95vw] max-h-[90vh] rounded-lg shadow-2xl bg-black"
          :src="activeItem.src"
          :poster="activeItem.poster"
          controls
          autoplay
        />
      </div>

      <div
        v-if="activeIndex >= 0"
        class="absolute bottom-4 left-1/2 -translate-x-1/2 text-xs text-gray-200/90 bg-black/30 border border-white/10 px-3 py-1.5 rounded-full"
      >
        {{ activeIndex + 1 }} / {{ items.length }}
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, watch } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

export type MediaLightboxItem = {
  key: string
  type: 'image' | 'video'
  src: string
  title?: string
  poster?: string
}

const props = defineProps<{
  open: boolean
  items: MediaLightboxItem[]
  activeKey: string | null
}>()

const emit = defineEmits<{
  (e: 'update:open', v: boolean): void
  (e: 'update:activeKey', v: string | null): void
}>()

const activeIndex = computed(() => {
  const key = props.activeKey
  if (!key) return -1
  return props.items.findIndex((x) => x.key === key)
})

const activeItem = computed(() => {
  const idx = activeIndex.value
  if (idx < 0) return null
  return props.items[idx] || null
})

const canPrev = computed(() => activeIndex.value > 0)
const canNext = computed(() => {
  const idx = activeIndex.value
  return idx >= 0 && idx < props.items.length - 1
})

function emitClose() {
  emit('update:open', false)
}

function prev() {
  const idx = activeIndex.value
  if (idx <= 0) return
  emit('update:activeKey', props.items[idx - 1]!.key)
}

function next() {
  const idx = activeIndex.value
  if (idx < 0 || idx >= props.items.length - 1) return
  emit('update:activeKey', props.items[idx + 1]!.key)
}

function onKeydown(e: KeyboardEvent) {
  if (!props.open) return
  if (e.key === 'Escape') {
    e.preventDefault()
    emitClose()
    return
  }
  if (e.key === 'ArrowLeft') {
    e.preventDefault()
    prev()
    return
  }
  if (e.key === 'ArrowRight') {
    e.preventDefault()
    next()
  }
}

watch(
  () => props.open,
  (open) => {
    if (typeof document === 'undefined') return
    document.body.style.overflow = open ? 'hidden' : ''
    if (open) {
      // If opening without a valid activeKey, pick first item.
      const idx = activeIndex.value
      if (idx === -1 && props.items.length > 0) emit('update:activeKey', props.items[0]!.key)
    }
  }
)

watch(
  () => props.items,
  () => {
    if (!props.open) return
    if (!props.activeKey) return
    if (activeIndex.value === -1) {
      // If current disappears, close for safety.
      emitClose()
    }
  }
)

onMounted(() => {
  window.addEventListener('keydown', onKeydown)
})
onBeforeUnmount(() => {
  window.removeEventListener('keydown', onKeydown)
  if (typeof document !== 'undefined') document.body.style.overflow = ''
})
</script>


