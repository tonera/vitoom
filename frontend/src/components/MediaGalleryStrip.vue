<template>
  <div class="vt-card p-3 flex-1 flex overflow-hidden min-h-0">
    <!-- 左侧：缩略图队列 + 大图预览 -->
    <div class="flex-1 min-w-0 flex flex-col overflow-hidden">
      <!-- 左上：缩略图队列 -->
      <div class="flex items-center gap-2">
        <button
          type="button"
          class="shrink-0 w-8 h-8 text-gray-500 flex items-center justify-center cursor-pointer rounded-full hover:bg-gray-100 pb-3 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-900/50"
          :title="t('components.media.scrollLeft')"
          @click="scrollThumbs('left')"
        >
          ‹
        </button>

        <div class="flex-1 min-w-0 overflow-hidden">
          <div ref="thumbStripRef" class="thumb-strip flex items-center gap-2 overflow-x-auto overflow-y-hidden py-1 pr-1 h-18" @scroll.passive="onThumbStripScroll">
            <button
              v-for="item in items"
              :key="item.key"
              type="button"
              class="group relative shrink-0 w-14 h-14 rounded-lg overflow-hidden border bg-transparent cursor-pointer transition"
              :class="activeKey === item.key ? 'border-indigo-500 ring-2 ring-indigo-500/60' : 'border-gray-200 hover:border-indigo-500/60 hover:ring-2 hover:ring-indigo-500/30 [.dark_&]:border-gray-700/60'"
              :title="item.title || ''"
              @click="emit('update:activeKey', item.key)"
            >
              <img :src="item.thumbSrc" class="w-full h-full object-cover group-hover:brightness-110 transition" />
              <div
                v-if="item.source === 'generated'"
                class="absolute left-1 top-1 px-1.5 py-0.5 rounded bg-black/60 text-[10px] text-gray-100"
                :title="t('components.media.newBadge')"
              >
                {{ t('components.media.newBadge') }}
              </div>
            </button>

            <div v-if="items.length === 0" class="text-xs text-gray-500 py-2 [.dark_&]:text-gray-400">
              {{ resolvedEmptyQueueText }}
            </div>
          </div>
        </div>

        <button
          type="button"
          class="shrink-0 w-8 h-8 text-gray-500 flex items-center justify-center cursor-pointer rounded-full hover:bg-gray-100 pb-3 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-900/50"
          :title="t('components.media.scrollRight')"
          @click="scrollThumbs('right')"
        >
          ›
        </button>
      </div>

      <!-- 左下：大图预览 -->
      <div class="mt-3 flex-1 min-h-0 overflow-hidden">
        <div
          class="relative w-full h-full rounded-xl overflow-hidden border border-gray-200/90 bg-transparent [.dark_&]:border-gray-700/60"
        >
          <div v-if="activeItem" class="w-full h-full flex items-center justify-center">
            <img
              v-if="activeItem.kind === 'image'"
              :src="activeItem.originalSrc"
              class="h-full object-contain cursor-zoom-in"
              :alt="activeItem.title || 'preview'"
              @click="emit('open', activeItem.key)"
            />
            <video v-else class="w-full h-full" controls :src="activeItem.originalSrc" :poster="activeItem.posterSrc || activeItem.thumbSrc" />

            <!-- Overlay actions -->
            <div class="absolute bottom-4 right-4 flex items-center gap-2">
              <button
                v-if="showUpload"
                type="button"
                class="w-10 h-10 rounded-full bg-black/40 hover:bg-black/60 text-white flex items-center justify-center cursor-pointer transition-all backdrop-blur-sm border border-white/20 hover:border-white/30"
                :title="t('components.media.uploadImage')"
                @click="emit('upload')"
              >
                <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </button>

              <button
                v-if="showDownload"
                type="button"
                class="w-10 h-10 rounded-full bg-black/40 hover:bg-black/60 text-white flex items-center justify-center cursor-pointer transition-all backdrop-blur-sm border border-white/20 hover:border-white/30"
                :title="t('components.media.download')"
                @click="emit('download')"
              >
                <svg xmlns="http://www.w3.org/2000/svg" class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2">
                  <path stroke-linecap="round" stroke-linejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
              </button>
            </div>
          </div>

          <div v-else class="w-full h-full flex items-center justify-center">
            <div class="text-center">
              <div
                class="w-10 h-10 mx-auto rounded-full border-2 border-gray-300 border-t-indigo-500 animate-spin [.dark_&]:border-gray-600"
              ></div>
              <div class="mt-2 text-xs text-gray-500 [.dark_&]:text-gray-400">
                {{ loading ? resolvedLoadingText : (errorText ? errorText : resolvedEmptyPreviewText) }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Optional side panel (e.g. quick buttons) -->
    <div v-if="$slots.side" class="w-72 shrink-0 pl-4 border-l border-gray-200 [.dark_&]:border-gray-800/60">
      <slot name="side" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

export type MediaStripItem = {
  key: string
  kind: 'image' | 'video'
  thumbSrc: string
  originalSrc: string
  posterSrc?: string
  title?: string
  source?: 'generated' | 'userfile' | string
}

const props = withDefaults(
  defineProps<{
    items: MediaStripItem[]
    activeKey: string | null
    activeItem?: MediaStripItem | null
    loading?: boolean
    errorText?: string
    emptyQueueText?: string
    emptyPreviewText?: string
    loadingText?: string
    showUpload?: boolean
    showDownload?: boolean
    canLoadMore?: boolean
    loadMoreThresholdPx?: number
  }>(),
  {
    loading: false,
    errorText: '',
    emptyQueueText: '',
    emptyPreviewText: '',
    loadingText: '',
    showUpload: false,
    showDownload: true,
    canLoadMore: true,
    loadMoreThresholdPx: 64,
  }
)

const emit = defineEmits<{
  (e: 'update:activeKey', v: string | null): void
  (e: 'open', key: string): void
  (e: 'upload'): void
  (e: 'download'): void
  (e: 'load-more'): void
}>()

const resolvedEmptyQueueText = computed(() => props.emptyQueueText || t('components.media.emptyQueue'))
const resolvedEmptyPreviewText = computed(() => props.emptyPreviewText || t('components.media.emptyPreview'))
const resolvedLoadingText = computed(() => props.loadingText || t('components.media.loadingGallery'))

const thumbStripRef = ref<HTMLElement | null>(null)

const activeItem = computed(() => {
  if (props.activeItem !== undefined) return props.activeItem
  if (!props.items.length) return null
  const byKey = props.activeKey ? props.items.find((x) => x.key === props.activeKey) : undefined
  return byKey || props.items[0]!
})

function scrollThumbs(dir: 'left' | 'right') {
  const el = thumbStripRef.value
  if (!el) return
  const delta = dir === 'left' ? -280 : 280
  el.scrollBy({ left: delta, behavior: 'smooth' })
}

function onThumbStripScroll() {
  if (!props.canLoadMore) return
  const el = thumbStripRef.value
  if (!el) return
  const thresholdPx = Number(props.loadMoreThresholdPx) || 64
  const remain = el.scrollWidth - el.scrollLeft - el.clientWidth
  if (remain <= thresholdPx) emit('load-more')
}
</script>

<style scoped>
.thumb-strip {
  scrollbar-width: thin; /* Firefox */
  scrollbar-color: rgba(99, 102, 241, 0.55) transparent;
}
.thumb-strip::-webkit-scrollbar {
  height: 4px; /* 横向滚动条变细，避免撑高导致不对齐 */
}
.thumb-strip::-webkit-scrollbar-track {
  background: transparent;
}
.thumb-strip::-webkit-scrollbar-thumb {
  background: rgba(99, 102, 241, 0.55);
  border-radius: 999px;
}
.thumb-strip::-webkit-scrollbar-thumb:hover {
  background: rgba(99, 102, 241, 0.75);
}
</style>

