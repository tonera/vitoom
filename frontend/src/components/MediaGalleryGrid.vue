<template>
  <div class="vt-card p-3 flex-1 flex flex-col overflow-hidden">
    <div class="flex-1 overflow-y-auto vt-scroll pr-1">
      <div class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 2xl:grid-cols-5">
        <template v-for="item in items" :key="item.key">
          <GeneratedImageCard
            v-if="item.kind === 'image' || item.kind === 'video'"
            :thumb-src="item.thumbSrc"
            :original-src="item.originalSrc || item.thumbSrc"
            :title="item.title"
            :download-name="item.downloadName"
            :details="item.details"
            :media-type="item.kind"
            :poster-src="item.kind === 'video' ? item.posterSrc : undefined"
            @open="emit('open', item.key)"
          />
          <div v-else class="relative overflow-hidden rounded-xl border border-gray-700/60 bg-gray-900/30 aspect-2/3">
            <div class="w-full h-full flex items-center justify-center">
              <div class="text-center">
                <div class="w-10 h-10 mx-auto rounded-full border-2 border-gray-600 border-t-indigo-500 animate-spin"></div>
                <div class="mt-2 text-xs text-gray-500">{{ resolvedPlaceholderText }}</div>
              </div>
            </div>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import GeneratedImageCard from './GeneratedImageCard.vue'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

type GridItem =
  | {
      kind: 'image' | 'video'
      key: string
      thumbSrc: string
      originalSrc?: string
      posterSrc?: string
      title?: string
      downloadName?: string
      details?: any
    }
  | {
      kind: 'placeholder'
      key: string
      runId?: string
    }

const props = withDefaults(
  defineProps<{
    items: GridItem[]
    placeholderText?: string
  }>(),
  { placeholderText: undefined }
)

const resolvedPlaceholderText = computed(() => props.placeholderText ?? t('components.media.generating'))

const emit = defineEmits<{
  (e: 'open', key: string): void
}>()

// keep props referenced (TS)
void props
</script>

