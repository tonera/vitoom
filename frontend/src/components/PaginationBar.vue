<template>
  <div v-if="lastPage > 1 || showWhenSingle" class="flex items-center justify-end gap-1.5">
    <button
      type="button"
      class="px-3 py-1.5 rounded-lg border border-gray-200 bg-white text-gray-700 transition-colors hover:bg-gray-50 cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/40 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800/70"
      :disabled="page <= 1"
      @click="emitChange(page - 1)"
    >
      {{ t('components.pagination.prev') }}
    </button>

    <button
      v-for="p in pageWindow.pages"
      :key="`p-${p}`"
      type="button"
      class="min-w-9 px-3 py-1.5 rounded-lg border transition-colors cursor-pointer"
      :class="
        p === page
          ? 'border-indigo-500/50 bg-indigo-600 text-white cursor-default'
          : 'border-gray-200 bg-white text-gray-700 hover:bg-gray-50 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/40 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800/70'
      "
      :disabled="p === page"
      @click="emitChange(p)"
    >
      {{ p }}
    </button>

    <span v-if="pageWindow.showEllipsis" class="px-2 select-none text-gray-400 [.dark_&]:text-gray-500">...</span>

    <button
      v-if="pageWindow.showLast"
      type="button"
      class="min-w-9 px-3 py-1.5 rounded-lg border transition-colors cursor-pointer"
      :class="
        pageWindow.last === page
          ? 'border-indigo-500/50 bg-indigo-600 text-white cursor-default'
          : 'border-gray-200 bg-white text-gray-700 hover:bg-gray-50 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/40 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800/70'
      "
      :disabled="pageWindow.last === page"
      @click="emitChange(pageWindow.last)"
    >
      {{ pageWindow.last }}
    </button>

    <button
      type="button"
      class="px-3 py-1.5 rounded-lg border border-gray-200 bg-white text-gray-700 transition-colors hover:bg-gray-50 cursor-pointer disabled:cursor-not-allowed disabled:opacity-50 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/40 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800/70"
      :disabled="page >= lastPage"
      @click="emitChange(page + 1)"
    >
      {{ t('components.pagination.next') }}
    </button>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = withDefaults(
  defineProps<{
    page: number
    lastPage: number
    perPage: number
    windowSize?: number
    showWhenSingle?: boolean
  }>(),
  {
    windowSize: 10,
    showWhenSingle: false,
  }
)

const emit = defineEmits<{
  (e: 'change', payload: { page: number; perPage: number }): void
}>()

const page = computed(() => Math.max(1, Math.floor(Number(props.page || 1))))
const lastPage = computed(() => Math.max(1, Math.floor(Number(props.lastPage || 1))))
const perPage = computed(() => Math.max(1, Math.floor(Number(props.perPage || 10))))

const pageWindow = computed(() => {
  const total = lastPage.value
  const current = Math.min(Math.max(1, page.value), total)
  const windowSize = Math.max(3, Math.floor(Number(props.windowSize || 10)))

  if (total <= windowSize) {
    return {
      pages: Array.from({ length: total }, (_, i) => i + 1),
      showEllipsis: false,
      showLast: false,
      last: total,
    }
  }

  let start = current - Math.floor(windowSize / 2)
  let end = start + windowSize - 1

  if (start < 1) {
    start = 1
    end = windowSize
  }
  if (end > total) {
    end = total
    start = total - windowSize + 1
  }

  const pages = Array.from({ length: end - start + 1 }, (_, i) => start + i)
  const lastInWindow = pages.length > 0 ? pages[pages.length - 1]! : total
  const showLast = lastInWindow < total
  const showEllipsis = showLast && lastInWindow < total - 1

  return { pages, showEllipsis, showLast, last: total }
})

function emitChange(nextPage: number) {
  const p = Math.min(Math.max(1, Math.floor(nextPage)), lastPage.value)
  emit('change', { page: p, perPage: perPage.value })
}
</script>


