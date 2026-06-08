<script setup lang="ts">
import { onBeforeUnmount, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { showTopSnack } from '../../../composables/useTopSnack'
import { useAnchoredPopover } from '../../../composables/useAnchoredPopover'

const { t } = useI18n()

const props = defineProps<{
  /** 本条助手消息的原始 Markdown（与流式/落库一致） */
  rawMarkdown: string
}>()

const { open, anchorRef, popoverRef, style, toggle, close, schedulePosition } = useAnchoredPopover({
  gapPx: 6,
  paddingPx: 10,
})

const onMoreClick = (e: MouseEvent) => {
  e.stopPropagation()
  toggle()
}

const onCopyClick = (e: MouseEvent) => {
  e.stopPropagation()
  void navigator.clipboard.writeText(props.rawMarkdown ?? '').then(
    () => {
      close()
    },
    () => {
      showTopSnack(t('agents.messageMenu.copyFailed'))
    },
  )
}

function exportFilename(): string {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `assistant-${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}.md`
}

const onExportClick = (e: MouseEvent) => {
  e.stopPropagation()
  const text = props.rawMarkdown ?? ''
  try {
    const blob = new Blob([text], { type: 'text/markdown;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = exportFilename()
    a.rel = 'noopener'
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
    close()
  } catch {
    showTopSnack(t('agents.messageMenu.exportFailed'))
  }
}

let stopOutside: (() => void) | null = null

watch(open, (isOpen) => {
  stopOutside?.()
  stopOutside = null
  if (!isOpen || typeof window === 'undefined') return
  const onPointerDown = (ev: PointerEvent) => {
    const t = ev.target as Node | null
    if (!t) return
    if (anchorRef.value?.contains(t)) return
    if (popoverRef.value?.contains(t)) return
    close()
  }
  window.addEventListener('pointerdown', onPointerDown, true)
  stopOutside = () => window.removeEventListener('pointerdown', onPointerDown, true)
})

watch(
  () => props.rawMarkdown,
  () => {
    if (open.value) schedulePosition()
  },
)

onBeforeUnmount(() => {
  stopOutside?.()
  stopOutside = null
})
</script>

<template>
  <div class="mt-1 flex justify-end">
    <button
      ref="anchorRef"
      type="button"
      class="inline-flex h-6 min-w-5 cursor-pointer items-center justify-center rounded px-0.5 text-[11px] font-semibold leading-none text-gray-500 hover:bg-gray-200/80 hover:text-gray-900 [.dark_&]:hover:bg-gray-700/40 [.dark_&]:hover:text-gray-300"
      :title="t('agents.messageMenu.more')"
      :aria-label="t('agents.messageMenu.moreActions')"
      aria-haspopup="menu"
      :aria-expanded="open"
      @click="onMoreClick"
    >
      <span class="font-mono tracking-tighter" aria-hidden="true">...</span>
    </button>
    <Teleport to="body">
      <div
        v-if="open"
        ref="popoverRef"
        class="fixed z-10050 min-w-34 overflow-hidden rounded-lg border border-gray-200 bg-white/98 py-1 text-sm text-gray-900 shadow-xl backdrop-blur-sm [.dark_&]:border-gray-600/85 [.dark_&]:bg-gray-900/98 [.dark_&]:text-gray-100"
        :style="style"
        role="menu"
        @click.stop
      >
        <button
          type="button"
          role="menuitem"
          class="flex w-full cursor-pointer items-center gap-2 px-3 py-2 text-left text-gray-900 hover:bg-gray-100 [.dark_&]:text-gray-100 [.dark_&]:hover:bg-gray-800/90"
          @click="onCopyClick"
        >
          <svg
            class="h-4 w-4 shrink-0 text-gray-500 [.dark_&]:text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
            />
          </svg>
          {{ t('agents.messageMenu.copy') }}
        </button>
        <button
          type="button"
          role="menuitem"
          disabled
          class="flex w-full cursor-not-allowed items-center gap-2 px-3 py-2 text-left text-gray-400 [.dark_&]:text-gray-500"
          :title="t('agents.messageMenu.comingSoon')"
        >
          <svg
            class="h-4 w-4 shrink-0 text-gray-400 [.dark_&]:text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"
            />
          </svg>
          {{ t('agents.messageMenu.favorite') }}
        </button>
        <button
          type="button"
          role="menuitem"
          class="flex w-full cursor-pointer items-center gap-2 px-3 py-2 text-left text-gray-900 hover:bg-gray-100 [.dark_&]:text-gray-100 [.dark_&]:hover:bg-gray-800/90"
          @click="onExportClick"
        >
          <svg
            class="h-4 w-4 shrink-0 text-gray-500 [.dark_&]:text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
            />
          </svg>
          {{ t('agents.messageMenu.export') }}
        </button>
      </div>
    </Teleport>
  </div>
</template>
