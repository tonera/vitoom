<template>
  <div
    class="flex flex-wrap items-center gap-1 border-t border-gray-200 px-2 py-1.5 [.dark_&]:border-gray-700/50 sm:px-2.5"
  >
    <button
      type="button"
      class="inline-flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-[11px] font-medium text-sky-600 hover:bg-sky-50 hover:text-sky-800 [.dark_&]:text-sky-300/95 [.dark_&]:hover:bg-sky-500/10 [.dark_&]:hover:text-sky-200"
      @click="onDownload"
    >
      <svg class="h-3.5 w-3.5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          d="M12 3v11m0 0 4-4m-4 4-4-4M5 17v1a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-1"
        />
      </svg>
      <span>{{ t('agents.artifacts.downloadFile') }}</span>
    </button>
    <span class="text-gray-300 [.dark_&]:text-gray-600" aria-hidden="true">·</span>
    <button
      type="button"
      class="group inline-flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-[11px] font-medium text-gray-600 hover:bg-gray-100 hover:text-gray-900 [.dark_&]:text-gray-400 [.dark_&]:hover:bg-gray-800/80 [.dark_&]:hover:text-gray-200"
      @click="onCopy"
    >
      <span class="relative inline-flex h-3.5 w-3.5 shrink-0 items-center justify-center" aria-hidden="true">
        <svg
          class="absolute h-3.5 w-3.5 transition-opacity"
          :class="copied ? 'pointer-events-none opacity-0' : 'opacity-100'"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2"
        >
          <rect x="9" y="9" width="10" height="10" rx="2" />
          <path stroke-linecap="round" stroke-linejoin="round" d="M15 9V7a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h2" />
        </svg>
        <svg
          class="absolute h-3.5 w-3.5 text-emerald-600 transition-opacity [.dark_&]:text-emerald-400"
          :class="copied ? 'opacity-100' : 'pointer-events-none opacity-0'"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          stroke-width="2.5"
        >
          <path stroke-linecap="round" stroke-linejoin="round" d="M20 6 9 17l-5-5" />
        </svg>
      </span>
      {{ t('agents.artifacts.copyLink') }}
    </button>
  </div>
</template>

<script setup lang="ts">
import { onBeforeUnmount, ref } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

const props = defineProps<{
  url: string
}>()

const COPY_FEEDBACK_MS = 2000
const copied = ref(false)
let copyFeedbackTimer: number | null = null

function fileNameFromUrl(url: string): string {
  try {
    const parsed = new URL(url, window.location.origin)
    const tail = parsed.pathname.split('/').filter(Boolean).pop()
    return tail ? decodeURIComponent(tail) : 'download'
  } catch {
    return 'download'
  }
}

function triggerDownload(href: string, filename: string) {
  const link = document.createElement('a')
  link.href = href
  link.download = filename
  link.rel = 'noopener'
  document.body.appendChild(link)
  link.click()
  link.remove()
}

async function copyToClipboard(text: string): Promise<boolean> {
  const value = String(text || '').trim()
  if (!value) return false
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(value)
      return true
    }
  } catch {
    // Fall back to the textarea path below.
  }

  const textarea = document.createElement('textarea')
  textarea.value = value
  textarea.setAttribute('readonly', 'true')
  textarea.style.position = 'fixed'
  textarea.style.left = '-9999px'
  textarea.style.top = '0'
  document.body.appendChild(textarea)
  textarea.focus()
  textarea.select()
  try {
    return document.execCommand('copy')
  } catch {
    return false
  } finally {
    textarea.remove()
  }
}

async function onDownload() {
  const u = String(props.url || '').trim()
  if (!u) return
  const filename = fileNameFromUrl(u)
  try {
    const res = await fetch(u)
    if (!res.ok) throw new Error(`download failed: ${res.status}`)
    const blob = await res.blob()
    const objectUrl = URL.createObjectURL(blob)
    triggerDownload(objectUrl, filename)
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000)
  } catch {
    triggerDownload(u, filename)
  }
}

async function onCopy() {
  const u = String(props.url || '').trim()
  if (!u) return
  const ok = await copyToClipboard(u)
  if (!ok) return
  copied.value = true
  if (copyFeedbackTimer) window.clearTimeout(copyFeedbackTimer)
  copyFeedbackTimer = window.setTimeout(() => {
    copied.value = false
    copyFeedbackTimer = null
  }, COPY_FEEDBACK_MS)
}

onBeforeUnmount(() => {
  if (copyFeedbackTimer) window.clearTimeout(copyFeedbackTimer)
})
</script>
