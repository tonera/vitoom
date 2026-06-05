<template>
  <form
    ref="formRef"
    class="agent-home-composer-form shrink-0"
    :class="wrapperClass"
    @submit.prevent="emitSubmit"
  >
    <div
      ref="composerRef"
      class="agent-composer relative flex w-full flex-col gap-2 rounded-3xl border border-gray-200 bg-white px-3 py-2.5 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/55 sm:gap-3 sm:px-4 sm:py-3"
      :class="dragAttachmentOverlay ? 'border-blue-400/70 ring-2 ring-blue-400/25' : ''"
      @dragleave.prevent="onComposerDragLeave"
      @dragover.capture.prevent="onComposerDragOver"
      @drop.capture.prevent="onComposerDrop"
    >
      <div
        v-if="dragAttachmentOverlay"
        class="pointer-events-none absolute inset-0 z-30 flex items-center justify-center rounded-[inherit] bg-white/75 px-3 text-center text-xs font-medium text-blue-700 backdrop-blur-[2px] [.dark_&]:bg-gray-900/55 [.dark_&]:text-sky-100/95"
        aria-hidden="true"
      >
        {{ t('agents.composer.dropToUpload') }}
      </div>
      <div
        v-if="hasQueryInput"
        class="pointer-events-none absolute right-2 top-2 bottom-2 z-20 flex flex-col items-end justify-between sm:right-3"
      >
        <button
          type="button"
          class="pointer-events-auto flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-950 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-700/90 [.dark_&]:hover:text-white"
          :aria-label="t('agents.composer.clear')"
          @click="clearQuery"
        >
          <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
        <div class="pointer-events-auto flex shrink-0 items-center gap-0.5 sm:gap-1">
          <button
            v-if="isStreaming"
            type="button"
            class="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-amber-600/90 text-white shadow-sm transition-colors hover:bg-amber-500"
            :aria-label="t('agents.composer.stopGenerating')"
            @click="emit('interrupt')"
          >
            <svg class="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M6 6h12v12H6z" />
            </svg>
          </button>
          <button
            v-else-if="playbackActive"
            type="button"
            class="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-sky-600/90 text-white shadow-sm transition-colors hover:bg-sky-500"
            :aria-label="t('agents.composer.stopPlayback')"
            @click="emit('stop-playback')"
          >
            <svg class="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M6 6h12v12H6z" />
            </svg>
          </button>
          <button
            v-else-if="micEnabled"
            type="button"
            class="flex h-10 w-10 items-center justify-center rounded-full transition-colors"
            :class="
              micRecording
                ? 'bg-red-600/90 text-white shadow-[0_0_0_3px_rgba(239,68,68,0.25)] animate-pulse hover:bg-red-500'
                : 'text-gray-500 hover:bg-gray-100 hover:text-gray-950 [.dark_&]:text-gray-400 [.dark_&]:hover:bg-gray-700/80 [.dark_&]:hover:text-gray-100 disabled:pointer-events-none disabled:opacity-40'
            "
            :aria-label="micAriaLabel"
            :title="micAriaLabel"
            :disabled="micButtonDisabled"
            @click="onMicClick"
          >
            <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
              />
            </svg>
          </button>
          <button
            type="button"
            class="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-blue-600 text-white shadow-sm transition-colors hover:bg-blue-700 disabled:pointer-events-none disabled:opacity-40"
            :aria-label="t('agents.composer.send')"
            :disabled="submitDisabled"
            @click="emitSubmit"
          >
            <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.25" d="M5 15l7-7 7 7" />
            </svg>
          </button>
        </div>
      </div>

      <div
        v-if="attachmentFile"
        class="flex min-w-0 items-start gap-2"
        :class="hasQueryInput ? 'pr-23 sm:pr-24' : ''"
      >
        <div v-if="attachmentIsImage" class="relative inline-flex shrink-0">
          <div
            v-if="attachmentUploading"
            class="relative flex h-16 w-16 shrink-0 items-center justify-center rounded-xl bg-gray-50 ring-1 ring-gray-200 [.dark_&]:bg-gray-900/70 [.dark_&]:ring-gray-600 sm:h-18 sm:w-18"
            role="status"
            aria-live="polite"
            :aria-label="t('agents.composer.attachmentUploading')"
          >
            <span
              class="h-7 w-7 animate-spin rounded-full border-2 border-gray-200 border-t-blue-500 [.dark_&]:border-gray-600 [.dark_&]:border-t-blue-400"
              aria-hidden="true"
            />
            <button
              type="button"
              class="absolute -right-1 -top-1 flex h-6 w-6 items-center justify-center rounded-full bg-white text-gray-500 ring-1 ring-gray-200 shadow-sm hover:bg-gray-100 hover:text-gray-950 [.dark_&]:bg-gray-900/90 [.dark_&]:text-gray-200 [.dark_&]:ring-gray-600 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-white"
              :aria-label="t('agents.composer.cancelUploadRemove')"
              @click="clearAttachment"
            >
              <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <template v-else-if="attachmentPreviewUrl">
            <img
              :src="attachmentPreviewUrl"
              :alt="t('agents.composer.imagePreviewAlt')"
              class="h-16 w-16 rounded-xl object-cover ring-1 ring-gray-200 [.dark_&]:ring-gray-600 sm:h-18 sm:w-18"
            />
            <button
              type="button"
              class="absolute -right-1 -top-1 flex h-6 w-6 items-center justify-center rounded-full bg-white text-gray-500 ring-1 ring-gray-200 shadow-sm hover:bg-gray-100 hover:text-gray-950 [.dark_&]:bg-gray-900/90 [.dark_&]:text-gray-200 [.dark_&]:ring-gray-600 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-white"
              :aria-label="t('agents.composer.removeAttachment')"
              @click="clearAttachment"
            >
              <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </template>
        </div>
        <div
          v-else
          class="relative flex min-h-16 max-w-full min-w-0 items-center gap-2 rounded-xl bg-gray-50 py-2 pl-3 pr-9 ring-1 ring-gray-200 [.dark_&]:bg-gray-900/40 [.dark_&]:ring-gray-600"
        >
          <span
            v-if="attachmentUploading"
            class="flex h-8 w-8 shrink-0 items-center justify-center"
            role="status"
            :aria-label="t('agents.composer.attachmentUploading')"
          >
            <span
              class="h-6 w-6 animate-spin rounded-full border-2 border-gray-200 border-t-blue-500 [.dark_&]:border-gray-600 [.dark_&]:border-t-blue-400"
              aria-hidden="true"
            />
          </span>
          <svg
            v-else
            class="h-8 w-8 shrink-0 text-gray-500 [.dark_&]:text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="1.5"
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <div class="min-w-0 flex-1">
            <p class="truncate text-sm font-medium text-gray-900 [.dark_&]:text-gray-100">{{ attachmentFile.name }}</p>
            <p class="truncate text-xs text-gray-500 [.dark_&]:text-gray-400">
              <template v-if="attachmentUploading">{{ t('agents.composer.uploading') }}</template>
              <template v-else-if="attachmentRemoteUrl">{{ t('agents.composer.uploaded') }}</template>
              <template v-else>{{ attachmentFile.type || t('agents.composer.unknownType') }}</template>
            </p>
          </div>
          <button
            type="button"
            class="absolute right-1 top-1 flex h-6 w-6 items-center justify-center rounded-full bg-white text-gray-500 ring-1 ring-gray-200 shadow-sm hover:bg-gray-100 hover:text-gray-950 [.dark_&]:bg-gray-900/90 [.dark_&]:text-gray-200 [.dark_&]:ring-gray-600 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-white"
            :aria-label="t('agents.composer.removeAttachment')"
            @click="clearAttachment"
          >
            <svg class="h-3.5 w-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      <textarea
        ref="queryInputRef"
        v-model="query"
        name="q"
        rows="1"
        autocomplete="off"
        spellcheck="false"
        class="agent-query-field min-h-11 w-full min-w-0 resize-none overflow-hidden wrap-anywhere border-0 bg-transparent py-1.5 text-base leading-6 text-gray-950 outline-none ring-0 placeholder:text-gray-400 [.dark_&]:text-gray-100 [.dark_&]:placeholder:text-gray-500 sm:py-2"
        :class="hasQueryInput ? 'pl-0 pr-23 sm:pr-24' : 'px-0'"
        :placeholder="resolvedPlaceholder"
        :disabled="submitDisabled"
        @input="syncQueryBoxHeight"
        @keydown="onQueryKeydown"
        @paste="onQueryPaste"
      />

      <div class="flex min-h-0 w-full min-w-0 items-center justify-between gap-2">
        <div class="flex shrink-0 items-center gap-0.5">
          <button
            type="button"
            class="flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-950 [.dark_&]:text-gray-400 [.dark_&]:hover:bg-gray-700/80 [.dark_&]:hover:text-gray-100 disabled:pointer-events-none disabled:opacity-40"
            :aria-label="t('agents.composer.uploadAttachment')"
            :disabled="submitDisabled || attachmentUploading"
            @click="attachmentInputRef?.click()"
          >
            <svg
              v-if="!attachmentUploading"
              class="h-5 w-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
            </svg>
            <span v-else class="h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-gray-700 [.dark_&]:border-gray-500 [.dark_&]:border-t-gray-200" aria-hidden="true" />
          </button>
          <button
            type="button"
            class="flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-950 [.dark_&]:text-gray-400 [.dark_&]:hover:bg-gray-700/80 [.dark_&]:hover:text-gray-100 disabled:pointer-events-none disabled:opacity-40"
            :aria-label="t('agents.composer.scanUploadAttachment')"
            :disabled="submitDisabled || attachmentUploading"
            @click="qrModalRef?.open()"
          >
            <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M12 18h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z"
              />
            </svg>
          </button>
        </div>

        <div v-if="!hasQueryInput && (micEnabled || isStreaming || playbackActive)" class="flex shrink-0 items-center gap-0.5 sm:gap-1">
          <button
            v-if="isStreaming"
            type="button"
            class="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-amber-600/90 text-white shadow-sm transition-colors hover:bg-amber-500"
            :aria-label="t('agents.composer.stopGenerating')"
            @click="emit('interrupt')"
          >
            <svg class="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M6 6h12v12H6z" />
            </svg>
          </button>
          <button
            v-else-if="playbackActive"
            type="button"
            class="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-sky-600/90 text-white shadow-sm transition-colors hover:bg-sky-500"
            :aria-label="t('agents.composer.stopPlayback')"
            @click="emit('stop-playback')"
          >
            <svg class="h-5 w-5" fill="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M6 6h12v12H6z" />
            </svg>
          </button>
          <button
            v-else-if="micEnabled"
            type="button"
            class="flex h-10 w-10 items-center justify-center rounded-full transition-colors"
            :class="
              micRecording
                ? 'bg-red-600/90 text-white shadow-[0_0_0_3px_rgba(239,68,68,0.25)] animate-pulse hover:bg-red-500'
                : 'text-gray-500 hover:bg-gray-100 hover:text-gray-950 [.dark_&]:text-gray-400 [.dark_&]:hover:bg-gray-700/80 [.dark_&]:hover:text-gray-100 disabled:pointer-events-none disabled:opacity-40'
            "
            :aria-label="micAriaLabel"
            :disabled="micButtonDisabled"
            @click="onMicClick"
          >
            <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
              />
            </svg>
          </button>
        </div>
      </div>

      <input
        ref="attachmentInputRef"
        type="file"
        class="hidden"
        :accept="COMPOSER_ATTACHMENT_ACCEPT"
        @change="onAttachmentPicked"
      />
      <QrUploadModal
        ref="qrModalRef"
        :accept="COMPOSER_ATTACHMENT_ACCEPT"
        @uploaded="onQrAttachmentUploaded"
      />
    </div>
  </form>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import axios from 'axios'
import type { AgentComposerSubmitPayload } from '../agentComposerTypes'
import { showTopSnack } from '../../../composables/useTopSnack'
import type { QrUploadResult } from '../../../composables/useQrCodeUpload'
import { uploadFile } from '../../../utils/upload'
import QrUploadModal from '../../../components/QrUploadModal.vue'

const COMPOSER_ATTACHMENT_ACCEPT =
  'image/*,video/*,audio/*,.pdf,.doc,.docx,.txt,.xls,.xlsx,.ppt,.pptx,.odt,.ods,.odp,.csv,.rtf,.md,.markdown,.html,.htm,.zip,.rar,.7z,.tar,.gz,.tgz,.bz2,.tbz2,.xz,.zst,.lz4,.cab,.jar,.war'

const { t } = useI18n()

const props = withDefaults(
  defineProps<{
    /** 用于计算输入框最大高度：父组件模板中传同名 ref 时会自动解包为元素 */
    boundaryElement?: HTMLElement | null
    wrapperClass?: string
    placeholder?: string
    /** 会话未就绪或流式生成中时为 true */
    submitDisabled?: boolean
    /** 麦克风是否启用（不启用则按钮隐藏） */
    micEnabled?: boolean
    /** 是否可按"开始录音"（通常来自 canStartAudioRecord） */
    micCanStart?: boolean
    /** 当前正在录音 */
    micRecording?: boolean
    /** 显式禁用麦克风按钮（权限异常 / 正在请求等） */
    micDisabled?: boolean
    /** 助手正在流式输出：与麦克风同槽位显示「停止」图标 */
    isStreaming?: boolean
    /** TTS 正在缓冲或播放：显示“停止播放”入口 */
    playbackActive?: boolean
  }>(),
  {
    boundaryElement: undefined,
    wrapperClass: 'w-full max-w-[720px] shrink-0',
    placeholder: '',
    submitDisabled: false,
    micEnabled: false,
    micCanStart: true,
    micRecording: false,
    micDisabled: false,
    isStreaming: false,
    playbackActive: false,
  },
)

const boundaryEl = () => props.boundaryElement ?? null

const emit = defineEmits<{
  submit: [payload: AgentComposerSubmitPayload]
  'mic-toggle': []
  interrupt: []
  'stop-playback': []
}>()

const micButtonDisabled = computed(() => {
  if (!props.micEnabled) return true
  if (props.micDisabled) return true
  if (props.micRecording) return false
  return !props.micCanStart
})

const resolvedPlaceholder = computed(
  () => props.placeholder || t('agents.composer.askPlaceholder'),
)

const micAriaLabel = computed(() =>
  props.micRecording ? t('agents.composer.stopRecording') : t('agents.composer.startRecording'),
)

const onMicClick = () => {
  if (micButtonDisabled.value && !props.micRecording) return
  emit('mic-toggle')
}

const query = ref('')
const attachmentFile = ref<File | null>(null)
const attachmentPreviewUrl = ref<string | null>(null)
const attachmentRemoteUrl = ref<string | null>(null)
const attachmentUploading = ref(false)
const attachmentInputRef = ref<HTMLInputElement | null>(null)
const qrModalRef = ref<InstanceType<typeof QrUploadModal> | null>(null)
const formRef = ref<HTMLElement | null>(null)
const composerRef = ref<HTMLElement | null>(null)
const queryInputRef = ref<HTMLTextAreaElement | null>(null)
const dragAttachmentOverlay = ref(false)

let attachmentUploadEpoch = 0
let queryBoxResizeObserver: ResizeObserver | null = null

const hasQueryInput = computed(() => query.value.trim().length > 0 || !!attachmentFile.value)
const attachmentIsImage = computed(() => !!attachmentFile.value?.type?.startsWith('image/'))
const canAcceptDragAttachment = computed(() => !props.submitDisabled && !attachmentUploading.value)

const clearAttachment = () => {
  attachmentUploadEpoch++
  const prev = attachmentPreviewUrl.value
  if (prev?.startsWith('blob:')) {
    URL.revokeObjectURL(prev)
  }
  attachmentPreviewUrl.value = null
  attachmentFile.value = null
  attachmentRemoteUrl.value = null
  attachmentUploading.value = false
}

const focusQueryInput = (): boolean => {
  const el = queryInputRef.value
  if (!el || el.disabled) return false
  el.focus({ preventScroll: true })
  return document.activeElement === el
}

/** 输入框解禁/布局更新可能晚于 isStreaming，用 rAF 重试几次避免偶发失焦 */
const scheduleQueryInputFocus = () => {
  let attempts = 0
  const maxAttempts = 12
  const tryFocus = () => {
    if (focusQueryInput()) return
    if (++attempts >= maxAttempts) return
    requestAnimationFrame(tryFocus)
  }
  nextTick(tryFocus)
}

const clearQuery = () => {
  query.value = ''
  clearAttachment()
  nextTick(() => {
    syncQueryBoxHeight()
    scheduleQueryInputFocus()
  })
}

const syncQueryBoxHeight = () => {
  const root = boundaryEl()
  const composer = composerRef.value
  const ta = queryInputRef.value
  if (!composer || !ta) return

  const compRect = composer.getBoundingClientRect()
  const cs = getComputedStyle(composer)
  const padY = parseFloat(cs.paddingTop) + parseFloat(cs.paddingBottom)
  let inner: number
  if (root) {
    const rootRect = root.getBoundingClientRect()
    inner = rootRect.bottom - compRect.top - padY - 12
  } else {
    inner = window.innerHeight - compRect.top - padY - 24
  }
  const maxH = Math.max(96, Math.floor(Math.max(inner, 0) * 0.9))

  ta.style.maxHeight = `${maxH}px`
  ta.style.height = 'auto'
  const contentH = ta.scrollHeight
  const next = Math.min(contentH, maxH)
  ta.style.height = `${next}px`
  ta.style.overflowY = contentH > maxH ? 'auto' : 'hidden'
}

const onQueryKeydown = (e: KeyboardEvent) => {
  if (e.key !== 'Enter' || e.isComposing) return

  // Ctrl+Enter / ⌘+Enter：插入换行（与常见「Enter 发送」产品一致）
  if (e.ctrlKey || e.metaKey) {
    e.preventDefault()
    const ta = queryInputRef.value
    if (!ta) return
    const start = ta.selectionStart ?? 0
    const end = ta.selectionEnd ?? 0
    const v = query.value
    query.value = `${v.slice(0, start)}\n${v.slice(end)}`
    nextTick(() => {
      const el = queryInputRef.value
      if (!el) return
      const pos = start + 1
      el.selectionStart = pos
      el.selectionEnd = pos
      syncQueryBoxHeight()
    })
    return
  }

  // Shift+Enter / Alt+Enter 等：保留浏览器默认（换行）
  if (e.shiftKey || e.altKey) return

  // 单独 Enter：发送
  e.preventDefault()
  emitSubmit()
}

const attachmentUploadErrorMessage = (e: unknown) => {
  if (axios.isAxiosError(e)) {
    const d = e.response?.data?.detail
    if (typeof d === 'string' && d.trim()) return d
  }
  return e instanceof Error ? e.message : t('agents.composer.uploadFailed')
}

const emitSubmit = () => {
  if (props.submitDisabled) return
  const text = query.value.trim()
  if (!text && !attachmentFile.value) return

  if (attachmentFile.value) {
    if (attachmentUploading.value) {
      showTopSnack(t('agents.composer.attachmentUploadingWait'))
      return
    }
    if (!attachmentRemoteUrl.value) {
      showTopSnack(t('agents.composer.attachmentNotUploaded'))
      return
    }
  }

  emit('submit', {
    text,
    attachmentRemoteUrl: attachmentRemoteUrl.value,
    attachmentFile: attachmentFile.value,
  })
}

const applyAttachmentFile = async (file: File) => {
  clearAttachment()
  attachmentFile.value = file
  const uploadEpoch = attachmentUploadEpoch
  const blobPreview = file.type.startsWith('image/') ? URL.createObjectURL(file) : null
  attachmentPreviewUrl.value = blobPreview
  attachmentUploading.value = true

  try {
    const url = await uploadFile(file)
    if (uploadEpoch !== attachmentUploadEpoch) return
    attachmentRemoteUrl.value = url
    if (blobPreview) {
      URL.revokeObjectURL(blobPreview)
      attachmentPreviewUrl.value = url
    }
    showTopSnack(t('agents.composer.attachmentUploaded'))
  } catch (e: unknown) {
    if (uploadEpoch !== attachmentUploadEpoch) return
    showTopSnack(attachmentUploadErrorMessage(e))
    clearAttachment()
  } finally {
    if (uploadEpoch === attachmentUploadEpoch) {
      attachmentUploading.value = false
    }
    nextTick(() => syncQueryBoxHeight())
  }
}

const applyAttachmentFromRemote = (result: QrUploadResult) => {
  const url = String(result.url || '').trim()
  if (!url) return
  clearAttachment()
  const name = String(result.file_name || '').trim() || 'upload'
  const mime = String(result.mime_type || '').trim() || 'application/octet-stream'
  attachmentFile.value = new File([], name, { type: mime })
  attachmentRemoteUrl.value = url
  attachmentPreviewUrl.value = mime.startsWith('image/') ? url : null
  showTopSnack(t('agents.composer.attachmentUploaded'))
  nextTick(() => syncQueryBoxHeight())
}

const onQrAttachmentUploaded = (result: QrUploadResult) => {
  applyAttachmentFromRemote(result)
}

const onAttachmentPicked = async (ev: Event) => {
  const input = ev.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (!file) return
  await applyAttachmentFile(file)
}

/** 剪贴板中的首个文件（截图 / 复制文件等）；无文件时返回 null，保留默认文本粘贴 */
const firstFileFromClipboard = (dt: DataTransfer | null): File | null => {
  if (!dt) return null
  if (dt.files?.length) {
    const f = dt.files.item(0)
    if (f) return f
  }
  const items = dt.items
  if (!items?.length) return null
  for (let i = 0; i < items.length; i++) {
    const it = items[i]
    if (!it || it.kind !== 'file') continue
    const f = it.getAsFile()
    if (f) return f
  }
  return null
}

const onQueryPaste = (e: ClipboardEvent) => {
  if (!canAcceptDragAttachment.value) return
  const file = firstFileFromClipboard(e.clipboardData)
  if (!file) return
  e.preventDefault()
  void applyAttachmentFile(file)
}

const onComposerDragOver = (e: DragEvent) => {
  if (!canAcceptDragAttachment.value) return
  if (!e.dataTransfer?.types?.includes('Files')) return
  e.preventDefault()
  try {
    e.dataTransfer.dropEffect = 'copy'
  } catch {
    /* ignore */
  }
  dragAttachmentOverlay.value = true
}

const onComposerDragLeave = (e: DragEvent) => {
  const el = e.currentTarget as HTMLElement
  const related = e.relatedTarget as Node | null
  if (related && el.contains(related)) return
  dragAttachmentOverlay.value = false
}

const onComposerDrop = (e: DragEvent) => {
  dragAttachmentOverlay.value = false
  if (!e.dataTransfer?.types?.includes('Files')) return
  if (!canAcceptDragAttachment.value) {
    showTopSnack(t('agents.composer.cannotAddAttachment'))
    return
  }
  e.preventDefault()
  const files = e.dataTransfer?.files
  const file = files && files.length > 0 ? files[0] : null
  if (!file) return
  void applyAttachmentFile(file)
}

const onWindowDragEnd = () => {
  dragAttachmentOverlay.value = false
}

watch(query, () => nextTick(() => syncQueryBoxHeight()))
watch(attachmentPreviewUrl, () => nextTick(() => syncQueryBoxHeight()))
watch(attachmentFile, () => nextTick(() => syncQueryBoxHeight()))

/**
 * 助手回复结束后收回输入框焦点。
 * isStreaming 先结束而 submitDisabled 仍 true（session 尚未回到 ready）时，
 * 仅监听 isStreaming 会错过聚焦；需等输入解禁后再 focus。
 */
watch(
  () => [props.isStreaming, props.submitDisabled] as const,
  ([streaming, submitDisabled], prev) => {
    if (streaming || submitDisabled) return
    const prevStreaming = prev?.[0] ?? false
    const prevSubmitDisabled = prev?.[1] ?? true
    if (!prevStreaming && !prevSubmitDisabled) return
    scheduleQueryInputFocus()
  },
)

const onWindowResize = () => syncQueryBoxHeight()

onMounted(() => {
  window.addEventListener('resize', onWindowResize)
  window.addEventListener('dragend', onWindowDragEnd)
  nextTick(() => {
    requestAnimationFrame(() => {
      syncQueryBoxHeight()
      if (typeof ResizeObserver === 'undefined') return
      queryBoxResizeObserver = new ResizeObserver(() => syncQueryBoxHeight())
      const observe = (el: HTMLElement | null | undefined) => {
        if (el) queryBoxResizeObserver!.observe(el)
      }
      observe(boundaryEl())
      observe(formRef.value)
      observe(composerRef.value)
    })
  })
})

watch(
  () => props.boundaryElement,
  (el, prev) => {
    if (!queryBoxResizeObserver) return
    if (prev) queryBoxResizeObserver.unobserve(prev)
    if (el) queryBoxResizeObserver.observe(el)
    nextTick(() => syncQueryBoxHeight())
  },
)

onUnmounted(() => {
  window.removeEventListener('resize', onWindowResize)
  window.removeEventListener('dragend', onWindowDragEnd)
  queryBoxResizeObserver?.disconnect()
  queryBoxResizeObserver = null
  clearAttachment()
})

defineExpose({
  reset: clearQuery,
  syncHeight: syncQueryBoxHeight,
})
</script>

<style scoped>
.agent-home-composer-form :where(button:not(:disabled)) {
  cursor: pointer;
}

.agent-query-field::-webkit-scrollbar {
  width: 8px;
}
.agent-query-field::-webkit-scrollbar-track {
  background: rgba(243, 244, 246, 0.95);
  border-radius: 999px;
}
.agent-query-field::-webkit-scrollbar-thumb {
  background: rgba(156, 163, 175, 0.85);
  border-radius: 999px;
}
.agent-query-field::-webkit-scrollbar-thumb:hover {
  background: rgba(107, 114, 128, 0.9);
}
</style>
<style>
html.dark .agent-home-composer-form .agent-query-field::-webkit-scrollbar-track {
  background: rgba(31, 41, 55, 0.45);
}
html.dark .agent-home-composer-form .agent-query-field::-webkit-scrollbar-thumb {
  background: rgba(75, 85, 99, 0.95);
}
html.dark .agent-home-composer-form .agent-query-field::-webkit-scrollbar-thumb:hover {
  background: rgba(107, 114, 128, 0.95);
}
</style>
