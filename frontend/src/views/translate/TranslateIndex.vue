<template>
  <div class="flex h-full min-h-0 flex-col overflow-hidden bg-gray-50 text-gray-950 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100">
    <div class="shrink-0 border-b border-gray-200 bg-white px-5 py-4 [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
      <h1 class="text-xl font-semibold">{{ t('translate.title') }}</h1>
      <p class="mt-1 text-sm text-gray-500 [.dark_&]:text-gray-400">{{ t('translate.subtitle') }}</p>
    </div>

    <div class="grid min-h-0 flex-1 grid-cols-1 overflow-hidden lg:grid-cols-2">
      <!-- Source -->
      <section class="flex min-h-0 flex-col border-b border-gray-200 bg-white [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900 lg:border-b-0 lg:border-r">
        <div class="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-gray-100 px-4 py-3 [.dark_&]:border-gray-800">
          <h2 class="text-sm font-semibold">{{ t('translate.sourcePanel') }}</h2>
          <TranslateLanguagePicker
            v-model="sourceLang"
            :label="t('translate.sourceLanguage')"
            :options="languageOptions"
          />
        </div>

        <div
          class="relative flex min-h-0 flex-1 flex-col p-4"
          :class="dragOverlay ? 'ring-2 ring-inset ring-blue-400/40' : ''"
          @dragleave.prevent="onDragLeave"
          @dragover.prevent="onDragOver"
          @drop.prevent="onDrop"
        >
          <div
            v-if="dragOverlay"
            class="pointer-events-none absolute inset-3 z-20 flex items-center justify-center rounded-2xl border-2 border-dashed border-blue-400/70 bg-blue-50/80 text-sm font-medium text-blue-700 [.dark_&]:bg-blue-500/10 [.dark_&]:text-sky-100"
          >
            {{ t('translate.dropToUpload') }}
          </div>

          <div
            class="relative flex min-h-0 flex-1 flex-col rounded-2xl border border-gray-200 bg-gray-50/80 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950/40"
          >
            <div v-if="attachmentFile" class="shrink-0 border-b border-gray-200 px-3 py-3 [.dark_&]:border-gray-700">
              <div v-if="attachmentIsImage && attachmentPreviewUrl" class="relative inline-flex shrink-0">
                <img
                  :src="attachmentPreviewUrl"
                  :alt="t('translate.imagePreviewAlt')"
                  class="h-20 w-20 rounded-xl object-cover ring-1 ring-gray-200 [.dark_&]:ring-gray-700"
                />
                <button
                  type="button"
                  class="absolute -right-1 -top-1 flex h-6 w-6 cursor-pointer items-center justify-center rounded-full bg-white text-gray-500 ring-1 ring-gray-200 shadow-sm hover:bg-gray-100 [.dark_&]:bg-gray-900 [.dark_&]:ring-gray-600"
                  :aria-label="t('translate.removeAttachment')"
                  @click="clearAttachment"
                >
                  ×
                </button>
              </div>
              <div
                v-else
                class="flex min-w-0 items-center gap-2 rounded-xl bg-white/80 px-3 py-2 ring-1 ring-gray-200 [.dark_&]:bg-gray-900/50 [.dark_&]:ring-gray-700"
              >
                <span v-if="attachmentUploading || convertingDocument" class="h-5 w-5 shrink-0 animate-spin rounded-full border-2 border-gray-300 border-t-indigo-500" />
                <div class="min-w-0 flex-1">
                  <p class="truncate text-sm font-medium">{{ attachmentFile.name }}</p>
                  <p class="truncate text-xs text-gray-500 [.dark_&]:text-gray-400">
                    <template v-if="attachmentUploading">{{ t('translate.attachmentUploading') }}</template>
                    <template v-else-if="convertingDocument">{{ convertProgressLabel }}</template>
                    <template v-else-if="attachmentRemoteUrl">{{ t('translate.uploaded') }}</template>
                  </p>
                  <div
                    v-if="convertingDocument"
                    class="mt-2 h-1.5 overflow-hidden rounded-full bg-gray-200 [.dark_&]:bg-gray-700"
                  >
                    <div
                      class="h-full rounded-full bg-indigo-500 transition-all duration-300"
                      :class="convertProgress > 0 ? '' : 'w-1/3 animate-pulse'"
                      :style="convertProgress > 0 ? { width: `${Math.min(100, convertProgress)}%` } : undefined"
                    />
                  </div>
                </div>
                <button
                  type="button"
                  class="cursor-pointer text-xs text-gray-500 hover:text-gray-800 [.dark_&]:hover:text-gray-200"
                  @click="clearAttachment"
                >
                  {{ t('translate.removeAttachment') }}
                </button>
              </div>
            </div>

            <textarea
              v-model="sourceText"
              class="min-h-0 flex-1 resize-none border-0 bg-transparent px-4 py-3 text-sm leading-6 outline-none ring-0 [.dark_&]:text-gray-100"
              :placeholder="t('translate.sourcePlaceholder')"
              :disabled="translating"
            />

            <div class="flex shrink-0 items-center justify-between gap-2 px-2 pb-2">
              <button
                type="button"
                class="flex h-10 w-10 shrink-0 cursor-pointer items-center justify-center rounded-full text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-950 [.dark_&]:text-gray-400 [.dark_&]:hover:bg-gray-700/80 [.dark_&]:hover:text-gray-100 disabled:cursor-not-allowed disabled:pointer-events-none disabled:opacity-40"
                :aria-label="t('agents.composer.uploadAttachment')"
                :disabled="translating || attachmentUploading"
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
              <div class="ml-auto flex min-w-0 flex-col items-end gap-0.5">
                <span v-if="sourceCharCount" class="text-xs text-gray-400">{{ sourceCharCount }}</span>
                <span
                  v-if="sourceChunkHint"
                  class="text-xs text-amber-600 [.dark_&]:text-amber-400"
                >
                  {{ sourceChunkHint }}
                </span>
              </div>
            </div>

            <input
              ref="attachmentInputRef"
              type="file"
              class="hidden"
              :accept="attachmentAccept"
              @change="onFileInputChange"
            />
          </div>
        </div>
      </section>

      <!-- Result -->
      <section class="flex min-h-0 flex-col bg-white [.dark_&]:bg-gray-900">
        <div class="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-gray-100 px-4 py-3 [.dark_&]:border-gray-800">
          <h2 class="text-sm font-semibold">{{ t('translate.resultPanel') }}</h2>
          <div class="flex flex-wrap items-center gap-2">
            <TranslateLanguagePicker
              v-model="targetLang"
              :label="t('translate.targetLanguage')"
              :options="languageOptions"
            />
            <select
              v-model="selectedLoadName"
              class="max-w-[12rem] cursor-pointer rounded-xl border border-gray-200 bg-white px-3 py-1.5 text-sm text-gray-800 outline-none transition focus:border-indigo-400 focus:ring-2 focus:ring-indigo-500/20 disabled:cursor-not-allowed disabled:opacity-50 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900/60 [.dark_&]:text-gray-100"
              :disabled="translating || translateModelsLoading"
              :aria-label="t('translate.model')"
            >
              <option value="">{{ t('translate.defaultModel') }}</option>
              <option
                v-for="m in translateModels"
                :key="m.load_name"
                :value="m.load_name"
              >
                {{ m.label }}
              </option>
            </select>
            <button
              type="button"
              class="cursor-pointer rounded-xl bg-indigo-600 px-4 py-1.5 text-sm font-medium text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-50"
              :disabled="translateDisabled"
              @click="submitTranslate"
            >
              {{ translateButtonLabel }}
            </button>
          </div>
        </div>

        <div class="relative flex min-h-0 flex-1 flex-col p-4">
          <div
            v-if="translatingTask && translateChunkTotal > 1"
            class="mb-2 shrink-0 rounded-xl border border-indigo-200 bg-indigo-50/80 px-3 py-2 text-xs text-indigo-700 [.dark_&]:border-indigo-500/30 [.dark_&]:bg-indigo-500/10 [.dark_&]:text-indigo-200"
          >
            {{ t('translate.chunkProgress', { current: translateChunkIndex, total: translateChunkTotal }) }}
          </div>
          <textarea
            :value="outputText"
            readonly
            class="min-h-0 flex-1 resize-none rounded-2xl border border-gray-200 bg-gray-50/50 px-4 py-3 pb-12 text-sm leading-6 outline-none [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950/30 [.dark_&]:text-gray-100"
            :placeholder="t('translate.resultPlaceholder')"
          />
          <button
            type="button"
            class="absolute bottom-7 right-7 inline-flex h-9 w-9 cursor-pointer items-center justify-center rounded-xl border border-gray-200 bg-white text-gray-600 shadow-sm transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800"
            :disabled="!outputText.trim()"
            :title="outputCopied ? t('common.copied') : t('common.copy')"
            :aria-label="outputCopied ? t('common.copied') : t('common.copy')"
            @click="copyOutput"
          >
            <svg
              v-if="outputCopied"
              class="h-4 w-4 text-emerald-600 [.dark_&]:text-emerald-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M20 6 9 17l-5-5" />
            </svg>
            <svg
              v-else
              class="h-4 w-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              aria-hidden="true"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M8 7V6a2 2 0 012-2h7a2 2 0 012 2v9a2 2 0 01-2 2h-1M8 7h7a2 2 0 012 2v7a2 2 0 01-2 2H10a2 2 0 01-2-2V7z"
              />
            </svg>
          </button>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import TranslateLanguagePicker from './components/TranslateLanguagePicker.vue'
import { documentToText, getDocumentConvertConfig } from '../../api/documents'
import { listModels, type ModelRecord } from '../../api/models'
import { showTopSnack, showTopSnackError } from '../../composables/useTopSnack'
import { handleApiError } from '../../utils/api'
import { resolveWsMessage } from '../../utils/errorMessage'
import { countTranslateChunks, joinTranslatedChunks, splitTextForTranslation } from '../../utils/translateChunk'
import { uploadFile } from '../../utils/upload'
import {
  connectTaskWs,
  createTask,
  getOriginalFileUrl,
  type TaskCreateRequest,
  type TaskWsMessage,
} from '../../utils/taskRunner'

const ATTACHMENT_ACCEPT = 'image/*,.pdf,.doc,.docx,.txt,.md,.markdown,.rtf'

const LANGUAGE_CODES = ['zh', 'en', 'ja', 'ko', 'de', 'fr', 'es', 'ru', 'ar', 'pt', 'it', 'cs'] as const

const { t } = useI18n()

function showTranslateError(error: unknown, fallbackKey = 'translate.translateFailed') {
  showTopSnackError(handleApiError(error).message || t(fallbackKey))
}

function resolveTaskWsError(msg: TaskWsMessage): string {
  const payload = msg as TaskWsMessage & {
    message_code?: string
    message_params?: Record<string, unknown>
  }
  return (
    resolveWsMessage({
      message_code: payload.message_code,
      message_params: payload.message_params,
      msg: payload.error,
      message: payload.error,
    }) || String(payload.error || payload.status || t('translate.translateFailed'))
  )
}

const sourceLang = ref('zh')
const targetLang = ref('en')
const selectedLoadName = ref('')
const sourceText = ref('')
const outputText = ref('')
const outputCopied = ref(false)

const attachmentFile = ref<File | null>(null)
const attachmentInputRef = ref<HTMLInputElement | null>(null)
const attachmentRemoteUrl = ref('')
const attachmentPreviewUrl = ref('')
const attachmentUploading = ref(false)
const convertingDocument = ref(false)
const convertProgress = ref(0)
const convertStage = ref('')
const convertElapsedSec = ref(0)
const translatingTask = ref(false)
const translateChunkIndex = ref(0)
const translateChunkTotal = ref(0)
const dragOverlay = ref(false)
const dragDepth = ref(0)

type TranslateModelOption = { load_name: string; label: string }
const translateModels = ref<TranslateModelOption[]>([])
const translateModelsLoading = ref(false)

let wsHandle: { disconnect: () => void } | null = null
let convertWsHandle: { disconnect: () => void } | null = null
let copyFeedbackTimer: number | null = null
let convertElapsedTimer: number | null = null

const COPY_FEEDBACK_MS = 3000

const languageOptions = computed(() =>
  LANGUAGE_CODES.map((code) => ({
    value: code,
    label: t(`translate.languages.${code}`),
  })),
)

const attachmentAccept = ATTACHMENT_ACCEPT
const attachmentIsImage = computed(() => Boolean(attachmentFile.value && attachmentFile.value.type.startsWith('image/')))
const sourceCharCount = computed(() => {
  const n = sourceText.value.length
  return n > 0 ? `${n}` : ''
})
const sourceChunkHint = computed(() => {
  const text = sourceText.value.trim()
  if (!text) return ''
  const total = countTranslateChunks(text)
  if (total <= 1) return ''
  return t('translate.willChunkHint', { total })
})
const translateButtonLabel = computed(() => {
  if (!translatingTask.value) return t('translate.translate')
  if (translateChunkTotal.value > 1) {
    return t('translate.chunkProgress', {
      current: translateChunkIndex.value,
      total: translateChunkTotal.value,
    })
  }
  return t('translate.translating')
})
const translating = computed(() => translatingTask.value || attachmentUploading.value || convertingDocument.value)
const translateDisabled = computed(() => translating.value)

const convertProgressLabel = computed(() => {
  if (convertProgress.value > 0) {
    return t('translate.convertingDocumentProgress', { progress: convertProgress.value })
  }
  if (convertStage.value === 'ocr') {
    return t('translate.convertingDocumentOcr', { seconds: convertElapsedSec.value })
  }
  return t('translate.convertingDocument', { seconds: convertElapsedSec.value })
})

function mapTranslateModel(m: ModelRecord): TranslateModelOption | null {
  const loadName = String(m?.load_name || '').trim()
  if (!loadName) return null
  const name = String(m?.name || '').trim()
  return {
    load_name: loadName,
    label: name && name !== loadName ? `${name} (${loadName})` : loadName,
  }
}

async function fetchTranslateModels() {
  translateModelsLoading.value = true
  try {
    const resp = await listModels({
      modality: 'translate',
      limit: 200,
      offset: 0,
    })
    const rows = Array.isArray(resp?.data) ? resp.data : []
    const seen = new Set<string>()
    const options: TranslateModelOption[] = []
    for (const row of rows) {
      const opt = mapTranslateModel(row)
      if (!opt || seen.has(opt.load_name)) continue
      seen.add(opt.load_name)
      options.push(opt)
    }
    options.sort((a, b) => a.label.localeCompare(b.label, undefined, { sensitivity: 'base' }))
    translateModels.value = options
  } catch (e) {
    console.warn('[TranslateIndex] failed to load translate models', e)
    translateModels.value = []
  } finally {
    translateModelsLoading.value = false
  }
}

function isDocLike(file: File) {
  const name = file.name.toLowerCase()
  return name.endsWith('.docx') || name.endsWith('.doc') || name.endsWith('.pdf')
}

function isPdfFile(file: File) {
  const name = file.name.toLowerCase()
  return name.endsWith('.pdf') || file.type === 'application/pdf'
}

function isOfficeDoc(file: File) {
  const name = file.name.toLowerCase()
  return name.endsWith('.docx') || name.endsWith('.doc')
}

function isPlainTextFile(file: File) {
  const name = file.name.toLowerCase()
  return (
    file.type.startsWith('text/') ||
    name.endsWith('.txt') ||
    name.endsWith('.md') ||
    name.endsWith('.markdown') ||
    name.endsWith('.rtf')
  )
}

function startConvertElapsedTimer() {
  stopConvertElapsedTimer()
  convertElapsedSec.value = 0
  convertElapsedTimer = window.setInterval(() => {
    convertElapsedSec.value += 1
  }, 1000)
}

function stopConvertElapsedTimer() {
  if (convertElapsedTimer) {
    window.clearInterval(convertElapsedTimer)
    convertElapsedTimer = null
  }
}

function resetConvertProgress() {
  convertProgress.value = 0
  convertStage.value = ''
  convertElapsedSec.value = 0
}

function disconnectConvertWs() {
  if (convertWsHandle) {
    convertWsHandle.disconnect()
    convertWsHandle = null
  }
}

function extractOcrTextFromMessage(msg: TaskWsMessage): string {
  const inline = String((msg as any)?.content || (msg as any)?.text || '').trim()
  if (inline) return inline
  return ''
}

async function convertPdfViaOcr(url: string): Promise<string> {
  const cfg = await getDocumentConvertConfig()
  const loadName = String(cfg?.pdf_ocr_model || '').trim()
  if (!loadName) throw new Error('missing pdf_ocr_model')

  convertingDocument.value = true
  resetConvertProgress()
  convertStage.value = 'ocr'
  startConvertElapsedTimer()
  disconnectConvertWs()

  const timeoutMs = Number(cfg?.timeout_seconds) > 0
    ? Math.ceil(Number(cfg.timeout_seconds) * 1000 + 60_000)
    : 660_000

  return new Promise((resolve, reject) => {
    let settled = false
    let timeoutHandle: number | undefined

    const finish = (fn: () => void) => {
      if (settled) return
      settled = true
      if (timeoutHandle) window.clearTimeout(timeoutHandle)
      disconnectConvertWs()
      stopConvertElapsedTimer()
      convertingDocument.value = false
      fn()
    }

    timeoutHandle = window.setTimeout(() => {
      finish(() => reject(new Error(t('translate.convertDocumentTimeout'))))
    }, timeoutMs)

    createTask({
      task_type: 'mini',
      job_type: 'OCR',
      load_name: loadName,
      tpl_list: [url],
      extract: { task: 'text' },
    })
      .then((created) => {
        const taskId = String(created.task_id || '').trim()
        if (!taskId) throw new Error('missing task_id')

        convertWsHandle = connectTaskWs(taskId, async (msg) => {
          const progress = Number(msg.progress)
          if (Number.isFinite(progress) && progress >= 0) {
            convertProgress.value = Math.min(100, Math.round(progress))
          }

          const msgType = String(msg.type || '').toLowerCase()
          if (msgType === 'result') {
            const text = extractOcrTextFromMessage(msg)
            if (text) {
              finish(() => resolve(text))
            }
          }

          const status = String(msg.status || '').toLowerCase()
          if (status === 'failed' || status === 'cancelled') {
            finish(() => reject(new Error(String(msg.error || status))))
            return
          }
          if (status === 'completed') {
            const text = extractOcrTextFromMessage(msg)
            if (text) {
              finish(() => resolve(text))
              return
            }
            const files = Array.isArray(msg.files) ? msg.files : []
            for (const file of files) {
              const fileUrl = getOriginalFileUrl(file)
              if (!fileUrl) continue
              try {
                const resp = await fetch(fileUrl)
                if (!resp.ok) continue
                const body = await resp.text()
                if (body.trim()) {
                  finish(() => resolve(body.trim()))
                  return
                }
              } catch {
                // try next file
              }
            }
            finish(() => reject(new Error(t('translate.convertDocumentFailed', { msg: 'empty' }))))
          }
        })
      })
      .catch((err) => {
        finish(() => reject(err))
      })
  })
}

async function convertOfficeDocToText(url: string): Promise<string> {
  convertingDocument.value = true
  resetConvertProgress()
  convertStage.value = 'parse'
  startConvertElapsedTimer()
  try {
    const resp = await documentToText(url)
    const md = String(resp?.text || '').trim()
    if (!md) throw new Error(t('translate.convertDocumentFailed', { msg: 'empty' }))
    return md
  } finally {
    stopConvertElapsedTimer()
    convertingDocument.value = false
  }
}

function readFileAsText(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result || ''))
    reader.onerror = () => reject(reader.error || new Error('read failed'))
    reader.readAsText(file)
  })
}

async function ensureAttachmentUploaded(): Promise<string> {
  if (attachmentRemoteUrl.value) return attachmentRemoteUrl.value
  if (!attachmentFile.value) return ''
  attachmentUploading.value = true
  try {
    const url = await uploadFile(attachmentFile.value)
    attachmentRemoteUrl.value = url
    return url
  } finally {
    attachmentUploading.value = false
  }
}

async function resolveSourcePayload(): Promise<{ prompt?: string; tpl_list?: string[] }> {
  const text = sourceText.value.trim()

  if (attachmentFile.value && attachmentIsImage.value) {
    const url = await ensureAttachmentUploaded()
    if (!url) throw new Error(t('translate.emptySource'))
    return { tpl_list: [url], prompt: text || undefined }
  }

  if (attachmentFile.value && isPdfFile(attachmentFile.value)) {
    const url = await ensureAttachmentUploaded()
    if (!url) throw new Error(t('translate.emptySource'))
    const md = await convertPdfViaOcr(url)
    sourceText.value = md
    return { prompt: md }
  }

  if (attachmentFile.value && isOfficeDoc(attachmentFile.value)) {
    const url = await ensureAttachmentUploaded()
    if (!url) throw new Error(t('translate.emptySource'))
    const md = await convertOfficeDocToText(url)
    sourceText.value = md
    return { prompt: md }
  }

  if (attachmentFile.value && isPlainTextFile(attachmentFile.value)) {
    try {
      const content = (await readFileAsText(attachmentFile.value)).trim()
      if (content) {
        sourceText.value = content
        return { prompt: content }
      }
    } catch {
      throw new Error(t('translate.readFileFailed'))
    }
  }

  if (text) return { prompt: text }
  throw new Error(t('translate.emptySource'))
}

function disconnectWs() {
  if (wsHandle) {
    wsHandle.disconnect()
    wsHandle = null
  }
}

async function extractTranslateTextFromMessage(msg: TaskWsMessage): Promise<string> {
  const inline = String((msg as any)?.content || (msg as any)?.text || '').trim()
  if (inline) return inline
  const files = Array.isArray(msg.files) ? msg.files : []
  for (const file of files) {
    const url = getOriginalFileUrl(file)
    if (!url) continue
    try {
      const resp = await fetch(url)
      if (!resp.ok) continue
      const body = await resp.text()
      if (body.trim()) return body.trim()
    } catch {
      // try next file
    }
  }
  return ''
}

function isFinalTranslateResult(msg: TaskWsMessage): boolean {
  const msgType = String(msg.type || '').toLowerCase()
  if (msgType !== 'result') return false
  const status = String(msg.status || '').toLowerCase()
  const progress = Number(msg.progress)
  if (status === 'completed') return true
  return Number.isFinite(progress) && progress >= 100
}

function waitForTranslateTask(req: TaskCreateRequest): Promise<string> {
  return new Promise((resolve, reject) => {
    let settled = false
    const finish = (fn: () => void) => {
      if (settled) return
      settled = true
      disconnectWs()
      fn()
    }

    createTask(req)
      .then((created) => {
        const taskId = String(created.task_id || '').trim()
        if (!taskId) {
          finish(() => reject(new Error('missing task_id')))
          return
        }

        wsHandle = connectTaskWs(taskId, async (msg) => {
          const status = String(msg.status || '').toLowerCase()
          if (status === 'failed' || status === 'cancelled') {
            finish(() => reject(new Error(String(msg.error || status))))
            return
          }

          if (!isFinalTranslateResult(msg)) return

          const text = await extractTranslateTextFromMessage(msg)
          if (text) {
            finish(() => resolve(text))
          }
        })
      })
      .catch((err) => finish(() => reject(err)))
  })
}

async function translateTextInChunks(prompt: string, baseReq: TaskCreateRequest): Promise<string> {
  let chunks: string[]
  try {
    chunks = splitTextForTranslation(prompt)
  } catch (err: any) {
    throw new Error(err?.message || 'translate chunk split failed')
  }
  translateChunkTotal.value = chunks.length
  translateChunkIndex.value = 0

  if (chunks.length <= 1) {
    translateChunkTotal.value = 0
    return waitForTranslateTask({ ...baseReq, prompt: chunks[0] || prompt, tpl_list: undefined })
  }

  const parts: string[] = []
  for (let i = 0; i < chunks.length; i += 1) {
    translateChunkIndex.value = i + 1
    const part = await waitForTranslateTask({
      ...baseReq,
      prompt: chunks[i],
      tpl_list: undefined,
    })
    parts.push(part)
    outputText.value = joinTranslatedChunks(parts, chunks)
  }
  return joinTranslatedChunks(parts, chunks)
}

async function applyResultFromMessage(msg: TaskWsMessage) {
  const text = await extractTranslateTextFromMessage(msg)
  if (text) outputText.value = text
}

async function submitTranslate() {
  if (sourceLang.value === targetLang.value) {
    showTopSnack(t('translate.sameLanguage'))
    return
  }

  disconnectWs()
  outputText.value = ''
  translatingTask.value = true
  translateChunkIndex.value = 0
  translateChunkTotal.value = 0

  try {
    const payload = await resolveSourcePayload()
    const req: TaskCreateRequest = {
      task_type: 'translate',
      job_type: 'TRANSLATE',
      prompt: payload.prompt,
      tpl_list: payload.tpl_list,
      extract: {
        source_lang: sourceLang.value,
        target_lang: targetLang.value,
      },
    }
    const loadName = String(selectedLoadName.value || '').trim()
    if (loadName) req.load_name = loadName

    const prompt = String(payload.prompt || '').trim()
    const hasImage = Array.isArray(payload.tpl_list) && payload.tpl_list.length > 0

    if (prompt && !hasImage && countTranslateChunks(prompt) > 1) {
      const translated = await translateTextInChunks(prompt, req)
      outputText.value = translated
      translatingTask.value = false
      translateChunkIndex.value = 0
      translateChunkTotal.value = 0
      return
    }

    const created = await createTask(req)
    const taskId = String(created.task_id || '').trim()
    if (!taskId) throw new Error('missing task_id')

    wsHandle = connectTaskWs(taskId, async (msg) => {
      const status = String(msg.status || '').toLowerCase()
      if (status === 'failed' || status === 'cancelled') {
        showTopSnackError(resolveTaskWsError(msg))
        translatingTask.value = false
        disconnectWs()
        return
      }
      if (status === 'completed' || (msg.files && msg.files.length > 0)) {
        await applyResultFromMessage(msg)
        if (status === 'completed') {
          translatingTask.value = false
          disconnectWs()
        }
      }
    })
  } catch (e: unknown) {
    translatingTask.value = false
    translateChunkIndex.value = 0
    translateChunkTotal.value = 0
    disconnectWs()
    showTranslateError(e)
  }
}

async function copyOutput() {
  const text = outputText.value.trim()
  if (!text) return
  try {
    await navigator.clipboard.writeText(text)
    outputCopied.value = true
    if (copyFeedbackTimer) window.clearTimeout(copyFeedbackTimer)
    copyFeedbackTimer = window.setTimeout(() => {
      outputCopied.value = false
      copyFeedbackTimer = null
    }, COPY_FEEDBACK_MS)
  } catch {
    showTopSnackError(t('translate.copyFailed'))
  }
}

function revokePreview() {
  if (attachmentPreviewUrl.value) {
    URL.revokeObjectURL(attachmentPreviewUrl.value)
    attachmentPreviewUrl.value = ''
  }
}

function clearAttachment() {
  attachmentFile.value = null
  attachmentRemoteUrl.value = ''
  revokePreview()
}

async function applyAttachmentFile(file: File) {
  clearAttachment()
  attachmentFile.value = file
  if (file.type.startsWith('image/')) {
    attachmentPreviewUrl.value = URL.createObjectURL(file)
  }
  if (isPlainTextFile(file) && !isDocLike(file)) {
    try {
      sourceText.value = await readFileAsText(file)
    } catch {
      showTopSnackError(t('translate.readFileFailed'))
    }
  }
}

async function onFileInputChange(e: Event) {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (!file) return
  try {
    await applyAttachmentFile(file)
  } catch (err: unknown) {
    showTopSnackError(t('translate.uploadFailed', { msg: handleApiError(err).message }))
  }
}

function onDragOver() {
  dragDepth.value += 1
  dragOverlay.value = true
}

function onDragLeave() {
  dragDepth.value = Math.max(0, dragDepth.value - 1)
  if (dragDepth.value === 0) dragOverlay.value = false
}

async function onDrop(e: DragEvent) {
  dragDepth.value = 0
  dragOverlay.value = false
  const file = e.dataTransfer?.files?.[0]
  if (!file) return
  try {
    await applyAttachmentFile(file)
  } catch (err: unknown) {
    showTopSnackError(t('translate.uploadFailed', { msg: handleApiError(err).message }))
  }
}

watch(attachmentFile, (_, prev) => {
  if (prev && prev.type.startsWith('image/')) revokePreview()
})

onMounted(() => {
  void fetchTranslateModels()
})

onBeforeUnmount(() => {
  disconnectWs()
  disconnectConvertWs()
  stopConvertElapsedTimer()
  revokePreview()
  if (copyFeedbackTimer) window.clearTimeout(copyFeedbackTimer)
})
</script>
