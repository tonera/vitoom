<template>
  <Teleport to="body">
    <div
      v-if="open"
      class="fixed inset-0 z-9999 bg-black/70 flex items-center justify-center p-4"
      @click.self="emitClose()"
    >
      <div class="vt-card max-h-[min(85vh,920px)] w-full max-w-[900px] overflow-hidden">
        <div
          class="flex items-center justify-between gap-3 border-b border-gray-200 bg-gray-50 px-4 py-3 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40"
        >
          <div class="min-w-0">
            <div class="truncate text-sm font-semibold text-gray-900 [.dark_&]:text-white">{{ t('models.downloadTerminal.title') }}</div>
            <div class="truncate text-xs text-gray-500 [.dark_&]:text-gray-400">
              model_key: <span class="text-gray-800 [.dark_&]:text-gray-200">{{ modelKey }}</span>
              <span v-if="statusText" class="ml-2">| {{ statusText }}</span>
            </div>
            <div v-if="bytesTotal > 0" class="mt-2">
              <div
                class="h-1.5 w-full overflow-hidden rounded-full bg-gray-200 [.dark_&]:bg-gray-800"
              >
                <div
                  class="h-full bg-indigo-500/90"
                  :style="{ width: `${Math.max(0, Math.min(100, progressPct))}%` }"
                />
              </div>
            </div>
          </div>

          <div class="flex items-center gap-2">
            <button
              v-if="showRefresh"
              type="button"
              class="px-3 py-1.5 rounded-lg bg-sky-600/80 text-white text-xs hover:bg-sky-600 transition-colors cursor-pointer"
              @click="onRefresh()"
            >
              {{ t('common.refresh') }}
            </button>
            <button
              v-if="showRedownload"
              type="button"
              class="px-3 py-1.5 rounded-lg bg-indigo-600/80 text-white text-xs hover:bg-indigo-600 transition-colors cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
              :disabled="retrying"
              @click="onContinue()"
            >
              {{ retrying ? t('models.downloadTerminal.processing') : t('models.downloadTerminal.redownload') }}
            </button>
            <button
              v-if="showContinue"
              type="button"
              class="px-3 py-1.5 rounded-lg bg-indigo-600/80 text-white text-xs hover:bg-indigo-600 transition-colors cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
              :disabled="retrying"
              @click="onContinue()"
            >
              {{ retrying ? t('models.downloadTerminal.processing') : continueLabel }}
            </button>
            <button
              v-if="showCancel"
              type="button"
              class="px-3 py-1.5 rounded-lg bg-rose-600/80 text-white text-xs hover:bg-rose-600 transition-colors cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
              :disabled="canceling"
              @click="onCancel()"
            >
              {{ canceling ? t('models.downloadTerminal.canceling') : t('models.downloadTerminal.cancelDownload') }}
            </button>
            <button
              type="button"
              class="cursor-pointer rounded-lg p-2 text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-900 [.dark_&]:text-gray-400 [.dark_&]:hover:bg-gray-800/60 [.dark_&]:hover:text-white"
              @click="emitClose()"
              :aria-label="t('common.close')"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        <!-- Terminal -->
        <div class="bg-black text-gray-200 font-mono text-xs leading-relaxed">
          <div ref="terminalRef" class="h-[min(72vh,720px)] overflow-y-auto vt-scroll px-4 py-3">
            <div v-if="lines.length === 0" class="text-gray-500">
              {{ t('models.downloadTerminal.waitingWs') }}
            </div>
            <div v-for="(ln, idx) in lines" :key="`ln-${idx}`" class="whitespace-pre-wrap wrap-break-word">
              {{ ln }}
            </div>
          </div>
        </div>

        <div
          class="border-t border-gray-200 bg-gray-50 px-4 py-2 text-[11px] text-gray-500 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-400"
        >
          {{ t('models.downloadTerminal.footerHint') }}
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { connectModelDownload, type WebSocketMessage } from '../../../utils/websocket'
import { handleApiError } from '../../../utils/api'
import { downloadActionModel, type ModelRecord } from '../../../api/models'
import { formatBytes } from '../../../utils/format'

const { t } = useI18n()

const props = defineProps<{
  open: boolean
  modelKey: string
  model?: ModelRecord | null
  showCancel?: boolean
}>()

const emit = defineEmits<{
  (e: 'close'): void
  (e: 'status', payload: any): void
}>()

const terminalRef = ref<HTMLElement | null>(null)
const lines = ref<string[]>([])
const canceling = ref(false)
const retrying = ref(false)
const statusText = ref('')
const currentStatus = ref<string>('')
const lastActivityAt = ref<number>(Date.now())
const nowTs = ref<number>(Date.now())
const progressPct = ref<number>(0)
const bytesDownloaded = ref<number>(0)
const bytesTotal = ref<number>(0)

let wsClient: any = null
let heartbeatTimer: number | null = null
// 记录上一次打开时的 model_key，用于决定是否需要清空日志
let lastOpenedModelKey = ''

const showCancel = computed(() => Boolean(props.showCancel !== false))
function normStatus(v: any) {
  return String(v ?? '')
    .trim()
    .toLowerCase()
}

const showContinue = computed(() => {
  const st = normStatus(currentStatus.value || (props.model as any)?.download_status || '')
  return st === 'failed' || st === 'canceled' || st === 'pending'
})
const showRedownload = computed(() => {
  const st = normStatus(currentStatus.value || (props.model as any)?.download_status || '')
  return st === 'completed'
})
const showRefresh = computed(() => {
  const st = normStatus(currentStatus.value || (props.model as any)?.download_status || '')
  return st === 'downloading'
})
const continueLabel = computed(() => {
  const st = normStatus(currentStatus.value || (props.model as any)?.download_status || '')
  if (st === 'failed' || st === 'canceled') return t('models.downloadTerminal.continueDownload')
  if (st === 'pending') return t('models.downloadTerminal.startDownload')
  if (st === 'downloading') return t('models.downloadTerminal.retryDownload')
  return t('models.downloadTerminal.continueDownload')
})

function emitClose() {
  emit('close')
}

function stripAnsiAndControls(input: string) {
  // 去掉常见 ANSI 转义序列（例如 tqdm 的光标上移 \x1b[A）以及回车覆盖字符
  let s = String(input ?? '')
  s = s.replace(/\r/g, '')
  s = s.replace(/\x1B\[[0-?]*[ -/]*[@-~]/g, '') // CSI
  s = s.replace(/\x1B\][^\x07]*(\x07|\x1B\\)/g, '') // OSC (best-effort)
  return s
}

function formatProgressTextBytes(input: string) {
  const s = String(input ?? '').trim()
  if (!s) return s
  // e.g. "208187392/456485084 bytes (45%)" -> "198.54MB/435.34MB (45%)"
  return s.replace(/(\d{1,})\s*\/\s*(\d{1,})\s*bytes\b/gi, (_m, a, b) => `${formatBytes(Number(a))}/${formatBytes(Number(b))}`)
}

function pushLines(newLines: string[]) {
  if (!newLines.length) return
  // 清洗 + 过滤空行 + 连续重复去重（避免 tqdm/progress 输出刷屏）
  for (const raw of newLines) {
    const cleaned = stripAnsiAndControls(String(raw ?? '')).trimEnd()
    if (!cleaned) continue
    const last = lines.value.length ? lines.value[lines.value.length - 1] : ''
    if (cleaned === last) continue
    lines.value.push(cleaned)
  }
  if (lines.value.length > 500) {
    lines.value = lines.value.slice(lines.value.length - 500)
  }
  nextTick(() => {
    const el = terminalRef.value
    if (el) el.scrollTop = el.scrollHeight
  })
}

function formatStatusMessage(m: any) {
  const st = String(m?.status || '').trim()
  const p = formatProgressTextBytes(stripAnsiAndControls(String(m?.progress_text || '')).trim())
  const e = stripAnsiAndControls(String(m?.error_text || '')).trim()
  const loadName = String(m?.load_name || '').trim()
  const bd = Number((m as any)?.bytes_downloaded || 0)
  const bt = Number((m as any)?.bytes_total || 0)
  const pct = Number((m as any)?.progress || 0)
  bytesDownloaded.value = isFinite(bd) ? bd : 0
  bytesTotal.value = isFinite(bt) ? bt : 0
  progressPct.value = isFinite(pct) ? Math.max(0, Math.min(100, pct)) : 0
  currentStatus.value = st
  const progressBits = []
  if (progressPct.value > 0) progressBits.push(`${progressPct.value}%`)
  if (bytesTotal.value > 0) progressBits.push(`${formatBytes(bytesDownloaded.value)}/${formatBytes(bytesTotal.value)}`)
  statusText.value = [st, progressBits.join(' '), p].filter(Boolean).join(' | ')
  const parts = [
    `[status] status=${st || '-'}${p ? ` progress="${p}"` : ''}${loadName ? ` load_name="${loadName}"` : ''}${e ? ` error="${e}"` : ''}`,
  ]
  return parts
}

function handleWsMessage(msg: WebSocketMessage) {
  lastActivityAt.value = Date.now()
  const t = String((msg as any)?.type || '').trim()
  if (t === 'download_log') {
    const seq = (msg as any)?.seq
    const arr = Array.isArray((msg as any)?.lines) ? (msg as any).lines : []
    const prefix = typeof seq === 'number' ? `[log#${seq}] ` : '[log] '
    pushLines(arr.map((x: any) => prefix + String(x ?? '')))
    return
  }
  if (t === 'download_status') {
    pushLines(formatStatusMessage(msg))
    // 将实时状态抛给上层（用于刷新列表卡片的进度条/状态）
    try {
      emit('status', msg as any)
    } catch {}
    return
  }
  // fallback：打印原始
  pushLines([`[ws] ${JSON.stringify(msg)}`])
}

async function onCancel() {
  if (canceling.value) return
  const modelKey = String(props.modelKey || '').trim()
  if (!modelKey) return
  canceling.value = true
  try {
    await downloadActionModel(modelKey, { action: 'cancel' })
    pushLines([`[ui] ${t('models.downloadTerminal.uiCancelSent')}`])
  } catch (e: any) {
    const ae = handleApiError(e)
    pushLines([`[ui] ${t('models.downloadTerminal.uiCancelFailed', { msg: ae?.message || e?.message || String(e) })}`])
  } finally {
    canceling.value = false
  }
}

async function onContinue() {
  if (retrying.value) return
  const modelKey = String(props.modelKey || '').trim()
  if (!modelKey) return
  retrying.value = true
  try {
    // 先尝试 cancel 再触发 download，避免 worker 卡死/锁占用导致“重新触发也没反应”
    try {
      await downloadActionModel(modelKey, { action: 'cancel' })
      pushLines([`[ui] ${t('models.downloadTerminal.uiCancelForRetry')}`])
      await new Promise((r) => setTimeout(r, 400))
    } catch {}

    const source = ((props.model as any)?.source && typeof (props.model as any).source === 'object') ? (props.model as any).source : {}
    const assetType = String((props.model as any)?.asset_type || '').trim()
    const body: any = {}
    if (Object.keys(source).length) body.source = source
    if (assetType) body.asset_type = assetType
    await downloadActionModel(modelKey, { action: 'start', ...body })
    pushLines([`[ui] ${t('models.downloadTerminal.uiContinueTriggered', { source: source.provider || source.repo_id ? ` (${source.provider || ''}:${source.repo_id || ''})` : '' })}`])
  } catch (e: any) {
    const ae = handleApiError(e)
    pushLines([`[ui] ${t('models.downloadTerminal.uiContinueFailed', { msg: ae?.message || e?.message || String(e) })}`])
  } finally {
    retrying.value = false
  }
}

async function onRefresh() {
  const modelKey = String(props.modelKey || '').trim()
  if (!modelKey) return
  try {
    await downloadActionModel(modelKey, { action: 'refresh' })
    pushLines([`[ui] ${t('models.downloadTerminal.uiRefreshSent')}`])
  } catch (e: any) {
    const ae = handleApiError(e)
    pushLines([`[ui] ${t('models.downloadTerminal.uiRefreshFailed', { msg: ae?.message || e?.message || String(e) })}`])
  }
}

function connectWs() {
  const modelKey = String(props.modelKey || '').trim()
  if (!modelKey) return
  wsClient = connectModelDownload(modelKey, handleWsMessage)
  // 心跳：防止中间层因长时间无帧关闭连接（服务端会忽略任意文本）
  if (heartbeatTimer) window.clearInterval(heartbeatTimer)
  heartbeatTimer = window.setInterval(() => {
    try {
      wsClient?.send?.('ping')
    } catch {}
  }, 20000)
}

function disconnectWs() {
  try {
    wsClient?.disconnect?.()
  } catch {}
  wsClient = null
  if (heartbeatTimer) {
    window.clearInterval(heartbeatTimer)
    heartbeatTimer = null
  }
}

watch(
  () => [props.open, props.modelKey] as const,
  ([open, modelKey]) => {
    if (!open) {
      disconnectWs()
      return
    }
    const key = String(modelKey || '').trim()
    // 只有切换到新的 model_key 时才清空；同一个 model 关闭再打开保留本地已接收日志
    if (key && key !== lastOpenedModelKey) {
      lines.value = []
      statusText.value = ''
      currentStatus.value = ''
    }
    currentStatus.value = normStatus((props.model as any)?.download_status || '')
    lastActivityAt.value = Date.now()
    if (key) lastOpenedModelKey = key
    connectWs()
  },
  { immediate: true }
)

onMounted(() => {
  const t = window.setInterval(() => {
    nowTs.value = Date.now()
  }, 500)
  onBeforeUnmount(() => window.clearInterval(t))
})

onBeforeUnmount(() => {
  disconnectWs()
})
</script>

