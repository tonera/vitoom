import { computed, onBeforeUnmount, ref, shallowRef } from 'vue'
import type { ChatMessageRecord, ChatSession, ChatWsServerEvent } from '../api/chat'
import {
  getChatSession,
  isSessionForbiddenError,
  isSessionNotFoundError,
  listChatMessages,
  refreshAccessTokenForWs,
  upsertRecentChatSessionFromSession,
} from '../api/chat'
import { getAccessToken } from '../utils/auth'
import {
  createWebSocketClient,
  type WebSocketBinaryHandler,
  type WebSocketCloseInfo,
  WebSocketClient,
} from '../utils/websocket'
import { currentLocaleTag, translate } from '../utils/translate'

export type ChatRole = 'user' | 'assistant' | 'transcript' | 'tool'
export type ChatContentType = 'text' | 'audio'

export interface ChatArtifact {
  fileId: string
  url: string
  mime: string
  category: string
  name?: string
  size?: number
  taskId?: string
  status?: 'pending' | 'completed' | 'failed'
  progress?: number | null
}

export interface ChatMessage {
  id: string
  role: ChatRole
  contentType: ChatContentType
  text: string
  streaming: boolean
  interruptReason?: string | null
  artifacts?: ChatArtifact[]
  audioChunks?: Array<{ bytes: ArrayBuffer; mime: string; isFinal: boolean; sampleRate?: number | null }>
  usage?: Record<string, number>
  turnId?: string | null
  runId?: string | null
  createdAt: number
  /** 是否已收到过 message_delta（用于 typing 指示） */
  hasAssistantDelta?: boolean
  /** 不可恢复错误气泡 */
  isError?: boolean
}

export type ActivityLevel = 'info' | 'success' | 'warn' | 'progress' | 'error'

export interface ActivityItem {
  id: string
  level: ActivityLevel
  title: string
  detail?: string
  /** 完整详情（列表里的 detail 可能被截断；详情弹窗优先用此字段） */
  detailFull?: string
  time: string
  progress?: number | null
}

export type ChatPhase = 'idle' | 'connecting' | 'opening' | 'ready' | 'closed' | 'error'

/** P1 PR-3：audio_delta 下行分片（PCM binary），供播放器订阅 */
export interface AudioDeltaEvent {
  bytes: ArrayBuffer
  mime: string
  isFinal: boolean
  runId: string | null
  turnId: string | null
  /** 后端 payload.sample_rate（可选；WAV 自解析优先用 header，PCM 按此兜底） */
  sampleRate?: number | null
}

export type AudioCancelReason =
  | 'interrupt-sent'
  | 'status-interrupted'
  | 'status-failed'
  | 'session-closed'
  | 'audio-auto-commit'
  | 'user-speech-start'
  | 'disconnect'

export interface AudioCancelEvent {
  reason: AudioCancelReason
  runId?: string | null
}

export type AudioChunkPurpose = 'user_turn' | 'barge_in_probe'

const ACTIVITY_MAX = 100
const SESSION_READY_MS = 5000
/** 与后端 list_messages 默认上限对齐，用于判断是否还有更早消息 */
const HISTORY_PAGE = 200
const AUDIO_UPLINK_STATES = new Set(['ready', 'turn_buffering', 'reasoning', 'tool_running', 'streaming_output', 'waiting_task'])
const SESSION_STATE_KEYS = new Set([
  'opening',
  'ready',
  'turn_buffering',
  'reasoning',
  'tool_running',
  'streaming_output',
  'waiting_task',
  'completed',
  'interrupted',
  'failed',
  'closed',
])

function formatClock(iso?: string): string {
  const d = iso ? new Date(iso) : new Date()
  const locale = currentLocaleTag()
  if (Number.isNaN(d.getTime())) {
    return new Date().toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }
  return d.toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

function formatSessionState(state?: string): string {
  const key = String(state ?? '').trim()
  if (!key) return translate('agents.chatSession.processing')
  if (SESSION_STATE_KEYS.has(key)) {
    return translate(`agents.chatSession.sessionState.${key}`)
  }
  return key
}

function coercePercent(raw: unknown): number | undefined {
  if (typeof raw !== 'number' || !Number.isFinite(raw)) return undefined
  return Math.max(0, Math.min(100, Math.round(raw)))
}

function buildStatusActivity(payload: Record<string, unknown>): Omit<ActivityItem, 'id' | 'time'> | null {
  const state = String(payload.state ?? '').trim()
  if (!state) return null

  const taskKind = String(payload.task_kind ?? '').trim()
  const taskStatus = String(payload.task_status ?? '').trim()
  const progress = coercePercent(payload.progress)
  const filesCount = typeof payload.files_count === 'number' ? payload.files_count : undefined
  const total = typeof payload.total === 'number' ? payload.total : undefined
  const taskId = String(payload.task_id ?? '').trim()
  const error = String(payload.error ?? '').trim()

  if (taskKind || taskStatus || progress !== undefined || filesCount !== undefined || total !== undefined || taskId) {
    const parts: string[] = []
    if (taskStatus) parts.push(translate('agents.chatSession.activity.taskStatus', { status: taskStatus }))
    if (progress !== undefined) parts.push(`${progress}%`)
    if (filesCount !== undefined) {
      parts.push(
        total && total > 0
          ? translate('agents.chatSession.activity.resultTotal', { count: filesCount, total })
          : translate('agents.chatSession.activity.resultCount', { count: filesCount }),
      )
    }
    if (error) parts.push(error)
    return {
      level: error || taskStatus === 'failed' ? 'error' : taskStatus === 'completed' ? 'success' : 'progress',
      title: translate('agents.chatSession.activity.taskProgress', {
        kind: taskKind || formatSessionState(state),
      }),
      detail: parts.join(' · ') || undefined,
      progress: progress ?? (taskStatus === 'completed' ? 100 : state === 'waiting_task' ? 45 : undefined),
    }
  }

  if (state === 'failed' || state === 'interrupted') {
    return {
      level: state === 'failed' ? 'error' : 'warn',
      title: translate('agents.chatSession.activity.status', { state: formatSessionState(state) }),
      detail: String(payload.prev ?? ''),
    }
  }

  if (state === 'reasoning' || state === 'tool_running' || state === 'waiting_task' || state === 'streaming_output') {
    const prev = String(payload.prev ?? '').trim()
    return {
      level: 'progress',
      title: translate('agents.chatSession.activity.phase', { state: formatSessionState(state) }),
      detail: prev
        ? translate('agents.chatSession.activity.prevPhase', { prev: formatSessionState(prev) })
        : undefined,
      progress:
        state === 'reasoning'
          ? 12
          : state === 'tool_running'
            ? 28
            : state === 'waiting_task'
              ? 52
              : 84,
    }
  }

  return null
}

function mapArtifactFromPayload(p: Record<string, unknown>): ChatArtifact {
  const rawUrl = String(p.url ?? p.http_url ?? '').trim()
  const rawName = p.file_name != null ? String(p.file_name) : p.name != null ? String(p.name) : ''
  const fileId = String(p.file_id ?? p.fileId ?? '').trim() || rawUrl || rawName || 'unknown'
  let url = rawUrl
  if (rawUrl.startsWith('/') && typeof window !== 'undefined') {
    url = `${window.location.origin}${rawUrl}`
  }
  return {
    fileId,
    url,
    mime: String(p.mime ?? p.mime_type ?? 'application/octet-stream'),
    category: String(p.category ?? 'file'),
    name: rawName || undefined,
    size: typeof p.file_size === 'number' ? p.file_size : undefined,
    taskId: String(p.derived_task_id ?? p.task_id ?? '').trim() || undefined,
    status: String(p.status ?? '').trim() === 'pending'
      ? 'pending'
      : String(p.status ?? '').trim() === 'failed'
        ? 'failed'
        : undefined,
    progress: typeof p.progress === 'number' ? p.progress : undefined,
  }
}

function dedupeArtifacts(items: ChatArtifact[]): ChatArtifact[] {
  const seen = new Set<string>()
  const result: ChatArtifact[] = []
  for (const item of items) {
    const key = item.fileId || item.url
    if (!key || seen.has(key)) continue
    seen.add(key)
    result.push(item)
  }
  return result
}

function pendingAudioArtifact(taskId: string, progress?: number | null): ChatArtifact {
  return {
    fileId: `pending-audio-${taskId}`,
    url: '',
    mime: 'audio/wav',
    category: 'audio',
    name: translate('agents.artifacts.audioGenerating'),
    taskId,
    status: 'pending',
    progress: progress ?? null,
  }
}

function mapArtifactsFromFilesPayload(payload: Record<string, unknown>): ChatArtifact[] {
  const raw = payload.files
  if (!Array.isArray(raw)) return []
  return raw
    .map((item) => (item && typeof item === 'object' ? mapArtifactFromPayload(item as Record<string, unknown>) : null))
    .filter((item): item is ChatArtifact => Boolean(item && (item.url || item.fileId)))
}

function recordToChatMessage(r: ChatMessageRecord): ChatMessage | null {
  const role = String(r.role || '').toLowerCase()
  const createdAt = r.created_at ? Date.parse(r.created_at) : Date.now()
  if (role === 'user') {
    return {
      id: r.id,
      role: 'user',
      contentType: 'text',
      text: String(r.content ?? ''),
      streaming: false,
      turnId: r.turn_id ?? null,
      runId: r.agent_run_id ?? null,
      createdAt: Number.isFinite(createdAt) ? createdAt : Date.now(),
    }
  }
  if (role === 'assistant') {
    const runKey = String(r.agent_run_id || r.id)
    const meta = (r.metadata || {}) as Record<string, unknown>
    const persisted = mapArtifactsFromFilesPayload({ files: meta.files })
    const artifacts = persisted.length ? dedupeArtifacts(persisted) : undefined
    return {
      id: runKey,
      role: 'assistant',
      contentType: 'text',
      text: String(r.content ?? ''),
      streaming: false,
      turnId: r.turn_id ?? null,
      runId: r.agent_run_id ?? null,
      createdAt: Number.isFinite(createdAt) ? createdAt : Date.now(),
      ...(artifacts?.length ? { artifacts } : {}),
    }
  }
  if (role === 'tool' || role === 'system') {
    return {
      id: r.id,
      role: 'tool',
      contentType: 'text',
      text: String(r.content ?? ''),
      streaming: false,
      createdAt: Number.isFinite(createdAt) ? createdAt : Date.now(),
    }
  }
  return null
}

function readSessionModes(session: ChatSession): { input_mode: string; output_mode: string } {
  const md = (session.metadata || {}) as Record<string, unknown>
  return {
    input_mode: String(md.input_mode ?? 'text'),
    output_mode: String(md.output_mode ?? 'text_stream'),
  }
}

function modeSupportsAudioInput(inputMode: string): boolean {
  const mode = String(inputMode || '').trim().toLowerCase()
  return mode === 'audio_once' || mode === 'audio_stream' || mode === 'mixed'
}

function modeSupportsAudioOutput(outputMode: string): boolean {
  const mode = String(outputMode || '').trim().toLowerCase()
  return mode === 'audio_once' || mode === 'audio_stream' || mode === 'multimodal' || mode === 'multimodal_result'
}

export function useAgentChatSession() {
  const phase = ref<ChatPhase>('idle')
  const messages = ref<ChatMessage[]>([])
  const activity = ref<ActivityItem[]>([])
  const sessionState = ref<string | null>(null)
  const currentStatus = ref<string | null>(null)
  const errorMessage = ref<string | null>(null)
  const recoverableToast = ref<string | null>(null)

  const sessionId = ref<string | null>(null)
  const sessionModes = ref({ input_mode: 'text', output_mode: 'text_stream' })
  const sessionCapabilities = ref({
    supportsAudioInput: false,
    supportsAudioOutput: false,
    supportsToolArtifacts: true,
  })
  const wsClient = shallowRef<WebSocketClient | null>(null)

  // 数字人 enable 意图持久化。
  // 必要性：用户进 chat 页面后立即点"启用数字人"时，chat WS 可能仍在 connectPath，
  // wsClient.isConnected=false 会让 sendAvatarToggle 静默丢；之后即使 WS 连上，
  // sendAvatarToggle 也不会自动补发，后端 LiveTalkingClient 的 _sessions[sid]
  // 永远建不起来 → push_pcm 全部 short-circuit → 数字人有视频但无口型，三方
  // 无日志。这里把"用户最后一次表达的 enable 意图"挂到模块作用域，session_ready
  // 到达时自动 flush（set_enabled 是幂等的，重复发无副作用）。
  // null = 用户从未表过态，不需要 sync。
  let lastDesiredAvatarEnabled: boolean | null = null

  const artifactBuffer = new Map<string, ChatArtifact[]>()

  /** 当前已加载窗口里“最早一条消息”对应的远端 offset（升序列表语义） */
  const historyOldestOffset = ref(0)
  const historyHasMore = ref(false)
  const historyLoadingMore = ref(false)

  let readyTimer: number | undefined
  let removeCloseListener: (() => void) | undefined
  let removeMsgListener: (() => void) | undefined
  let removeBinaryListener: (() => void) | undefined
  let lastStatusActivityKey = ''
  const dramaAudioStatusNotes = new Set<string>()

  type PendingAudioDeltaMeta = {
    runId: string | null
    turnId: string | null
    mime: string
    isFinal: boolean
    sampleRate: number | null
    assistantIdx: number
  }
  const pendingAudioDeltaMeta: PendingAudioDeltaMeta[] = []
  let authFailStreak = 0

  // P1 PR-3：audio_delta / audio-cancel 事件发布订阅
  const audioDeltaListeners = new Set<(ev: AudioDeltaEvent) => void>()
  const audioCancelListeners = new Set<(ev: AudioCancelEvent) => void>()
  const onAudioDelta = (cb: (ev: AudioDeltaEvent) => void): (() => void) => {
    audioDeltaListeners.add(cb)
    return () => void audioDeltaListeners.delete(cb)
  }
  const onAudioCancel = (cb: (ev: AudioCancelEvent) => void): (() => void) => {
    audioCancelListeners.add(cb)
    return () => void audioCancelListeners.delete(cb)
  }
  const emitAudioDelta = (ev: AudioDeltaEvent) => {
    for (const cb of audioDeltaListeners) {
      try {
        cb(ev)
      } catch (err) {
        console.error('[useAgentChatSession] audio-delta listener error:', err)
      }
    }
  }
  const emitAudioCancel = (ev: AudioCancelEvent) => {
    for (const cb of audioCancelListeners) {
      try {
        cb(ev)
      } catch (err) {
        console.error('[useAgentChatSession] audio-cancel listener error:', err)
      }
    }
  }

  const isStreaming = computed(() => messages.value.some((m) => m.role === 'assistant' && m.streaming))
  const canSubmit = computed(() => phase.value === 'ready' && sessionState.value === 'ready' && !isStreaming.value)

  const pushActivity = (item: Omit<ActivityItem, 'id' | 'time'> & { id?: string; time?: string }) => {
    const row: ActivityItem = {
      id: item.id ?? `act-${Date.now()}-${Math.random().toString(16).slice(2)}`,
      level: item.level,
      title: item.title,
      detail: item.detail,
      detailFull: item.detailFull,
      time: item.time ?? formatClock(),
      progress: item.progress,
    }
    activity.value = [row, ...activity.value].slice(0, ACTIVITY_MAX)
  }

  const clearReadyTimer = () => {
    if (readyTimer) {
      clearTimeout(readyTimer)
      readyTimer = undefined
    }
  }

  const resetSessionState = () => {
    messages.value = []
    activity.value = []
    sessionState.value = null
    currentStatus.value = null
    errorMessage.value = null
    recoverableToast.value = null
    sessionCapabilities.value = {
      supportsAudioInput: false,
      supportsAudioOutput: false,
      supportsToolArtifacts: true,
    }
    historyOldestOffset.value = 0
    historyHasMore.value = false
    historyLoadingMore.value = false
    artifactBuffer.clear()
    dramaAudioStatusNotes.clear()
    lastStatusActivityKey = ''
  }

  const applySessionCapabilities = (payload: Record<string, unknown>) => {
    const rawCaps =
      payload.capabilities && typeof payload.capabilities === 'object'
        ? (payload.capabilities as Record<string, unknown>)
        : null
    sessionCapabilities.value = {
      supportsAudioInput: rawCaps ? Boolean(rawCaps.supports_audio_input) : modeSupportsAudioInput(sessionModes.value.input_mode),
      supportsAudioOutput: rawCaps
        ? Boolean(rawCaps.supports_audio_output)
        : modeSupportsAudioOutput(sessionModes.value.output_mode),
      supportsToolArtifacts: rawCaps ? Boolean(rawCaps.supports_tool_artifacts ?? true) : true,
    }
  }

  const mergeBufferedArtifacts = (runId: string) => {
    const pending = artifactBuffer.get(runId)
    if (!pending?.length) return
    const idx = messages.value.findIndex((m) => m.id === runId && m.role === 'assistant')
    if (idx < 0) return
    const cur = messages.value[idx]!
    const nextArts = dedupeArtifacts([...(cur.artifacts ?? []), ...pending])
    const next: ChatMessage = { ...cur, artifacts: nextArts }
    messages.value.splice(idx, 1, next)
    artifactBuffer.delete(runId)
  }

  const ensureAssistantMessage = (runId: string | null | undefined): number => {
    const rid = String(runId || '').trim()
    if (!rid) return -1
    let idx = messages.value.findIndex((m) => m.id === rid && m.role === 'assistant')
    if (idx >= 0) return idx
    messages.value.push({
      id: rid,
      role: 'assistant',
      contentType: 'text',
      text: '',
      streaming: true,
      runId: rid,
      createdAt: Date.now(),
    })
    idx = messages.value.length - 1
    mergeBufferedArtifacts(rid)
    return idx
  }

  const appendArtifactToRun = (runId: string | null | undefined, art: ChatArtifact) => {
    const rid = String(runId || '').trim()
    if (!rid) return
    const idx = messages.value.findIndex((m) => m.id === rid && m.role === 'assistant')
    if (idx < 0) {
      const existing = artifactBuffer.get(rid) ?? []
      const arr =
        art.status !== 'pending' && art.taskId
          ? existing.filter((item) => !(item.status === 'pending' && item.taskId === art.taskId))
          : existing
      arr.push(art)
      artifactBuffer.set(rid, arr)
      return
    }
    const cur = messages.value[idx]!
    const existing = cur.artifacts ?? []
    const withoutMatchingPending =
      art.status !== 'pending' && art.taskId
        ? existing.filter((item) => !(item.status === 'pending' && item.taskId === art.taskId))
        : existing
    const nextArts = dedupeArtifacts([...withoutMatchingPending, art])
    const next: ChatMessage = { ...cur, artifacts: nextArts }
    messages.value.splice(idx, 1, next)
  }

  const appendAssistantStatusNote = (runId: string | null | undefined, key: string, text: string) => {
    const rid = String(runId || '').trim()
    if (!rid || !text || dramaAudioStatusNotes.has(key)) return
    const idx = ensureAssistantMessage(rid)
    if (idx < 0) return
    const cur = messages.value[idx]!
    const separator = cur.text.trim() ? '\n\n' : ''
    const next: ChatMessage = {
      ...cur,
      text: `${cur.text}${separator}${text}`,
      hasAssistantDelta: true,
    }
    messages.value.splice(idx, 1, next)
    dramaAudioStatusNotes.add(key)
  }

  const findActiveAssistantIndex = (runId: string | null | undefined): number => {
    const rid = String(runId || '').trim()
    if (rid) {
      const byRun = messages.value.findIndex((m) => m.id === rid && m.role === 'assistant' && m.streaming)
      if (byRun >= 0) return byRun
    }
    for (let i = messages.value.length - 1; i >= 0; i -= 1) {
      const msg = messages.value[i]
      if (msg?.role === 'assistant' && msg.streaming) return i
    }
    return -1
  }

  const finalizeActiveAssistantMessage = (
    runId: string | null | undefined,
    fallbackText: string,
    options: { isError?: boolean; interruptReason?: string | null } = {},
  ): boolean => {
    const idx = findActiveAssistantIndex(runId)
    if (idx < 0) return false

    const cur = messages.value[idx]!
    const text = String(fallbackText || '').trim()
    const nextText =
      text && String(cur.text || '').trim()
        ? `${cur.text.trimEnd()}\n\n${text}`
        : text || cur.text
    const next: ChatMessage = {
      ...cur,
      text: nextText,
      streaming: false,
      interruptReason: options.interruptReason ?? cur.interruptReason ?? null,
      isError: options.isError ?? cur.isError,
    }
    messages.value.splice(idx, 1, next)
    return true
  }

  const handleWsPayload = (ev: ChatWsServerEvent) => {
    const t = ev.type
    const payload = (ev as { payload?: Record<string, unknown> }).payload ?? {}

    switch (t) {
      case 'session_ready': {
        applySessionCapabilities(payload)
        clearReadyTimer()
        sessionState.value = 'ready'
        currentStatus.value = formatSessionState('ready')
        phase.value = 'ready'
        authFailStreak = 0
        // 数字人意图补发：覆盖"用户进页面瞬间就 enable，但 chat WS 还没握手
        // 完成"的 race；以及 chat WS 中途断了又连后再次同步。set_enabled 在
        // 后端是幂等的，重复发无副作用。
        flushAvatarToggle()
        break
      }
      case 'capabilities_changed': {
        applySessionCapabilities(payload)
        break
      }
      case 'status_changed': {
        const state = String(payload.state ?? '').trim()
        if (state) {
          sessionState.value = state
          currentStatus.value = formatSessionState(state)
        }
        if (payload.auto_commit === true) {
          emitAudioCancel({
            reason: 'audio-auto-commit',
            runId: (ev as { run_id?: string | null }).run_id ?? null,
          })
        }
        // 后端 VAD 检测到用户开口（无论是真正的 barge-in 还是 ready→turn_buffering 的新一轮），
        // 都立即把本地 useAudioPlayback 队列里残留的 TTS 取消掉。
        // 否则会出现"后端 state 已切走、前端还在念旧回复"的情况——典型场景：用户说"停"
        // 想打断，但后端早就 complete_run 进入 ready，仅靠 status_changed=interrupted 路径无法触达。
        if (payload.vad_state === 'speech_start') {
          emitAudioCancel({
            reason: 'user-speech-start',
            runId: (ev as { run_id?: string | null }).run_id ?? null,
          })
        }
        const runId = (ev as { run_id?: string | null }).run_id ?? null
        const taskKind = String(payload.task_kind ?? '').trim()
        const taskStatus = String(payload.task_status ?? '').trim().toLowerCase()
        const taskId = String(payload.task_id ?? '').trim()
        if (taskKind === 'audio-drama-tts' && taskId) {
          if (taskStatus && taskStatus !== 'completed' && taskStatus !== 'failed') {
            appendArtifactToRun(runId, pendingAudioArtifact(taskId, coercePercent(payload.progress) ?? null))
            appendAssistantStatusNote(
              runId,
              `audio-drama-started-${taskId}`,
              translate('agents.chatSession.audioDrama.synthesizing'),
            )
          } else if (taskStatus === 'completed') {
            appendAssistantStatusNote(
              runId,
              `audio-drama-completed-${taskId}`,
              translate('agents.chatSession.audioDrama.ready'),
            )
          }
        }
        const statusActivity = buildStatusActivity(payload)
        if (statusActivity) {
          const key = JSON.stringify({
            state,
            prev: payload.prev ?? null,
            taskId: payload.task_id ?? null,
            taskStatus: payload.task_status ?? null,
            progress: payload.progress ?? null,
            filesCount: payload.files_count ?? null,
            total: payload.total ?? null,
            error: payload.error ?? null,
          })
          if (key !== lastStatusActivityKey) {
            lastStatusActivityKey = key
            pushActivity({
              ...statusActivity,
              detailFull: statusActivity.detail,
              time: formatClock((ev as { server_ts?: string }).server_ts),
            })
          }
        }
        if (state === 'failed' || state === 'interrupted') {
          const statusError = String(payload.error ?? '').trim()
          finalizeActiveAssistantMessage(
            (ev as { run_id?: string | null }).run_id ?? null,
            statusError || (state === 'failed' ? translate('agents.chatSession.runFailed') : translate('agents.chatSession.interrupted')),
            {
              isError: state === 'failed',
              interruptReason: state === 'interrupted' ? 'interrupted' : null,
            },
          )
          emitAudioCancel({
            reason: state === 'failed' ? 'status-failed' : 'status-interrupted',
            runId: (ev as { run_id?: string | null }).run_id ?? null,
          })
        }
        break
      }
      case 'message_started': {
        const runId = (ev as { run_id?: string | null }).run_id
        const role = String(payload.role ?? 'assistant').toLowerCase()
        if (role === 'transcript') {
          const rid = String(runId || `tr-${Date.now()}`)
          messages.value.push({
            id: rid,
            role: 'transcript',
            contentType: 'text',
            text: '',
            streaming: true,
            runId: runId ?? null,
            createdAt: Date.now(),
          })
          break
        }
        ensureAssistantMessage(runId)
        break
      }
      case 'message_delta': {
        const runId = (ev as { run_id?: string | null }).run_id
        const role = String(payload.role ?? 'assistant').toLowerCase()
        const delta = String(payload.delta ?? payload.text ?? '')
        if (role === 'transcript') {
          const rid = runId ?? null
          const byRun =
            rid != null && rid !== ''
              ? messages.value.findIndex((m) => m.role === 'transcript' && m.streaming && m.runId === rid)
              : -1
          let tidx = byRun >= 0 ? byRun : messages.value.findIndex((m) => m.role === 'transcript' && m.streaming)
          if (tidx < 0) {
            const idBase = rid ? `tr-${rid}` : `tr-${Date.now()}`
            messages.value.push({
              id: idBase,
              role: 'transcript',
              contentType: 'text',
              text: '',
              streaming: true,
              runId: rid,
              createdAt: Date.now(),
            })
            tidx = messages.value.length - 1
          }
          const cur = messages.value[tidx]!
          const next: ChatMessage = { ...cur, text: cur.text + delta, hasAssistantDelta: true }
          messages.value.splice(tidx, 1, next)
          break
        }
        const idx = ensureAssistantMessage(runId)
        if (idx < 0) break
        const cur = messages.value[idx]!
        const next: ChatMessage = { ...cur, text: cur.text + delta, hasAssistantDelta: true }
        messages.value.splice(idx, 1, next)
        break
      }
      case 'message_completed': {
        const runId = (ev as { run_id?: string | null }).run_id
        const role = String(payload.role ?? 'assistant').toLowerCase()
        const full = String(payload.content ?? '')
        const payloadArtifacts = mapArtifactsFromFilesPayload(payload)
        const interruptReason =
          payload.interrupt_reason != null && payload.interrupt_reason !== undefined
            ? String(payload.interrupt_reason)
            : null
        if (role === 'transcript') {
          const rid = runId ?? null
          const byRun =
            rid != null && rid !== ''
              ? messages.value.findIndex((m) => m.role === 'transcript' && m.streaming && m.runId === rid)
              : -1
          const tidx = byRun >= 0 ? byRun : messages.value.findIndex((m) => m.role === 'transcript' && m.streaming)
          if (tidx >= 0) {
            const cur = messages.value[tidx]!
            const next: ChatMessage = {
              ...cur,
              text: full || cur.text,
              streaming: false,
              interruptReason,
            }
            messages.value.splice(tidx, 1, next)
          } else if (full.trim()) {
            messages.value.push({
              id: `tr-${rid || Date.now()}`,
              role: 'transcript',
              contentType: 'text',
              text: full,
              streaming: false,
              interruptReason,
              runId: rid,
              createdAt: Date.now(),
            })
          }
          break
        }
        const rid = String(runId || '').trim()
        let idx = rid ? messages.value.findIndex((m) => m.id === rid && m.role === 'assistant') : -1
        if (idx < 0 && rid) {
          messages.value.push({
            id: rid,
            role: 'assistant',
            contentType: 'text',
            text: full,
            streaming: false,
            interruptReason,
            runId,
            artifacts: payloadArtifacts.length ? payloadArtifacts : undefined,
            usage:
              payload.usage_metrics && typeof payload.usage_metrics === 'object'
                ? (payload.usage_metrics as Record<string, number>)
                : undefined,
            createdAt: Date.now(),
          })
          mergeBufferedArtifacts(rid)
          break
        }
        if (idx >= 0) {
          const cur = messages.value[idx]!
          const next: ChatMessage = {
            ...cur,
            text: full || cur.text,
            streaming: false,
            interruptReason,
            artifacts: payloadArtifacts.length
              ? dedupeArtifacts([...(cur.artifacts ?? []), ...payloadArtifacts])
              : cur.artifacts,
            usage:
              payload.usage_metrics && typeof payload.usage_metrics === 'object'
                ? (payload.usage_metrics as Record<string, number>)
                : cur.usage,
          }
          messages.value.splice(idx, 1, next)
        }
        break
      }
      case 'tool_call_started': {
        const parent = String((payload as { parent_crew_tool?: unknown }).parent_crew_tool ?? '').trim()
        const tool = String(payload.tool_name ?? 'tool')
        const label = parent ? `${parent} › ${tool}` : tool
        const ap = (payload as { args_preview?: unknown }).args_preview
        const detailRaw =
          ap !== undefined && ap !== null
            ? typeof ap === 'string'
              ? ap
              : JSON.stringify(ap)
            : String((payload as { arguments?: unknown }).arguments ?? '').trim()
        pushActivity({
          level: 'progress',
          title: translate('agents.chatSession.activity.toolStart', { label }),
          detail: detailRaw ? detailRaw.slice(0, 220) : undefined,
          detailFull: detailRaw || undefined,
          time: formatClock((ev as { server_ts?: string }).server_ts),
          progress: 8,
        })
        break
      }
      case 'tool_call_completed': {
        const parent = String((payload as { parent_crew_tool?: unknown }).parent_crew_tool ?? '').trim()
        const tool = String(payload.tool_name ?? 'tool')
        const label = parent ? `${parent} › ${tool}` : tool
        const dur =
          typeof (payload as { duration_ms?: unknown }).duration_ms === 'number'
            ? `${(payload as { duration_ms: number }).duration_ms} ms`
            : undefined
        const out = String((payload as { output?: unknown }).output ?? '').trim()
        const detail = dur && out ? `${dur} · ${out.slice(0, 140)}` : dur || (out ? out.slice(0, 180) : undefined)
        const detailFullParts = [dur, out].filter((s) => Boolean(s && String(s).trim()))
        pushActivity({
          level: 'success',
          title: translate('agents.chatSession.activity.toolDone', { label }),
          detail,
          detailFull: detailFullParts.length ? detailFullParts.join('\n') : undefined,
          time: formatClock((ev as { server_ts?: string }).server_ts),
          progress: 100,
        })
        break
      }
      case 'tool_call_failed': {
        const parent = String((payload as { parent_crew_tool?: unknown }).parent_crew_tool ?? '').trim()
        const tool = String(payload.tool_name ?? 'tool')
        const label = parent ? `${parent} › ${tool}` : tool
        const errMsg = String(payload.error ?? '')
        pushActivity({
          level: 'error',
          title: translate('agents.chatSession.activity.toolFailed', { label }),
          detail: errMsg,
          detailFull: errMsg || undefined,
          time: formatClock((ev as { server_ts?: string }).server_ts),
        })
        break
      }
      case 'artifact_created': {
        const art = mapArtifactFromPayload(payload)
        appendArtifactToRun((ev as { run_id?: string | null }).run_id, art)
        const artLabel = art.name || art.fileId
        pushActivity({
          level: 'info',
          title: translate('agents.chatSession.activity.artifact', { category: art.category }),
          detail: artLabel,
          detailFull: [art.name, art.fileId, art.url].filter(Boolean).join('\n') || artLabel,
          time: formatClock((ev as { server_ts?: string }).server_ts),
        })
        break
      }
      case 'error': {
        const runId = (ev as { run_id?: string | null }).run_id ?? null
        const code = String(payload.code ?? 'error')
        const msg = String(payload.message ?? translate('common.unknownError'))
        const recoverable = Boolean(payload.recoverable)
        const isRecoverableTtsWarning = recoverable && (code === 'tts_synthesize_failed' || code === 'tts_failed')
        // 所有 error code 一视同仁推 activity 面板。前端**不**按 code 做业务
        // 过滤——如果某个 code 不该让用户看到（例如纯内部清理信号），后端就该
        // 用一个不同的事件类型 emit 它（参见 `transcript_canceled`），让前端做
        // "事件类型路由"而不是"按 code 判断显示与否"。
        pushActivity({
          level: isRecoverableTtsWarning ? 'warn' : 'error',
          title: `${isRecoverableTtsWarning ? translate('agents.chatSession.activity.warning') : translate('agents.chatSession.activity.error')} · ${code}`,
          detail: msg,
          detailFull: msg,
          time: formatClock((ev as { server_ts?: string }).server_ts),
        })
        if (!recoverable) {
          const finalized = finalizeActiveAssistantMessage(runId, msg, { isError: true })
          if (!finalized) {
            messages.value.push({
              id: `err-${Date.now()}`,
              role: 'assistant',
              contentType: 'text',
              text: msg,
              streaming: false,
              isError: true,
              runId,
              createdAt: Date.now(),
            })
          }
          phase.value = 'error'
          errorMessage.value = `${code}: ${msg}`
        } else if (!isRecoverableTtsWarning) {
          recoverableToast.value = `${code}: ${msg}`
          window.setTimeout(() => {
            if (recoverableToast.value === `${code}: ${msg}`) recoverableToast.value = null
          }, 4200)
        }
        break
      }
      case 'session_closed': {
        sessionState.value = 'closed'
        currentStatus.value = formatSessionState('closed')
        phase.value = 'closed'
        emitAudioCancel({ reason: 'session-closed' })
        break
      }
      case 'audio_delta': {
        const runId = (ev as { run_id?: string | null }).run_id ?? null
        const turnId = (ev as { turn_id?: string | null }).turn_id ?? null
        const idx = ensureAssistantMessage(runId)
        const sampleRateRaw = payload.sample_rate
        const sampleRateNum =
          typeof sampleRateRaw === 'number' && Number.isFinite(sampleRateRaw) && sampleRateRaw > 0
            ? sampleRateRaw
            : null
        const mime = String(payload.mime ?? 'audio/pcm')
        const isFinal = Boolean(payload.is_final)
        const bytesLenRaw = payload.bytes_len
        const bytesLen =
          typeof bytesLenRaw === 'number' && Number.isFinite(bytesLenRaw) && bytesLenRaw > 0
            ? bytesLenRaw
            : 0

        if (bytesLen <= 0) {
          const empty = new ArrayBuffer(0)
          const chunk = { bytes: empty, mime, isFinal, sampleRate: sampleRateNum }
          if (idx >= 0) {
            const cur = messages.value[idx]!
            const chunks = [...(cur.audioChunks ?? []), chunk]
            const next: ChatMessage = { ...cur, audioChunks: chunks }
            messages.value.splice(idx, 1, next)
          }
          emitAudioDelta({ ...chunk, runId, turnId })
          break
        }

        pendingAudioDeltaMeta.push({
          runId,
          turnId,
          mime,
          isFinal,
          sampleRate: sampleRateNum,
          assistantIdx: idx,
        })
        break
      }
      case 'transcript_canceled': {
        // 后端的"行为指令"事件：本 turn 用户输入被丢弃（环境噪声 / 空 transcript /
        // ASR 凑词等场景），请清掉对应的 in-flight transcript 气泡。
        //
        // 这是"按事件类型路由"的纯展示行为：所有 transcript_canceled 事件统一
        // 清最近的 streaming transcript 气泡，前端不对 payload.reason / text 做
        // 任何业务判断。同时不 pushActivity——这不是给用户看的信息（用户没说
        // 话，没有"事件"可言）；用户/开发者可见的描述由后端配套 emit 的
        // `error{empty_transcript}` 单独承载。
        for (let i = messages.value.length - 1; i >= 0; i -= 1) {
          const m = messages.value[i]
          if (!m || m.role !== 'transcript' || !m.streaming) continue
          messages.value.splice(i, 1)
          break
        }
        break
      }
      case 'transcript_delta': {
        const runId = (ev as { run_id?: string | null }).run_id ?? null
        const turnId = (ev as { turn_id?: string | null }).turn_id ?? null
        // 后端 payload.text 是「累计全文」而不是增量片段（参见 inference/audio/session_runtime.py
        // `_emit_transcript`：text=cumulative，delta=new portion）。因此这里做替换而非拼接，
        // 避免每次 delta/final 都把整句话再叠加一遍，出现 "你是谁? 你是谁? 你是谁?" 的复读效果。
        const text = String(payload.text ?? '')
        const isFinal = Boolean(payload.is_final)
        // 语音 ASR 阶段 turn.run_id 可能尚未分配，服务端会带 run_id=null；此前条件要求 runId 真值才建占位，导致增量全丢。
        let idx =
          runId != null && runId !== ''
            ? messages.value.findIndex((m) => m.role === 'transcript' && m.streaming && m.runId === runId)
            : -1
        if (idx < 0) {
          idx = messages.value.findIndex((m) => m.role === 'transcript' && m.streaming)
        }
        // 没有 in-flight 气泡且本次文本为空：跳过创建空气泡（环境噪声触发的空 turn
        // 等场景下，后端会用 `error{empty_transcript}` 单独通知）。
        if (idx < 0 && !text.trim()) break
        if (idx < 0) {
          const idKey = [runId, turnId].filter((x): x is string => Boolean(x && String(x).trim())).join('-') || `u${Date.now()}`
          messages.value.push({
            id: `tr-${idKey}`,
            role: 'transcript',
            contentType: 'text',
            text: '',
            streaming: !isFinal,
            runId,
            turnId,
            createdAt: Date.now(),
          })
          idx = messages.value.length - 1
        }
        const cur = messages.value[idx]!
        const next: ChatMessage = {
          ...cur,
          text,
          streaming: !isFinal,
          hasAssistantDelta: true,
        }
        messages.value.splice(idx, 1, next)
        break
      }
      default:
        console.warn('[chat ws] unhandled event type:', t, ev)
    }
  }

  const wireClient = (client: WebSocketClient, sid: string, modes: { input_mode: string; output_mode: string }) => {
    removeMsgListener?.()
    removeBinaryListener?.()
    pendingAudioDeltaMeta.length = 0

    const handler = (raw: { type?: string; [k: string]: unknown }) => {
      handleWsPayload(raw as ChatWsServerEvent)
    }
    client.on('message', handler)
    removeMsgListener = () => client.off('message', handler)

    const binaryHandler: WebSocketBinaryHandler = (buf: ArrayBuffer) => {
      const meta = pendingAudioDeltaMeta.shift()
      if (!meta) {
        console.warn('[useAgentChatSession] binary frame without pending audio_delta meta')
        return
      }
      const bytesCopy = buf.slice(0)
      const chunk = {
        bytes: bytesCopy,
        mime: meta.mime,
        isFinal: meta.isFinal,
        sampleRate: meta.sampleRate,
      }
      if (meta.assistantIdx >= 0) {
        const cur = messages.value[meta.assistantIdx]!
        const chunks = [...(cur.audioChunks ?? []), chunk]
        const next: ChatMessage = { ...cur, audioChunks: chunks }
        messages.value.splice(meta.assistantIdx, 1, next)
      }
      emitAudioDelta({
        bytes: bytesCopy,
        mime: meta.mime,
        isFinal: meta.isFinal,
        sampleRate: meta.sampleRate,
        runId: meta.runId,
        turnId: meta.turnId,
      })
    }
    client.on('binary', binaryHandler)
    removeBinaryListener = () => client.off('binary', binaryHandler)

    removeCloseListener?.()
    removeCloseListener = client.onClose(async (info: WebSocketCloseInfo) => {
      if (phase.value === 'closed' || phase.value === 'idle') return
      clearReadyTimer()
      if (info.code === 1000) {
        if (phase.value !== 'error') phase.value = 'closed'
        return
      }
      if (info.code === 1008) {
        authFailStreak += 1
        if (authFailStreak >= 2) {
          phase.value = 'error'
          errorMessage.value = translate('agents.chatSession.errors.authTooMany')
          const { clearAuthTokens } = await import('../utils/auth')
          clearAuthTokens()
          window.location.href = '/login'
          return
        }
        try {
          await refreshAccessTokenForWs()
          await openWebSocketOnly(sid, modes)
        } catch (e) {
          phase.value = 'error'
          errorMessage.value = e instanceof Error ? e.message : translate('agents.chatSession.errors.refreshTokenFailed')
        }
        return
      }
      if (info.code === 1011 || info.code === 1006) {
        phase.value = 'error'
        errorMessage.value = translate('agents.chatSession.errors.disconnected')
        return
      }
      phase.value = 'error'
      errorMessage.value = info.reason || translate('agents.chatSession.errors.connectionClosed', { code: info.code })
    })
  }

  const openWebSocketOnly = async (sid: string, modes: { input_mode: string; output_mode: string }) => {
    clearReadyTimer()
    removeCloseListener?.()
    removeCloseListener = undefined
    removeMsgListener?.()
    removeMsgListener = undefined
    removeBinaryListener?.()
    removeBinaryListener = undefined
    wsClient.value?.disconnect()
    wsClient.value = null

    const token = getAccessToken()
    if (!token) throw new Error(translate('agents.chatSession.errors.notLoggedIn'))

    phase.value = 'connecting'
    const client = createWebSocketClient('', { reconnect: false })
    wsClient.value = client
    wireClient(client, sid, modes)

    const path = `/chat/${encodeURIComponent(sid)}`
    await client.connectPath(path, token)
    if (wsClient.value !== client || !client.isConnected) return

    if ((phase.value as ChatPhase) !== 'ready') {
      phase.value = 'opening'
    }
    client.send({
      type: 'session_open',
      session_id: sid,
      turn_id: null,
      payload: {
        input_mode: modes.input_mode,
        output_mode: modes.output_mode,
      },
    })

    if ((phase.value as ChatPhase) !== 'ready') {
      readyTimer = window.setTimeout(() => {
        if (phase.value === 'opening') {
          phase.value = 'error'
          errorMessage.value = translate('agents.chatSession.errors.sessionReadyTimeout')
          try {
            client.disconnect()
          } catch {
            /* ignore */
          }
        }
      }, SESSION_READY_MS)
    }
  }

  const loadHistoryMessages = async (sid: string) => {
    historyOldestOffset.value = 0
    historyHasMore.value = false
    historyLoadingMore.value = false

    // 后端当前只提供升序 + offset 分页；为保证首屏展示最近一页，先探测到尾页。
    let offset = 0
    let total = 0
    let tailRows: ChatMessage[] = []

    while (true) {
      const res = await listChatMessages(sid, { limit: HISTORY_PAGE, offset })
      const batch = (res.items ?? [])
        .map(recordToChatMessage)
        .filter((m): m is ChatMessage => Boolean(m))

      if (!batch.length) break

      total += batch.length
      tailRows = batch
      offset += batch.length

      if (batch.length < HISTORY_PAGE) break
    }

    messages.value = tailRows
    historyOldestOffset.value = Math.max(0, total - tailRows.length)
    historyHasMore.value = historyOldestOffset.value > 0
  }

  const connect = async (sid: string) => {
    if (sessionId.value !== sid) {
      resetSessionState()
    } else {
      errorMessage.value = null
      recoverableToast.value = null
    }
    sessionId.value = sid
    phase.value = 'connecting'
    try {
      const session = await getChatSession(sid)
      upsertRecentChatSessionFromSession(session)
      const modes = readSessionModes(session)
      sessionModes.value = modes
      sessionCapabilities.value = {
        supportsAudioInput: modeSupportsAudioInput(modes.input_mode),
        supportsAudioOutput: modeSupportsAudioOutput(modes.output_mode),
        supportsToolArtifacts: true,
      }

      await loadHistoryMessages(sid)

      await openWebSocketOnly(sid, modes)
    } catch (e) {
      phase.value = 'error'
      errorMessage.value = e instanceof Error ? e.message : translate('agents.chatSession.errors.connectFailed')
      throw e
    }
  }

  const disconnect = () => {
    clearReadyTimer()
    emitAudioCancel({ reason: 'disconnect' })
    removeCloseListener?.()
    removeCloseListener = undefined
    removeMsgListener?.()
    removeMsgListener = undefined
    removeBinaryListener?.()
    removeBinaryListener = undefined
    const sid = sessionId.value
    const client = wsClient.value
    if (client?.isConnected && sid) {
      try {
        client.send({
          type: 'session_close',
          session_id: sid,
          turn_id: null,
          payload: {},
        })
      } catch {
        /* ignore */
      }
    }
    client?.disconnect()
    wsClient.value = null
    if (phase.value !== 'error') phase.value = 'closed'
  }

  const sendUserMessage = (text: string) => {
    const trimmed = text.trim()
    if (!trimmed) return
    if (phase.value !== 'ready' || !sessionId.value || !wsClient.value?.isConnected) return

    const uid = typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `u-${Date.now()}`
    messages.value.push({
      id: uid,
      role: 'user',
      contentType: 'text',
      text: trimmed,
      streaming: false,
      createdAt: Date.now(),
    })

    wsClient.value.send({
      type: 'user_message',
      session_id: sessionId.value,
      turn_id: null,
      payload: { text: trimmed },
    })
  }

  const interrupt = () => {
    if (!sessionId.value || !wsClient.value?.isConnected) return
    // 先本地 cancel 播放，再发送 interrupt；避免等后端 status_changed 回程期间还在出声
    emitAudioCancel({ reason: 'interrupt-sent' })
    wsClient.value.send({
      type: 'interrupt',
      session_id: sessionId.value,
      turn_id: null,
      payload: {},
    })
  }

  /**
   * 数字人开关同步：装饰性副链路，前端 toggle 后通过 chat WS 通知后端
   * `livetalking_client` 更新 enabled 标记。完全 fire-and-forget，
   * 失败 swallow（数字人挂了不应该影响主聊天）。
   *
   * 关键修复：用户意图必须**持久化** —— 如果调用时 chat WS 还没就绪
   * （connectPath 还在握手 / 用户进页面瞬间点 enable / 中途断线），仍然
   * 要把 desired 状态记下来，等 `session_ready` 时统一 flush。否则会
   * 出现"前端按钮已切到开启 + 看到 sidecar 视频，但后端 set_enabled
   * 从未被调，TTS 出声时数字人不动"的死结。
   */
  const sendAvatarToggle = (enabled: boolean): void => {
    const desired = !!enabled
    lastDesiredAvatarEnabled = desired
    flushAvatarToggle()
  }

  const flushAvatarToggle = (): void => {
    if (lastDesiredAvatarEnabled === null) return
    const client = wsClient.value
    const sid = sessionId.value
    if (!sid || !client?.isConnected) return
    try {
      client.send({
        type: 'avatar_toggle',
        session_id: sid,
        turn_id: null,
        payload: { enabled: lastDesiredAvatarEnabled },
      })
    } catch {
      /* ignore：装饰性副链路，绝不影响主链路 */
    }
  }

  const sendAudioChunk = (
    pcm: Int16Array | ArrayBuffer,
    seq: number,
    mime?: string,
    purpose: AudioChunkPurpose = 'user_turn',
  ): boolean => {
    const client = wsClient.value
    const sid = sessionId.value
    if (!sid || !client?.isConnected) return false
    if (phase.value === 'closed' || phase.value === 'error') return false
    let buf: ArrayBuffer
    let byteLen: number
    if (pcm instanceof Int16Array) {
      byteLen = pcm.byteLength
      if (!byteLen) return false
      buf = new ArrayBuffer(byteLen)
      new Uint8Array(buf).set(new Uint8Array(pcm.buffer, pcm.byteOffset, byteLen))
    } else {
      byteLen = pcm.byteLength
      if (!byteLen) return false
      buf = pcm.slice(0)
    }
    const payload: Record<string, unknown> = { seq, bytes_len: byteLen, purpose }
    if (mime) payload.mime = mime
    client.send({
      type: 'audio_chunk',
      session_id: sid,
      turn_id: null,
      payload,
    })
    client.sendBinary(buf)
    return true
  }

  /** 结束当前音频 Turn（PR-4）。仅在 ws 连着、非终态时发送。 */
  const commitAudio = (): boolean => {
    const client = wsClient.value
    const sid = sessionId.value
    if (!sid || !client?.isConnected) return false
    if (phase.value === 'closed' || phase.value === 'error') return false
    client.send({
      type: 'session_commit',
      session_id: sid,
      turn_id: null,
      payload: {},
    })
    return true
  }

  /** 当前会话（能力协商后）是否允许上行语音。 */
  const supportsAudioInput = computed(() => sessionCapabilities.value.supportsAudioInput)

  /** 录音按钮是否可"开始"：允许输出期间持续上行 PCM，后端 VAD 负责 barge-in 判定。 */
  const canStartAudioRecord = computed(
    () =>
      phase.value === 'ready' &&
      AUDIO_UPLINK_STATES.has(String(sessionState.value || '')) &&
      sessionCapabilities.value.supportsAudioInput &&
      !!wsClient.value?.isConnected,
  )

  const loadMoreHistory = async (): Promise<void> => {
    const sid = sessionId.value
    if (!sid || historyLoadingMore.value || !historyHasMore.value) return
    historyLoadingMore.value = true
    try {
      const nextOffset = Math.max(0, historyOldestOffset.value - HISTORY_PAGE)
      const nextLimit = historyOldestOffset.value - nextOffset
      const res = await listChatMessages(sid, {
        limit: nextLimit,
        offset: nextOffset,
      })
      const batch = (res.items ?? [])
        .map(recordToChatMessage)
        .filter((m): m is ChatMessage => Boolean(m))
      if (!batch.length) {
        historyHasMore.value = false
        return
      }
      messages.value = [...batch, ...messages.value]
      historyOldestOffset.value = nextOffset
      historyHasMore.value = nextOffset > 0
    } finally {
      historyLoadingMore.value = false
    }
  }

  const retryConnect = async () => {
    const sid = sessionId.value
    if (!sid) return
    errorMessage.value = null
    phase.value = 'connecting'
    try {
      await openWebSocketOnly(sid, sessionModes.value)
    } catch (e) {
      phase.value = 'error'
      errorMessage.value = e instanceof Error ? e.message : translate('agents.chatSession.errors.reconnectFailed')
    }
  }

  const onBeforeUnload = () => {
    const sid = sessionId.value
    const client = wsClient.value
    if (client?.isConnected && sid) {
      try {
        client.send({
          type: 'session_close',
          session_id: sid,
          turn_id: null,
          payload: {},
        })
      } catch {
        /* ignore */
      }
    }
  }

  if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', onBeforeUnload)
  }

  onBeforeUnmount(() => {
    if (typeof window !== 'undefined') {
      window.removeEventListener('beforeunload', onBeforeUnload)
    }
    disconnect()
  })

  return {
    phase,
    messages,
    activity,
    sessionState,
    currentStatus,
    errorMessage,
    recoverableToast,
    isStreaming,
    canSubmit,
    sessionId,
    connect,
    disconnect,
    sendUserMessage,
    interrupt,
    sendAvatarToggle,
    sendAudioChunk,
    commitAudio,
    canStartAudioRecord,
    supportsAudioInput,
    loadMoreHistory,
    historyHasMore,
    historyLoadingMore,
    retryConnect,
    isSessionNotFoundError,
    isSessionForbiddenError,
    // P1 PR-3：音频下行订阅
    onAudioDelta,
    onAudioCancel,
  }
}
