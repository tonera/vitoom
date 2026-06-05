import axios, { isAxiosError, type AxiosRequestConfig } from 'axios'
import { get, post } from '../utils/api'
import { getApiBaseURL, getBackendRequestBaseURL, getV1BasePath } from '../utils/runtimeConfig'
import { translate } from '../utils/translate'

/**
 * Chat 会话 HTTP 使用路径前缀 /v1/chat（与 OpenAI 的 /v1 同级），不在默认 axios 的 /api 下。
 * 单次请求通过 baseURL 覆盖：同源时为空 + /v1/...（走 Vite 代理）；分离部署时用 backendOrigin。
 */
function chatHttpConfig(): AxiosRequestConfig {
  const origin = getBackendRequestBaseURL().trim()
  return { baseURL: origin ? origin.replace(/\/+$/, '') : '' }
}

function chatV1Path(suffix: string): string {
  const root = getV1BasePath().replace(/\/+$/, '') || '/v1'
  const s = suffix.startsWith('/') ? suffix : `/${suffix}`
  return `${root}${s}`
}

/** 后端 conversations 行 + metadata */
export interface ChatSession {
  id: string
  title?: string | null
  status?: string
  metadata?: Record<string, unknown>
  created_at?: string | null
  updated_at?: string | null
}

export interface ChatMessageRecord {
  id: string
  conversation_id?: string
  role: string
  content: string
  agent_run_id?: string | null
  turn_id?: string | null
  metadata?: Record<string, unknown>
  created_at?: string | null
}

/** 语音回复偏好（TTS）。字段与 backend.ChatSessionAudioOutputRequest 对齐。 */
export interface ChatSessionAudioOutput {
  load_name?: string | null
  /** custom_voice / voice_design / voice_clone */
  tts_mode?: string | null
  /** custom_voice 模式下的说话人名 */
  speaker_name?: string | null
  /** speaker_name 的兼容别名 */
  voice_preset?: string | null
  /** 自然语言风格描述 */
  instruct?: string | null
  language?: string | null
  sample_rate?: number | null
  /** 音频格式，如 wav / mp3 / pcm */
  file_type?: string | null
}

export interface CreateChatSessionBody {
  agent_id?: string | null
  title?: string | null
  input_mode?: string
  output_mode?: string
  load_name?: string | null
  audio_output?: ChatSessionAudioOutput
  metadata?: Record<string, unknown>
}

export interface ListChatMessagesResult {
  items: ChatMessageRecord[]
  count: number
}

export type ChatWsClientType =
  | 'session_open'
  | 'user_message'
  | 'audio_chunk'
  | 'session_commit'
  | 'interrupt'
  | 'session_close'

export interface ChatWsClientEnvelope<T extends ChatWsClientType = ChatWsClientType> {
  type: T
  session_id: string
  turn_id?: string | null
  payload?: Record<string, unknown>
  client_ts?: string
}

/** 下行事件（按协议 root.type 判别；其余字段宽松解析） */
export type ChatWsServerEvent =
  | { type: 'session_ready'; session_id?: string; server_ts?: string; payload?: Record<string, unknown> }
  | { type: 'capabilities_changed'; session_id?: string; server_ts?: string; payload?: Record<string, unknown> }
  | { type: 'status_changed'; session_id?: string; turn_id?: string | null; run_id?: string | null; payload?: Record<string, unknown> }
  | { type: 'message_started'; session_id?: string; turn_id?: string | null; run_id?: string | null; payload?: Record<string, unknown> }
  | { type: 'message_delta'; session_id?: string; turn_id?: string | null; run_id?: string | null; payload?: Record<string, unknown> }
  | { type: 'message_completed'; session_id?: string; turn_id?: string | null; run_id?: string | null; payload?: Record<string, unknown> }
  | { type: 'transcript_delta'; session_id?: string; turn_id?: string | null; run_id?: string | null; payload?: Record<string, unknown> }
  | { type: 'audio_delta'; session_id?: string; turn_id?: string | null; run_id?: string | null; payload?: Record<string, unknown> }
  | { type: 'tool_call_started'; session_id?: string; turn_id?: string | null; run_id?: string | null; payload?: Record<string, unknown> }
  | { type: 'tool_call_completed'; session_id?: string; turn_id?: string | null; run_id?: string | null; payload?: Record<string, unknown> }
  | { type: 'tool_call_failed'; session_id?: string; turn_id?: string | null; run_id?: string | null; payload?: Record<string, unknown> }
  | { type: 'artifact_created'; session_id?: string; turn_id?: string | null; run_id?: string | null; payload?: Record<string, unknown> }
  | { type: 'error'; session_id?: string; payload?: Record<string, unknown> }
  | { type: 'session_closed'; session_id?: string; payload?: Record<string, unknown> }
  | { type: string; session_id?: string; payload?: Record<string, unknown> }

export function createChatSession(body: CreateChatSessionBody): Promise<ChatSession> {
  return post<ChatSession>(chatV1Path('/chat/sessions'), body, chatHttpConfig())
}

export function getChatSession(sessionId: string): Promise<ChatSession> {
  return get<ChatSession>(chatV1Path(`/chat/sessions/${encodeURIComponent(sessionId)}`), chatHttpConfig())
}

export function listChatMessages(
  sessionId: string,
  params?: { limit?: number; offset?: number },
): Promise<ListChatMessagesResult> {
  return get<ListChatMessagesResult>(
    chatV1Path(`/chat/sessions/${encodeURIComponent(sessionId)}/messages`),
    {
      ...chatHttpConfig(),
      params: { limit: params?.limit ?? 200, offset: params?.offset ?? 0 },
    },
  )
}

export interface ListChatSessionsResult {
  items: ChatSession[]
  count: number
}

/** 分页列出当前用户的会话；`q` 为按标题子串过滤（与后端 Query `q` 一致） */
export function listChatSessions(params?: {
  limit?: number
  offset?: number
  /** 标题关键字，空格前后会 trim；空字符串不传参 */
  q?: string
}): Promise<ListChatSessionsResult> {
  const limit = params?.limit ?? 50
  const offset = params?.offset ?? 0
  const q = params?.q?.trim()
  return get<ListChatSessionsResult>(chatV1Path('/chat/sessions'), {
    ...chatHttpConfig(),
    params: {
      limit,
      offset,
      ...(q ? { q } : {}),
    },
  })
}

export async function refreshAccessTokenForWs(): Promise<string> {
  const { getRefreshToken, setAccessToken, clearAuthTokens } = await import('../utils/auth')
  const refreshToken = getRefreshToken()
  if (!refreshToken) {
    clearAuthTokens()
    throw new Error(translate('agents.chatSession.errors.refreshTokenMissing'))
  }
  const API_BASE_URL = getApiBaseURL() || import.meta.env.VITE_API_BASE_URL || '/api'
  const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
    refresh_token: refreshToken,
  })
  const accessToken = response.data?.data?.access_token ?? response.data?.access_token
  if (!accessToken || typeof accessToken !== 'string') {
    clearAuthTokens()
    throw new Error(translate('agents.chatSession.errors.refreshTokenFailed'))
  }
  setAccessToken(accessToken)
  return accessToken
}

export function isSessionNotFoundError(e: unknown): boolean {
  return isAxiosError(e) && e.response?.status === 404
}

export function isSessionForbiddenError(e: unknown): boolean {
  return isAxiosError(e) && e.response?.status === 403
}

/** 本机最近会话（后端列表接口上线前过渡） */
export interface RecentChatSessionRow {
  id: string
  title: string
  updatedAt: number
}

const RECENT_SESSIONS_KEY = 'vitoom:recentChatSessions'
const RECENT_MAX = 30

function readRecentSessions(): RecentChatSessionRow[] {
  try {
    const raw = localStorage.getItem(RECENT_SESSIONS_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) return []
    return parsed
      .filter((x): x is RecentChatSessionRow => {
        if (!x || typeof x !== 'object') return false
        const o = x as Record<string, unknown>
        return typeof o.id === 'string' && typeof o.updatedAt === 'number'
      })
      .slice(0, RECENT_MAX)
  } catch {
    return []
  }
}

function writeRecentSessions(rows: RecentChatSessionRow[]): void {
  try {
    localStorage.setItem(RECENT_SESSIONS_KEY, JSON.stringify(rows.slice(0, RECENT_MAX)))
  } catch {
    /* ignore quota */
  }
}

export function pushRecentChatSession(row: RecentChatSessionRow): void {
  const rest = readRecentSessions().filter((r) => r.id !== row.id)
  writeRecentSessions([{ ...row, title: row.title || translate('agents.session.newSession') }, ...rest])
}

export function upsertRecentChatSessionFromSession(session: ChatSession): void {
  const title = String(session.title || '').trim() || translate('agents.session.newSession')
  const updatedAt = session.updated_at ? Date.parse(session.updated_at) : Date.now()
  pushRecentChatSession({
    id: session.id,
    title,
    updatedAt: Number.isFinite(updatedAt) ? updatedAt : Date.now(),
  })
}

export function listRecentChatSessions(): RecentChatSessionRow[] {
  return readRecentSessions().sort((a, b) => b.updatedAt - a.updatedAt)
}

export function removeRecentChatSession(sessionId: string): void {
  writeRecentSessions(readRecentSessions().filter((r) => r.id !== sessionId))
}
