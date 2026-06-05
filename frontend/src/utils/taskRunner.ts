import { post } from './api'
import { getAccessToken } from './auth'
import { createWebSocketClient, type WebSocketMessage } from './websocket'
import { setLocalCache } from './localCache'
import { getBackendRequestBaseURL, getOutputsBaseURL, resolveBackendPublicUrl } from './runtimeConfig'
import { stripUndefinedDeep } from './stripUndefined'

export type TaskType = 'image' | 'video' | 'audio' | 'text' | 'translate' | 'mini'

export const TASK_CREATE_CACHE_KEY = 'vitoom:taskCreate:last'
export const taskCreateCacheKeyByType = (taskType: TaskType) => `vitoom:taskCreate:last:${taskType}`

export interface TaskCreateRequest {
  task_type: TaskType
  job_type?: string
  prompt?: string

  // image
  negative_prompt?: string
  width?: number
  height?: number
  generate_num?: number
  // required by backend contract: family must equal family
  family?: string
  // model_catalog.load_name
  load_name?: string
  fast_mode?: boolean
  seed?: number

  // common
  model_key?: string

  // allow extension for future video/audio/text
  [key: string]: any
}

export interface TaskCreateResponse {
  task_id: string
  status: string
  message: string
}

export interface TaskFileInfo {
  file_id?: string
  file_name?: string
  file_size?: number
  storage_path?: string
  /**
   * Absolute public URL for original file (new backend field).
   */
  url?: string
  http_url?: string
  thumbnail_path?: string
  /**
   * Absolute public URL for thumbnail (new backend field).
   */
  thumb_url?: string
  index?: number
  width?: number
  height?: number
  mime_type?: string
  [key: string]: any
}

export interface TaskWsMessage extends WebSocketMessage {
  type?: string
  task_id: string
  status?: string
  progress?: number
  error?: string
  files?: TaskFileInfo[]
  [key: string]: any
}

export interface TaskRunOptions {
  /**
   * Auto-send a lightweight heartbeat message periodically.
   * Backend currently ignores client messages but keeps receive loop alive.
   */
  heartbeatMs?: number
}

export interface TaskRunHandle {
  taskId: string
  disconnect: () => void
}

export async function createTask(request: TaskCreateRequest): Promise<TaskCreateResponse> {
  const cleaned = stripUndefinedDeep(request) as TaskCreateRequest

  // Debug helper: print full request payload before sending.
  // Enable by default in dev; in prod you can opt-in via `localStorage.setItem('vitoom:debug:taskCreate', '1')`.
  try {
    const isDev =
      typeof import.meta !== 'undefined' && (import.meta as any)?.env ? Boolean((import.meta as any).env.DEV) : false
    const force = typeof localStorage !== 'undefined' && localStorage.getItem('vitoom:debug:taskCreate') === '1'
    if (isDev || force) {
      // Keep a shallow copy so console expansion doesn't show reactive mutations.
      // NOTE: This prints potentially large fields (e.g. loras); intended for debugging only.
      console.debug('[v1/tasks] createTask request:', { ...cleaned })
    }
  } catch {
    // ignore debug logging errors
  }

  // Note: our default axios baseURL is "/api" (see utils/api.ts).
  // Task APIs are served under "/v1/*", so we must bypass the "/api" prefix here.
  // Vite dev proxy should forward "^/v1/.*" to the backend.
  // 生产：若配置了 backendOrigin，则这里会直接请求后端 origin（不依赖同源/反代）
  const resp = await post<TaskCreateResponse>('/v1/tasks', cleaned, { baseURL: getBackendRequestBaseURL() })
  // Cache last successful request for UX: restore form state on refresh.
  // Keep a per-task_type entry to avoid mixing different task UIs.
  setLocalCache(TASK_CREATE_CACHE_KEY, cleaned, { ttlMs: 30 * 24 * 60 * 60 * 1000 })
  setLocalCache(taskCreateCacheKeyByType(cleaned.task_type), cleaned, { ttlMs: 30 * 24 * 60 * 60 * 1000 })
  return resp
}

export function connectTaskWs(
  taskId: string,
  onMessage: (msg: TaskWsMessage) => void,
  options: TaskRunOptions = {}
): TaskRunHandle {
  const token = getAccessToken() || undefined
  const ws = createWebSocketClient()
  const heartbeatMs = options.heartbeatMs ?? 25000

  let timer: number | undefined

  ws.on('message', (msg: unknown) => onMessage(msg as TaskWsMessage))

  ws.connect(taskId, token).catch((err) => {
    // Surface as a synthetic error message so UIs can display it consistently
    onMessage({
      type: 'client_error',
      task_id: taskId,
      status: 'failed',
      error: err?.message || String(err),
      timestamp: new Date().toISOString(),
    })
  })

  // Heartbeat: send a small string or JSON; backend logs and ignores it.
  timer = window.setInterval(() => {
    try {
      if (ws.isConnected) ws.send({ type: 'ping', t: Date.now() })
    } catch {
      // ignore
    }
  }, heartbeatMs)

  const disconnect = () => {
    if (timer) {
      window.clearInterval(timer)
      timer = undefined
    }
    ws.disconnect()
  }

  return { taskId, disconnect }
}

export function getBestFileUrl(file: TaskFileInfo): string | undefined {
  // Prefer explicit public URL from backend (newer contract).
  const thumb = (file as any)?.thumb_url
  if (typeof thumb === 'string' && thumb.trim()) return resolveBackendPublicUrl(thumb.trim())

  // Backward-compatible fallback: build URL from thumbnail_path under outputs base.
  if (!file.thumbnail_path) return undefined
  const base = getOutputsBaseURL()
  return `${base}/${String(file.thumbnail_path).replace(/^[\\/]+/, '')}`
}

export function getOriginalFileUrl(file: TaskFileInfo): string | undefined {
  // Prefer explicit public URL from backend (newer contract).
  const url = (file as any)?.url
  if (typeof url === 'string' && url.trim()) return resolveBackendPublicUrl(url.trim())

  // Some backends may send http_url as public URL.
  if (typeof file.http_url === 'string' && file.http_url.trim()) return resolveBackendPublicUrl(file.http_url.trim())

  // Backward-compatible fallback: build URL from storage_path under outputs base.
  if (!file.storage_path) return undefined
  const base = getOutputsBaseURL()
  return `${base}/${String(file.storage_path).replace(/^[\\/]+/, '')}`
}

