import { ref } from 'vue'
import { createTask, connectTaskWs, type TaskCreateRequest, type TaskWsMessage, type TaskRunHandle } from '../utils/taskRunner'
import { handleApiError } from '../utils/api'
import { resolveWsMessage } from '../utils/errorMessage'
import { i18n } from '../i18n'

type PendingRun = {
  remaining: number
  placeholderKeys: string[]
  req: TaskCreateRequest
  taskId?: string
}

export type TaskRunResultFileHook = (params: {
  file: any
  taskId: string
  msg: TaskWsMessage
  /**
   * The request used to create this task (useful for UI details).
   */
  req: TaskCreateRequest
  /**
   * If placeholders were created for this run, this is the placeholder key that
   * should be replaced by the received file (one per file).
   */
  replaceKey?: string
  /**
   * Unique ID of the run on the client side (useful for debugging).
   */
  runId: string
}) => boolean

export interface RunTaskHooks {
  /**
   * Called immediately when placeholders are created, before network request.
   * The page should insert placeholder UI items using these keys.
   */
  onAddPlaceholders: (placeholderKeys: string[], runId: string) => void
  /**
   * Called when placeholders (or other keys) should be removed from the page UI.
   */
  onRemoveKeys: (keys: string[]) => void
  /**
   * Called for each result file received from WS.
   * Return true if the file was inserted/handled; false if it was ignored/duplicate.
   * If false and replaceKey is provided, the manager will request removing that placeholder.
   */
  onResultFile: TaskRunResultFileHook
  /**
   * Called for any user-displayable error (create task, failed/cancelled, WS error message, client_error).
   * Page can decide how to format the message based on phase.
   */
  onError?: (err: TaskRunError) => void
  /**
   * Called for non-terminal progress/status messages from WS.
   */
  onProgress?: (params: { runId: string; taskId: string; status?: string; progress?: number; msg: TaskWsMessage }) => void
  /**
   * Called when the task reaches a terminal state on WS (completed/failed/cancelled).
   * Useful for per-button loading state, etc.
   */
  onTerminal?: (params: { runId: string; taskId: string; status: 'completed' | 'failed' | 'cancelled' }) => void
}

export interface UseTaskRunManagerOptions {
  heartbeatMs?: number
  /**
   * Placeholder key prefix. Keep stable so UI/test can rely on it.
   */
  placeholderPrefix?: string
}

export type TaskRunErrorPhase = 'create' | 'ws_failed' | 'ws_message'

export type TaskRunError = {
  phase: TaskRunErrorPhase
  message: string
  runId: string
  taskId?: string
}

export function useTaskRunManager(options: UseTaskRunManagerOptions = {}) {
  const placeholderPrefix = options.placeholderPrefix || 'ph'

  // Non-reactive internal state
  const activeWsHandles = new Map<string, TaskRunHandle>()
  const taskIdToRunId = new Map<string, string>()
  const pendingRuns = new Map<string, PendingRun>()

  // Reactive mirror for UIs (e.g. disable submit while generating)
  const generatingCount = ref(0)
  const syncGeneratingCount = () => {
    generatingCount.value = pendingRuns.size
  }

  const isTerminalWsStatus = (s?: string) => {
    const v = String(s || '').toLowerCase()
    return v === 'completed' || v === 'failed' || v === 'cancelled'
  }

  const makeRunId = () => `run-${Date.now()}-${Math.random().toString(16).slice(2)}`

  const makePlaceholderKeys = (runId: string, want: number) => {
    const n = Math.max(1, Math.min(9, Math.floor(Number(want) || 1)))
    const keys: string[] = []
    for (let i = 0; i < n; i++) keys.push(`${placeholderPrefix}:${runId}:${i}:${Date.now()}`)
    return keys
  }

  const cleanupTask = (taskId: string) => {
    const handle = activeWsHandles.get(taskId)
    if (handle) {
      try {
        handle.disconnect()
      } catch {
        // ignore
      }
    }
    activeWsHandles.delete(taskId)
    taskIdToRunId.delete(taskId)
  }

  const cleanupRun = (runId: string) => {
    pendingRuns.delete(runId)
    syncGeneratingCount()
  }

  const disconnectAll = () => {
    for (const h of activeWsHandles.values()) {
      try {
        h.disconnect()
      } catch {
        // ignore
      }
    }
    activeWsHandles.clear()
    taskIdToRunId.clear()
    pendingRuns.clear()
    syncGeneratingCount()
  }

  async function runTask(
    req: TaskCreateRequest,
    want: number,
    hooks: RunTaskHooks
  ): Promise<{ runId: string; taskId?: string }> {
    const runId = makeRunId()
    const placeholderKeys = makePlaceholderKeys(runId, want)

    // Insert placeholders immediately for better UX
    hooks.onAddPlaceholders(placeholderKeys, runId)

    pendingRuns.set(runId, {
      remaining: placeholderKeys.length,
      placeholderKeys: [...placeholderKeys],
      req,
    })
    syncGeneratingCount()

    try {
      const created = await createTask(req)
      const taskId = created.task_id
      taskIdToRunId.set(taskId, runId)

      const run = pendingRuns.get(runId)
      if (run) run.taskId = taskId

      const handle = connectTaskWs(
        taskId,
        (msg: TaskWsMessage) => {
          if (!msg || msg.task_id !== taskId) return

          const rid = taskIdToRunId.get(taskId) || runId
          const curRun = pendingRuns.get(rid)
          const status = String((msg as any).status || '').toLowerCase()
          const rawProgress = (msg as any).progress
          const progress = typeof rawProgress === 'number' && Number.isFinite(rawProgress) ? rawProgress : undefined

          if ((status || progress !== undefined) && curRun) {
            hooks.onProgress?.({ runId: rid, taskId, status: status || undefined, progress, msg })
          }

          // failed/cancelled: drop remaining placeholders + surface error
          if (status === 'failed' || status === 'cancelled') {
            const remain = curRun?.remaining || 0
            if (curRun && remain > 0) {
              const keysToRemove = curRun.placeholderKeys.splice(0, remain)
              hooks.onRemoveKeys(keysToRemove)
              curRun.remaining = 0
            }
            const errMsg = resolveWsMessage({
              message_code: (msg as any)?.message_code,
              message_params: (msg as any)?.message_params,
              message: (msg as any)?.message,
              error: (msg as any)?.error,
            }) || (status === 'cancelled'
              ? i18n.global.t('errors.task.cancelled')
              : i18n.global.t('errors.task.failed'))
            hooks.onError?.({ phase: 'ws_failed', message: errMsg, runId: rid, taskId })
            hooks.onTerminal?.({ runId: rid, taskId, status: status as 'failed' | 'cancelled' })

            cleanupTask(taskId)
            if (rid) cleanupRun(rid)
            return
          }

          if (msg.type === 'result' && Array.isArray((msg as any).files)) {
            const runReq = (curRun?.req || req) as TaskCreateRequest
            for (const file of (msg as any).files as any[]) {
              // consume one placeholder per received file (from the end)
              const replaceKey = curRun && curRun.remaining > 0 ? curRun.placeholderKeys.pop() : undefined
              if (curRun && curRun.remaining > 0) curRun.remaining = Math.max(0, curRun.remaining - 1)

              const inserted = hooks.onResultFile({
                file,
                taskId,
                msg,
                req: runReq,
                replaceKey,
                runId: rid,
              })
              // If it's a duplicate/ignored file, still remove the placeholder so we don't leave spinners hanging.
              if (!inserted && replaceKey) hooks.onRemoveKeys([replaceKey])
            }
          }

          // Some backends may send error without failed status
          if ((msg as any).error)
            hooks.onError?.({
              phase: 'ws_message',
              message: resolveWsMessage({
                message_code: (msg as any)?.message_code,
                message_params: (msg as any)?.message_params,
                message: (msg as any)?.message,
                error: String((msg as any).error),
              }),
              runId: rid,
              taskId,
            })

          if (isTerminalWsStatus((msg as any).status)) {
            // If completed but still has placeholders (rare), drop them
            if (status === 'completed' && curRun && curRun.remaining > 0) {
              const keysToRemove = curRun.placeholderKeys.splice(0, curRun.remaining)
              hooks.onRemoveKeys(keysToRemove)
              curRun.remaining = 0
            }
            if (status === 'completed' || status === 'failed' || status === 'cancelled') {
              hooks.onTerminal?.({ runId: rid, taskId, status: status as 'completed' | 'failed' | 'cancelled' })
            }
            cleanupTask(taskId)
            if (rid) cleanupRun(rid)
          }
        },
        { heartbeatMs: options.heartbeatMs }
      )
      activeWsHandles.set(taskId, handle)

      return { runId, taskId }
    } catch (error: any) {
      // createTask failed: remove placeholders
      hooks.onRemoveKeys([...placeholderKeys])
      cleanupRun(runId)

      const apiError = handleApiError(error)
      hooks.onError?.({ phase: 'create', message: apiError.message, runId })
      return { runId }
    }
  }

  return {
    generatingCount,
    runTask,
    disconnectAll,
  }
}


