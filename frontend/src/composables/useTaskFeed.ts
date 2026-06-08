import { shallowRef, type Ref } from 'vue'
import type { TaskCreateRequest, TaskWsMessage } from '../utils/taskRunner'
import { getBestFileUrl, getOriginalFileUrl } from '../utils/taskRunner'
import type { RunTaskHooks } from './useTaskRunManager'

export type TaskFeedMediaKind = 'image' | 'video'

export type TaskFeedMediaItem<TDetails = any> = {
  kind: TaskFeedMediaKind
  key: string
  thumbSrc: string
  originalSrc?: string
  posterSrc?: string
  title?: string
  downloadName?: string
  details?: TDetails
}

export type TaskFeedPlaceholderItem = {
  kind: 'placeholder'
  key: string
  runId: string
}

export type TaskFeedItem<TDetails = any> = TaskFeedMediaItem<TDetails> | TaskFeedPlaceholderItem

export type UseTaskFeedOptions = {
  /**
   * Called when a media item is inserted (including replacing a placeholder).
   * Useful for pages with preview panes (e.g. auto-select newly generated key).
   */
  onInsertedKey?: (key: string) => void
}

export function useTaskFeed<TDetails = any>(options: UseTaskFeedOptions = {}) {
  const feed = shallowRef<TaskFeedItem<TDetails>[]>([])
  const seenMediaKeys = new Set<string>()

  const insertPlaceholders = (placeholderKeys: string[], runId: string) => {
    if (!placeholderKeys?.length) return
    feed.value = [
      ...placeholderKeys.map((k) => ({ kind: 'placeholder' as const, key: k, runId })),
      ...feed.value,
    ]
  }

  const removeKeys = (keys: string[]) => {
    if (!keys?.length) return
    const keySet = new Set(keys)
    feed.value = feed.value.filter((x) => !keySet.has(x.key))
  }

  const addMedia = (item: TaskFeedMediaItem<TDetails>, opts: { replaceKey?: string } = {}) => {
    if (!item?.key) return false
    if (seenMediaKeys.has(item.key)) return false
    seenMediaKeys.add(item.key)

    if (opts.replaceKey) {
      const idx = feed.value.findIndex((x) => x.key === opts.replaceKey)
      if (idx !== -1) {
        const next = [...feed.value]
        next.splice(idx, 1, item)
        feed.value = next
        options.onInsertedKey?.(item.key)
        return true
      }
    }

    feed.value = [item, ...feed.value]
    options.onInsertedKey?.(item.key)
    return true
  }

  type BuildDetailsParams = {
    file: any
    taskId: string
    msg: TaskWsMessage
    req: TaskCreateRequest
  }

  type CreateRunHooksParams = {
    title?: { image?: string; video?: string }
    fallbackDownloadName?: { image?: string; video?: string }
    buildDetails?: (p: BuildDetailsParams) => TDetails | undefined
    onError?: RunTaskHooks['onError']
  }

  const createRunHooks = (params: CreateRunHooksParams = {}): RunTaskHooks => {
    return {
      onAddPlaceholders: insertPlaceholders,
      onRemoveKeys: removeKeys,
      onResultFile: ({ file, taskId, msg, req, replaceKey }) => {
        const url = getBestFileUrl(file)
        const originalUrl = getOriginalFileUrl(file)
        if (!url && !originalUrl) return false

        const key = String(originalUrl || url)
        const wsTaskType = String((msg as any)?.task_type || (req as any)?.task_type || '').toLowerCase()
        const mime = String((file as any)?.mime_type || '').toLowerCase()
        const isVideo = wsTaskType === 'video' || mime.startsWith('video/')
        const kind: TaskFeedMediaKind = isVideo ? 'video' : 'image'

        const details = params.buildDetails?.({ file, taskId, msg, req })
        const title = isVideo ? params.title?.video : params.title?.image
        const fallbackName = isVideo ? params.fallbackDownloadName?.video : params.fallbackDownloadName?.image

        return addMedia(
          {
            kind,
            key,
            thumbSrc: url || '',
            originalSrc: originalUrl || url,
            posterSrc: isVideo ? (url || undefined) : undefined,
            title,
            downloadName: (file as any)?.file_name || fallbackName,
            details,
          },
          { replaceKey }
        )
      },
      onError: params.onError,
    }
  }

  return {
    feed: feed as Ref<TaskFeedItem<TDetails>[]>,
    insertPlaceholders,
    removeKeys,
    addMedia,
    createRunHooks,
  }
}

