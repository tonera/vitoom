import axios from 'axios'
import { computed, ref } from 'vue'
import { getAccessToken } from '../utils/auth'
import { getBackendRequestBaseURL } from '../utils/runtimeConfig'
import { translate } from '../utils/translate'

export type UserFileItem = {
  id: string
  url: string
  thumb_url?: string
  file_name?: string
  created_at?: string
  mime_type?: string
  file_size?: number
}

export type UseUserFilesGalleryOptions = {
  category?: string
  limit?: number
  onLoaded?: (ctx: { reset: boolean; items: UserFileItem[]; total: number }) => void
  /**
   * 用于去重/合并的 key，默认优先 url 其次 id
   */
  getKey?: (it: UserFileItem) => string
}

function defaultKey(it: UserFileItem): string {
  return String(it?.url || it?.id || '')
}

function mergeUnique(prev: UserFileItem[], next: UserFileItem[], getKey: (it: UserFileItem) => string): UserFileItem[] {
  const seen = new Set<string>()
  const out: UserFileItem[] = []
  for (const it of prev) {
    const k = getKey(it)
    if (!k || seen.has(k)) continue
    seen.add(k)
    out.push(it)
  }
  for (const it of next) {
    const k = getKey(it)
    if (!k || seen.has(k)) continue
    seen.add(k)
    out.push(it)
  }
  return out
}

export function useUserFilesGallery(options: UseUserFilesGalleryOptions = {}) {
  const category = String(options.category || 'image')
  const getKey = options.getKey || defaultKey

  const items = ref<UserFileItem[]>([])
  const loading = ref(false)
  const error = ref('')
  const total = ref(0)
  const limit = ref(Number.isFinite(Number(options.limit)) ? Math.max(1, Math.floor(Number(options.limit))) : 60)
  const offset = ref(0)
  const loadingMore = ref(false)
  const exhausted = computed(() => total.value > 0 && items.value.length >= total.value)

  async function load(reset = false) {
    loading.value = true
    error.value = ''
    try {
      const token = getAccessToken()
      if (reset) {
        offset.value = 0
        total.value = 0
        items.value = []
      }

      const resp = await axios.get('/v1/user/files', {
        baseURL: getBackendRequestBaseURL(),
        headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
        params: { category, limit: limit.value, offset: offset.value },
      })

      const nextItems = (resp?.data?.data?.items || []) as UserFileItem[]
      const nextTotal = Number(resp?.data?.data?.total || 0)
      total.value = Number.isFinite(nextTotal) ? nextTotal : 0

      if (reset) items.value = nextItems
      else items.value = mergeUnique(items.value, nextItems, getKey)

      options.onLoaded?.({ reset, items: items.value, total: total.value })
    } catch (e: any) {
      error.value = e?.response?.data?.detail || e?.message || translate('assets.loadFailed')
    } finally {
      loading.value = false
    }
  }

  async function loadNext() {
    if (loading.value || loadingMore.value) return
    if (exhausted.value) return
    loadingMore.value = true
    try {
      offset.value = offset.value + limit.value
      await load(false)
    } finally {
      loadingMore.value = false
    }
  }

  function prepend(item: UserFileItem) {
    const k = getKey(item)
    if (!k) return
    const filtered = items.value.filter((x) => getKey(x) !== k)
    items.value = [item, ...filtered]
  }

  return {
    // state
    items,
    loading,
    loadingMore,
    error,
    total,
    limit,
    offset,
    exhausted,
    // actions
    load,
    loadNext,
    prepend,
  }
}

