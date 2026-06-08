import { ref, watch, type Ref } from 'vue'
import { getRaw } from '../utils/api'
import type { UiModelOption } from './useCheckpointModels'

export type UiLoraSelectionItem = { value: string; weight?: number; locked?: boolean }
export type LoraPayloadItem = { name: string; weight: number; trigger_word?: string }

export type UseLoraModelsOptions = {
  /**
   * Endpoint query "type" (default: image). Video page should pass "video".
   */
  taskType?: 'image' | 'video' | string
  /**
   * Base checkpoint family (used for filtering loras)
   */
  baseFamily: Ref<string>
  /**
   * v-model selection array from UI (e.g. `toRef(form, 'lora')`)
   */
  selected: Ref<UiLoraSelectionItem[]>
  /**
   * Output payload array used for request submit (e.g. `toRef(form, 'loras')`)
   */
  payload: Ref<LoraPayloadItem[]>
  /**
   * Default weight when UI item has no weight. (Generate/Edit: 0.5, Controler: 0.8)
   */
  defaultWeight?: number
  /**
   * Maximum number of loras to include in payload.
   */
  maxSelected?: number
  /**
   * Whether to automatically fetch loras when baseFamily changes.
   */
  autoFetchOnBaseModelChange?: boolean
}

export function useLoraModels(opts: UseLoraModelsOptions) {
  const taskType = opts.taskType ?? 'image'
  const defaultWeight = typeof opts.defaultWeight === 'number' ? opts.defaultWeight : 0.5
  const maxSelected = typeof opts.maxSelected === 'number' ? opts.maxSelected : 3
  const autoFetchOnBaseModelChange = opts.autoFetchOnBaseModelChange !== false

  const loraModelOptions = ref<UiModelOption[]>([])
  const loraLoading = ref(false)
  const loraTotal = ref(0)
  const loraMeta = ref<any>(null)
  const loraPage = ref(1)
  const loraPerPage = ref(8)
  const loraSearchKeyword = ref('')
  const loraStorageFilter = ref<string | undefined>(undefined)
  const loraClassFilter = ref<string | undefined>(undefined)
  const loraOptionCache = ref<Record<string, UiModelOption>>({})
  // de-dupe repeated fetches (page mount often triggers multiple callers & watchers)
  const inflight = new Map<string, Promise<void>>()
  let lastResolvedKey = ''

  function clampNum(v: number, min: number, max: number) {
    if (!Number.isFinite(v)) return min
    return Math.min(max, Math.max(min, v))
  }

  const fetchLoras = async () => {
    const baseFamily = String(opts.baseFamily.value || '').trim()
    if (!baseFamily) {
      loraModelOptions.value = []
      loraTotal.value = 0
      loraMeta.value = null
      return
    }

    const offset = (loraPage.value - 1) * loraPerPage.value
    const kw = loraSearchKeyword.value.trim()
    const nameQuery = kw ? `&name=${encodeURIComponent(kw)}` : ''
      const storageQuery = loraStorageFilter.value ? `&storage_mode=${encodeURIComponent(loraStorageFilter.value)}` : ''
      const classQuery = loraClassFilter.value ? `&lora_family=${encodeURIComponent(loraClassFilter.value)}` : ''
      const url = `/models?modality=${encodeURIComponent(String(taskType))}&service_status=active&asset_type=lora&family=${encodeURIComponent(
      baseFamily
    )}&limit=${loraPerPage.value}&offset=${offset}${nameQuery}${storageQuery}${classQuery}`
    const key = `lora:${url}`

    // if we've already loaded the same query, skip extra re-fetch
    if (key === lastResolvedKey && (loraModelOptions.value.length > 0 || loraTotal.value > 0)) return

    const running = inflight.get(key)
    if (running) return await running

    const task = (async () => {
      loraLoading.value = true
      let ok = false
      try {
        const resp = await getRaw<any[]>(url)
        const models = Array.isArray(resp?.data) ? resp.data : []
        loraTotal.value = Number(resp?.meta?.total || models.length || 0)
        loraMeta.value = resp?.meta || null

        const optsArr = models.map((m) => ({
          value: String(m?.model_key || ''),
          label: String(m?.name || m?.load_name || m?.model_key || ''),
          name: String(m?.name || m?.load_name || m?.model_key || ''),
          load_name: String(m?.load_name || m?.name || m?.id),
          thumb: (m?.thumb || m?.thumbnail || m?.cover || '') as string,
          storage_mode: m?.storage_mode ? String(m.storage_mode) : '',
          family: m?.family ? String(m.family) : '',
          asset_type: String(m?.asset_type || 'lora'),
          trigger_words: Array.isArray(m?.trigger_words) ? m.trigger_words : [],
        }))
        loraModelOptions.value = optsArr
        for (const o of optsArr) loraOptionCache.value[o.value] = o
        ok = true
      } catch (e) {
        console.error('Failed to fetch loras:', e)
        loraModelOptions.value = []
        loraTotal.value = 0
      } finally {
        loraLoading.value = false
        if (ok) lastResolvedKey = key
      }
    })()

    inflight.set(key, task)
    try {
      await task
    } finally {
      inflight.delete(key)
    }
  }

  const handleLoraPageChange = ({ page, perPage }: { page: number; perPage: number }) => {
    loraPerPage.value = perPage
    loraPage.value = page
    fetchLoras()
  }
  const handleLoraSearchChange = (keyword: string) => {
    loraSearchKeyword.value = keyword || ''
    loraPage.value = 1
    fetchLoras()
  }
  const handleLoraFilterChange = (payload: { storageType?: string; family?: string }) => {
    loraStorageFilter.value = payload.storageType
    loraClassFilter.value = payload.family
    loraPage.value = 1
    fetchLoras()
  }

  const resetLoras = () => {
    opts.selected.value = []
    opts.payload.value = []
    loraSearchKeyword.value = ''
    loraStorageFilter.value = undefined
    loraClassFilter.value = undefined
    loraPage.value = 1
    loraOptionCache.value = {}
    loraModelOptions.value = []
    loraTotal.value = 0
    loraMeta.value = null
  }

  // map UI selection -> request payload (keep consistent with current pages)
  watch(
    opts.selected,
    (arr) => {
      const input = Array.isArray(arr) ? arr : []
      const out = input
        .slice(0, maxSelected)
        .map((x) => {
          const id = String((x as any)?.value || '').trim()
          if (!id) return null
          const opt = loraOptionCache.value[id]
          const name = String(opt?.load_name || opt?.name || opt?.label || id)
          const trigger_word = Array.isArray((opt as any)?.trigger_words) ? String((opt as any).trigger_words.join(',')).trim() : ''
          const wRaw = typeof (x as any)?.weight === 'number' ? Number((x as any).weight) : defaultWeight
          const w = Math.round(clampNum(Number.isFinite(wRaw) ? wRaw : defaultWeight, -1, 2) * 10) / 10
          return { name, weight: w, trigger_word }
        })
        .filter(Boolean) as LoraPayloadItem[]

      opts.payload.value = out
    },
    { deep: true }
  )

  // when base model class becomes available / changes: fetch & reset selections like current pages do
  watch(
    opts.baseFamily,
    (val, prev) => {
      const next = String(val || '').trim()
      const old = String(prev || '').trim()
      if (next && !old) {
        if (autoFetchOnBaseModelChange) fetchLoras()
        return
      }
      if (!next || !old) return
      if (next === old) return
      resetLoras()
      if (autoFetchOnBaseModelChange) fetchLoras()
    },
    { immediate: false }
  )

  return {
    loraModelOptions,
    loraLoading,
    loraTotal,
    loraMeta,
    loraPage,
    loraPerPage,
    loraSearchKeyword,
    loraStorageFilter,
    loraClassFilter,
    loraOptionCache,
    fetchLoras,
    handleLoraPageChange,
    handleLoraSearchChange,
    handleLoraFilterChange,
    resetLoras,
  }
}

