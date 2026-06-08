import { computed, ref, type Ref } from 'vue'
import { getRaw } from '../utils/api'
import { getLocalCache, setLocalCache } from '../utils/localCache'

export interface UiModelOption {
  value: string
  model_key?: string
  label: string
  name?: string
  load_name?: string
  thumb?: string
  storage_mode?: string
  family?: string
  asset_type?: string
  capabilities?: Record<string, any>
  trigger_words?: string[]
  runtime_config?: any
  video_profile?: any
}

export const DEFAULT_MODEL_LAST_SELECTED_CACHE_KEY = 'vitoom:model:last:checkpoint'

export type UseCheckpointModelsOptions = {
  /**
   * v-model binding for checkpoint model_key (usually `form.modelKey`)
   */
  modelKey: Ref<string>
  /**
   * Only fetch models that are editable (used by edit/control pages).
   */
  onlyEditable?: boolean
  /**
   * Cache key for last selected checkpoint model.
   */
  cacheKey?: string
  /**
   * TTL for cached model selection.
   */
  cacheTtlMs?: number
  /**
   * Endpoint query "type" (default: image)
   */
  taskType?: 'image' | 'video' | string
  /**
   * ck_point for checkpoint models (default: checkpoint)
   */
  ckPoint?: 'checkpoint' | string
  /**
   * Restrict models by backend "model_family" (comma-separated), e.g. "sdxl,flux".
   * This is forwarded to /api/models as query param model_family.
   */
  modelFamily?: string | Ref<string>
  perPage?: number
}

export function useCheckpointModels(opts: UseCheckpointModelsOptions) {
  const taskType = opts.taskType ?? 'image'
  const ckPoint = opts.ckPoint ?? 'checkpoint'
  const cacheKey = opts.cacheKey ?? DEFAULT_MODEL_LAST_SELECTED_CACHE_KEY
  const cacheTtlMs = opts.cacheTtlMs ?? 30 * 24 * 60 * 60 * 1000
  const onlyEditable = Boolean(opts.onlyEditable)
  const fixedModelFamily = computed(() => {
    const mf = opts.modelFamily
    const raw = typeof mf === 'string' ? mf : mf?.value
    return String(raw || '').trim()
  })

  const modelOptions = ref<UiModelOption[]>([])
  const modelLoading = ref(false)
  const modelTotal = ref(0)
  const modelMeta = ref<any>(null)
  const modelPage = ref(1)
  const modelPerPage = ref(Math.max(1, Number(opts.perPage || 8)))
  const modelSearchKeyword = ref('')
  const modelStorageFilter = ref<string | undefined>(undefined)
  const familyFilter = ref<string | undefined>(undefined)

  const cachedCheckpointModel = ref<UiModelOption | null>(null)

  const normalizeCachedModel = (model: UiModelOption): UiModelOption => {
    const modelKey = String(model.model_key || model.value || '').trim()
    return {
      ...model,
      value: String(model.value || modelKey),
      model_key: modelKey,
    }
  }

  const selectedModelExists = computed(() => {
    if (!opts.modelKey.value) return false
    return modelOptions.value.some((m) => m.value === opts.modelKey.value)
  })

  const selectedCheckpointFamily = computed(() => {
    const meta = modelOptions.value.find((m) => m.value === opts.modelKey.value)
    return String(meta?.family || '').trim()
  })

  const selectedModelConfig = computed(() => {
    const meta = modelOptions.value.find((m) => m.value === opts.modelKey.value)
    return (meta as any)?.runtime_config
  })

  async function fetchModelByKey(key: string): Promise<any | null> {
    const modelKey = String(key || '').trim()
    if (!modelKey) return null
    try {
      const resp = await getRaw<any>(`/models/${encodeURIComponent(modelKey)}`)
      return resp?.data || null
    } catch {
      return null
    }
  }

  const fetchModels = async () => {
    modelLoading.value = true
    try {
      const offset = (modelPage.value - 1) * modelPerPage.value
      const kw = modelSearchKeyword.value.trim()
      const nameQuery = kw ? `&name=${encodeURIComponent(kw)}` : ''
      const storageQuery = modelStorageFilter.value ? `&storage_mode=${encodeURIComponent(modelStorageFilter.value)}` : ''
      const classQuery = familyFilter.value ? `&family=${encodeURIComponent(familyFilter.value)}` : ''
      const editableQuery = onlyEditable ? `&editable=1` : ''
      const familyQuery = fixedModelFamily.value ? `&model_family=${encodeURIComponent(fixedModelFamily.value)}` : ''

      const resp = await getRaw<any[]>(
        `/models?modality=${encodeURIComponent(String(taskType))}&service_status=active&asset_type=${encodeURIComponent(
          String(ckPoint)
        )}${editableQuery}${familyQuery}&limit=${modelPerPage.value}&offset=${offset}${nameQuery}${storageQuery}${classQuery}`
      )

      const models = Array.isArray(resp?.data) ? resp.data : []
      modelTotal.value = Number(resp?.meta?.total || models.length || 0)
      modelMeta.value = resp?.meta || null
      modelOptions.value = models.map((m) => ({
        value: String(m?.model_key || ''),
        model_key: String(m?.model_key || ''),
        label: String(m?.name || m?.load_name || m?.model_key || ''),
        name: String(m?.name || m?.load_name || m?.model_key || ''),
        load_name: m?.load_name ? String(m.load_name) : undefined,
        thumb: (m?.thumb || m?.thumbnail || m?.cover || '') as string,
        storage_mode: m?.storage_mode ? String(m.storage_mode) : '',
        family: m?.family ? String(m.family) : '',
        asset_type: String(m?.asset_type || String(ckPoint)),
        capabilities: (m as any)?.capabilities ?? undefined,
        runtime_config: (m as any)?.runtime_config ?? undefined,
        video_profile: (m as any)?.video_profile ?? (m as any)?.runtime_config?.video_profile ?? undefined,
      }))

      // Merge cached model (so selector can display even if it's not in current page)
      const cached = cachedCheckpointModel.value ? normalizeCachedModel(cachedCheckpointModel.value) : null
      if (cached?.value && !modelOptions.value.some((m) => m.value === cached.value)) {
        if (!onlyEditable) {
          modelOptions.value = [cached, ...modelOptions.value]
        } else {
          // editable pages: validate cached model by model_key and only merge if editable
          const md = await fetchModelByKey(cached.value)
          const editable = Boolean(md?.capabilities?.editable)
          if (editable) {
            const patched: UiModelOption = {
              value: String(md?.model_key || cached.value),
              model_key: String(md?.model_key || cached.model_key || cached.value),
              label: String(md?.name || md?.load_name || cached.label || cached.value),
              name: String(md?.name || md?.load_name || cached.name || cached.label || cached.value),
              load_name: md?.load_name ? String(md.load_name) : cached.load_name,
              thumb: (md?.thumb || '') as string,
              storage_mode: md?.storage_mode ? String(md.storage_mode) : cached.storage_mode,
              family: md?.family ? String(md.family) : cached.family,
              asset_type: String(md?.asset_type || cached.asset_type || String(ckPoint)),
              capabilities: { ...(md?.capabilities || {}), editable: true },
              runtime_config: md?.runtime_config ?? (cached as any)?.runtime_config ?? undefined,
              video_profile:
                md?.video_profile ??
                md?.runtime_config?.video_profile ??
                (cached as any)?.video_profile ??
                (cached as any)?.runtime_config?.video_profile ??
                undefined,
            }
            modelOptions.value = [patched, ...modelOptions.value]
            cachedCheckpointModel.value = patched
          } else {
            // cached model is not editable anymore
            cachedCheckpointModel.value = null
          }
        }
      }

      // Ensure there is a selected model if possible (keep behavior: only when empty)
      if (!opts.modelKey.value && modelOptions.value.length > 0) {
        opts.modelKey.value = modelOptions.value[0]!.value
      }
    } catch (e) {
      console.error('Failed to fetch models:', e)
      modelOptions.value = []
      modelTotal.value = 0
    } finally {
      modelLoading.value = false
    }
  }

  const handleModelPageChange = ({ page, perPage }: { page: number; perPage: number }) => {
    modelPerPage.value = perPage
    modelPage.value = page
    fetchModels()
  }
  const handleModelSearchChange = (keyword: string) => {
    modelSearchKeyword.value = keyword || ''
    modelPage.value = 1
    fetchModels()
  }
  const handleModelFilterChange = (payload: { storageType?: string; family?: string }) => {
    modelStorageFilter.value = payload.storageType
    familyFilter.value = payload.family
    modelPage.value = 1
    fetchModels()
  }

  const restoreCachedModel = () => {
    const cachedModelRaw = getLocalCache<UiModelOption>(cacheKey)
    const cachedModel = cachedModelRaw?.value ? normalizeCachedModel(cachedModelRaw) : null
    if (cachedModel?.value) {
      cachedCheckpointModel.value = cachedModel
      modelOptions.value = [cachedModel]
      modelTotal.value = 1
      modelMeta.value = null
      opts.modelKey.value = cachedModel.value
    }
    return cachedModel
  }

  const persistSelectedModel = (model: UiModelOption) => {
    if (!model?.value) return
    setLocalCache(cacheKey, model, { ttlMs: cacheTtlMs })
    cachedCheckpointModel.value = model
  }

  return {
    // reactive state
    modelOptions,
    modelLoading,
    modelTotal,
    modelMeta,
    modelPage,
    modelPerPage,
    modelSearchKeyword,
    modelStorageFilter,
    familyFilter,
    cachedCheckpointModel,

    // computed
    selectedModelExists,
    selectedCheckpointFamily,
    selectedModelConfig,

    // actions
    fetchModels,
    handleModelPageChange,
    handleModelSearchChange,
    handleModelFilterChange,
    restoreCachedModel,
    persistSelectedModel,
  }
}

