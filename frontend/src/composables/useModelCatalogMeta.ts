import { computed, ref, shallowRef } from 'vue'
import { getModelsMeta, type CatalogOption, type ModelsCatalogMeta } from '../api/models'

const cachedMeta = shallowRef<ModelsCatalogMeta | null>(null)
let inflight: Promise<ModelsCatalogMeta | null> | null = null

/** API 不可用时的 modality 兜底（family 仍以 API 为准，失败则为空列表）。 */
const FALLBACK_MODALITY_OPTIONS: CatalogOption[] = [
  { value: 'image', label: 'image' },
  { value: 'video', label: 'video' },
  { value: 'audio', label: 'audio' },
  { value: 'text', label: 'text' },
  { value: 'mini', label: 'mini' },
  { value: 'translate', label: 'translate' },
]

function normalizeFamilyWithOptions(input: string, familyOptions: ReadonlyArray<CatalogOption>): string {
  const raw = String(input || '').trim()
  if (!raw) return ''

  const exact = familyOptions.find((x) => x.value === raw)
  if (exact) return exact.value

  const norm = (s: string) =>
    String(s || '')
      .trim()
      .toLowerCase()
      .replace(/[\s._-]+/g, '-')
      .replace(/-+/g, '-')
      .replace(/^-|-$/g, '')

  const key = norm(raw)
  const hit = familyOptions.find((x) => norm(x.value) === key)
  if (hit) return hit.value

  return raw
}

function resolveOptions(meta: ModelsCatalogMeta | null, kind: 'modalities' | 'families'): CatalogOption[] {
  const rows = meta?.[kind]
  if (Array.isArray(rows) && rows.length > 0) {
    return rows
      .map((x) => ({
        value: String(x?.value || '').trim(),
        label: String(x?.label || x?.value || '').trim(),
      }))
      .filter((x) => x.value)
  }
  return kind === 'modalities' ? [...FALLBACK_MODALITY_OPTIONS] : []
}

export function useModelCatalogMeta() {
  const loading = ref(false)
  const error = ref<string | null>(null)

  const modalityOptions = computed(() => resolveOptions(cachedMeta.value, 'modalities'))
  const familyOptions = computed(() => resolveOptions(cachedMeta.value, 'families'))
  const modalityFilterOptions = computed(() => modalityOptions.value.map((x) => x.value))
  const familyFilterOptions = computed(() => familyOptions.value.map((x) => x.value))

  async function ensureLoaded(force = false): Promise<ModelsCatalogMeta | null> {
    if (!force && cachedMeta.value) return cachedMeta.value
    if (!force && inflight) return inflight

    loading.value = true
    error.value = null

    inflight = getModelsMeta()
      .then((meta) => {
        cachedMeta.value = meta
        return meta
      })
      .catch((e: any) => {
        error.value = String(e?.message || e || 'failed to load model catalog meta')
        console.warn('[useModelCatalogMeta]', error.value)
        return null
      })
      .finally(() => {
        loading.value = false
        inflight = null
      })

    return inflight
  }

  function normalizeFamily(input: string): string {
    return normalizeFamilyWithOptions(input, familyOptions.value)
  }

  return {
    meta: cachedMeta,
    loading,
    error,
    modalityOptions,
    familyOptions,
    modalityFilterOptions,
    familyFilterOptions,
    ensureLoaded,
    normalizeFamily,
  }
}
