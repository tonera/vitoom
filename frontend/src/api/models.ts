import { del, getRaw, post, postRaw, put } from '../utils/api'

export type ModelCkType = 'checkpoint' | 'lora' | string
export type ModelStatus = 'active' | 'inactive' | string
export type ModelTaskType = 'image' | 'video' | 'audio' | 'text' | 'mini' | 'translate' | string
export type ModelStorageType = 'local' | 'cloud' | string

export interface ModelRecord {
  id: string
  model_key: string
  name: string
  modality: ModelTaskType
  asset_type?: ModelCkType
  family?: string | null
  capabilities?: Record<string, any>
  runtime_engine?: string | null
  runtime_config?: Record<string, any>
  load_name?: string | null
  service_status: ModelStatus
  storage_mode: ModelStorageType
  download_status?: string | null
  source?: Record<string, any>
  thumb?: string | null
  description?: string | null
  tags?: string[]
  trigger_words?: string[]
  created_at?: string | null
  updated_at?: string | null
  deleted_at?: string | null
  [k: string]: any
}

export type ListModelsParams = {
  modality?: string
  storage_mode?: string
  service_status?: string
  name?: string
  asset_type?: string
  family?: string
  lora_family?: string
  editable?: 0 | 1
  limit?: number
  offset?: number
}

export type ModelsListMeta = {
  current_page?: number
  from?: number
  last_page?: number
  per_page?: number
  to?: number
  total?: number
  filter_options?: {
    storage_modes?: string[]
    families?: string[]
    asset_types?: string[]
  }
}

export type CatalogOption = {
  value: string
  label: string
}

export type ModelsCatalogMeta = {
  modalities: CatalogOption[]
  families: CatalogOption[]
  modality_ids?: string[]
  family_ids?: string[]
}

export async function getModelsMeta() {
  const res = await getRaw<ModelsCatalogMeta>('/models/meta')
  return (res.data ?? res) as ModelsCatalogMeta
}

export async function listModels(params: ListModelsParams) {
  const qs = new URLSearchParams()
  for (const [k, v] of Object.entries(params || {})) {
    if (v === undefined || v === null) continue
    const s = String(v).trim()
    if (!s) continue
    qs.set(k, s)
  }
  const url = qs.toString() ? `/models?${qs.toString()}` : '/models'
  return await getRaw<ModelRecord[]>(url)
}

export async function getModel(modelKey: string) {
  const key = encodeURIComponent(String(modelKey || '').trim())
  return await getRaw<ModelRecord>(`/models/${key}`)
}

export type CreateModelBody = {
  model_key?: string | null
  name: string
  modality: ModelTaskType
  asset_type?: ModelCkType
  family?: string | null
  capabilities?: Record<string, any>
  runtime_engine?: string | null
  runtime_config?: Record<string, any>
  load_name: string
  service_status?: ModelStatus
  storage_mode: ModelStorageType
  download_status?: string
  source?: Record<string, any>
  description?: string | null
  thumb?: string | null
  tags?: string[]
  trigger_words?: string[]
}

export async function createModel(body: CreateModelBody) {
  return await post<ModelRecord>('/models', body)
}

export async function createModelWithMeta(body: CreateModelBody) {
  const res = await postRaw<ModelRecord>('/models', body)
  return {
    record: (res.data ?? res) as ModelRecord,
    msg: String(res.msg || '').trim(),
  }
}

export type UpdateModelBody = Partial<CreateModelBody>

export async function updateModel(modelKey: string, body: UpdateModelBody) {
  const key = encodeURIComponent(String(modelKey || '').trim())
  return await put<ModelRecord>(`/models/${key}`, body)
}

export async function deleteModel(modelKey: string) {
  const key = encodeURIComponent(String(modelKey || '').trim())
  return await del<{ model_key: string }>(`/models/${key}`)
}

export async function activateModel(modelKey: string) {
  const key = encodeURIComponent(String(modelKey || '').trim())
  return await put<ModelRecord>(`/models/${key}/activate`)
}

export async function deactivateModel(modelKey: string) {
  const key = encodeURIComponent(String(modelKey || '').trim())
  return await put<ModelRecord>(`/models/${key}/deactivate`)
}

export type DownloadActionBody = {
  action: 'start' | 'cancel' | 'refresh' | string
  source?: Record<string, any>
  asset_type?: string
}

export async function downloadActionModel(modelKey: string, body: DownloadActionBody) {
  const key = encodeURIComponent(String(modelKey || '').trim())
  return await post<any>(`/models/${key}/download/action`, body, { timeout: 0 })
}

export type RemoteModelInfoResponse = {
  provider: 'huggingface' | 'modelscope' | string
  repo_id: string
  info: any
  thumb_candidates?: string[]
}

export async function getRemoteModelInfo(input: string) {
  return await post<RemoteModelInfoResponse>('/models/remote-info', { input: String(input || '').trim() })
}
