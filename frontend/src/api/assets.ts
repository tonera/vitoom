import { del, get } from '../utils/api'
import { getBackendRequestBaseURL } from '../utils/runtimeConfig'

export type AssetCategory = 'image' | 'video' | 'audio' | 'text'

export interface UserAsset {
  id: string
  task_id?: string | null
  category: AssetCategory | string
  file_name?: string | null
  mime_type?: string | null
  file_size?: number | null
  created_at?: string | null
  storage_path?: string | null
  url?: string | null
  thumb_url?: string | null
  http_url?: string | null
  thumb_http_url?: string | null
  task_type?: string | null
  task_status?: string | null
  prompt?: string | null
  task_params?: Record<string, any>
}

export interface ListUserAssetsParams {
  category?: AssetCategory
  keyword?: string
  limit?: number
  offset?: number
}

export interface ListUserAssetsResponse {
  items: UserAsset[]
  total: number
}

export async function listUserAssets(params: ListUserAssetsParams = {}): Promise<ListUserAssetsResponse> {
  const qs = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) continue
    const s = String(value).trim()
    if (!s) continue
    qs.set(key, s)
  }
  const url = qs.toString() ? `/v1/user/files?${qs.toString()}` : '/v1/user/files'
  return await get<ListUserAssetsResponse>(url, { baseURL: getBackendRequestBaseURL() })
}

export async function deleteUserAsset(fileId: string): Promise<void> {
  const id = String(fileId || '').trim()
  if (!id) throw new Error('missing file id')
  await del(`/v1/user/files/${encodeURIComponent(id)}`, { baseURL: getBackendRequestBaseURL() })
}
