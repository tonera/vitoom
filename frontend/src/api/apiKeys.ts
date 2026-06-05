import { del, get, post } from '../utils/api'

export type ApiKeyExpiration = '1d' | '1m' | '1y' | 'never'

export interface UserApiKey {
  id: string
  name: string
  key_prefix: string
  expires_at?: string | null
  last_used_at?: string | null
  created_at?: string | null
  is_expired?: boolean
}

export interface CreatedUserApiKey extends UserApiKey {
  key: string
}

export function listApiKeys(): Promise<{ items: UserApiKey[] }> {
  return get<{ items: UserApiKey[] }>('/api-keys')
}

export function createApiKey(body: {
  name?: string
  expires_in: ApiKeyExpiration
}): Promise<CreatedUserApiKey> {
  return post<CreatedUserApiKey>('/api-keys', body)
}

export function deleteApiKey(id: string): Promise<{ id: string }> {
  return del<{ id: string }>(`/api-keys/${encodeURIComponent(id)}`)
}
