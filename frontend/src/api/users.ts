import { del, get, post, put } from '../utils/api'

export type AdminUserStatus = 'active' | 'disabled'

export interface AdminUser {
  id: string
  email: string
  nickname?: string | null
  status: AdminUserStatus
  is_admin: boolean
  created_at?: string | null
  updated_at?: string | null
}

export interface AdminUserListResponse {
  items: AdminUser[]
  total: number
}

export function listUsers(params?: {
  keyword?: string
  limit?: number
  offset?: number
}): Promise<AdminUserListResponse> {
  return get<AdminUserListResponse>('/admin/users', { params })
}

export function createUser(body: {
  email: string
  password: string
  nickname?: string
  status?: AdminUserStatus
  is_admin?: boolean
}): Promise<AdminUser> {
  return post<AdminUser>('/admin/users', body)
}

export function updateUser(
  id: string,
  body: {
    email?: string
    password?: string
    nickname?: string | null
    status?: AdminUserStatus
    is_admin?: boolean
  }
): Promise<AdminUser> {
  return put<AdminUser>(`/admin/users/${encodeURIComponent(id)}`, body)
}

export function disableUser(id: string): Promise<{ id: string; status: string }> {
  return del<{ id: string; status: string }>(`/admin/users/${encodeURIComponent(id)}`)
}
