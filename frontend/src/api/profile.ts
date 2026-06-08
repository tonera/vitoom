import { put } from '../utils/api'
import type { CurrentUser } from '../utils/currentUser'

export function updateProfile(body: {
  nickname?: string | null
  new_password?: string
}): Promise<CurrentUser> {
  return put<CurrentUser>('/auth/me', body)
}
