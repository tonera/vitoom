import { get } from './api'

export interface CurrentUser {
  id: string
  email: string
  nickname?: string | null
  status: string
  is_admin: boolean
  created_at?: string
}

let cachedUser: CurrentUser | null = null
let pending: Promise<CurrentUser | null> | null = null

export async function fetchCurrentUser(force = false): Promise<CurrentUser | null> {
  if (cachedUser && !force) {
    return cachedUser
  }

  if (pending && !force) {
    return pending
  }

  pending = get<CurrentUser>('/auth/me')
    .then((user) => {
      cachedUser = user
      return user
    })
    .catch(() => {
      cachedUser = null
      return null
    })
    .finally(() => {
      pending = null
    })

  return pending
}

export function getCachedCurrentUser(): CurrentUser | null {
  return cachedUser
}

export function clearCurrentUserCache(): void {
  cachedUser = null
  pending = null
}

export async function isCurrentUserAdmin(force = false): Promise<boolean> {
  const user = await fetchCurrentUser(force)
  return Boolean(user?.is_admin)
}
