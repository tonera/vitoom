const INVALID_TOKEN_VALUES = new Set(['', 'undefined', 'null'])

function normalizeToken(value: string | null): string | null {
  const normalized = String(value ?? '').trim()
  if (INVALID_TOKEN_VALUES.has(normalized)) {
    return null
  }
  return normalized
}

export function getAccessToken(): string | null {
  const token = normalizeToken(localStorage.getItem('token'))
  if (!token) {
    localStorage.removeItem('token')
  }
  return token
}

export function getRefreshToken(): string | null {
  const token = normalizeToken(localStorage.getItem('refresh_token'))
  if (!token) {
    localStorage.removeItem('refresh_token')
  }
  return token
}

export function hasValidAccessToken(): boolean {
  return Boolean(getAccessToken())
}

export function setAccessToken(token: string): void {
  const normalized = normalizeToken(token)
  if (!normalized) {
    throw new Error('Invalid access token')
  }
  localStorage.setItem('token', normalized)
}

export function setRefreshToken(token: string): void {
  const normalized = normalizeToken(token)
  if (!normalized) {
    throw new Error('Invalid refresh token')
  }
  localStorage.setItem('refresh_token', normalized)
}

export function clearAuthTokens(): void {
  localStorage.removeItem('token')
  localStorage.removeItem('refresh_token')
}
