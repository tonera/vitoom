export type LocalCacheEnvelope<T> = {
  v: number
  ts: number
  ttlMs?: number
  data: T
}

function hasLocalStorage(): boolean {
  try {
    return typeof window !== 'undefined' && typeof window.localStorage !== 'undefined'
  } catch {
    return false
  }
}

export function setLocalCache<T>(key: string, data: T, opts: { ttlMs?: number } = {}): void {
  if (!hasLocalStorage()) return
  const env: LocalCacheEnvelope<T> = {
    v: 1,
    ts: Date.now(),
    ttlMs: opts.ttlMs,
    data,
  }
  try {
    window.localStorage.setItem(key, JSON.stringify(env))
  } catch {
    // ignore quota/serialization errors
  }
}

export function getLocalCache<T>(key: string): T | undefined {
  if (!hasLocalStorage()) return undefined
  try {
    const raw = window.localStorage.getItem(key)
    if (!raw) return undefined
    const env = JSON.parse(raw) as Partial<LocalCacheEnvelope<T>>
    if (!env || typeof env !== 'object') return undefined
    const ts = typeof env.ts === 'number' ? env.ts : 0
    const ttlMs = typeof env.ttlMs === 'number' ? env.ttlMs : undefined
    if (ttlMs && ts && Date.now() - ts > ttlMs) {
      window.localStorage.removeItem(key)
      return undefined
    }
    return env.data as T
  } catch {
    return undefined
  }
}

export function removeLocalCache(key: string): void {
  if (!hasLocalStorage()) return
  try {
    window.localStorage.removeItem(key)
  } catch {
    // ignore
  }
}


