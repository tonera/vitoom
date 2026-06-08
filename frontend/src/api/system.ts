import { get } from '../utils/api'

export interface HealthInfo {
  status: string
  service: string
  version: string
}

function parseHealthPayload(payload: unknown): HealthInfo | null {
  if (!payload || typeof payload !== 'object') return null
  const root = payload as Record<string, unknown>
  const data = (root.data ?? root) as Record<string, unknown>
  const version = typeof data.version === 'string' ? data.version.trim() : ''
  if (!version) return null
  return {
    status: typeof data.status === 'string' ? data.status : '',
    service: typeof data.service === 'string' ? data.service : '',
    version,
  }
}

/** 优先同源 /api/health，避免 config.json 中 backendOrigin 配错导致版本无法显示。 */
async function fetchHealthSameOrigin(): Promise<HealthInfo | null> {
  try {
    const resp = await fetch('/api/health', { cache: 'no-store' })
    if (!resp.ok) return null
    return parseHealthPayload(await resp.json())
  } catch {
    return null
  }
}

export async function fetchHealth(): Promise<HealthInfo> {
  const sameOrigin = await fetchHealthSameOrigin()
  if (sameOrigin) return sameOrigin

  const remote = parseHealthPayload(await get<HealthInfo>('/health'))
  if (remote) return remote

  throw new Error('Failed to fetch app version')
}

export function getBuildTimeVersion(): string {
  const version = (import.meta.env.VITE_APP_VERSION as string | undefined)?.trim()
  return version || ''
}
