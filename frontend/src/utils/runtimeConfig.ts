export type RuntimeConfig = {
  /**
   * 后端服务的 Origin（包含协议 + host + port），例如：
   * - "http://127.0.0.1:8888"
   * - "https://demo.example.com"
   *
   * 为空时表示“同源部署/反代”，即前端使用当前网页的 origin。
   */
  backendOrigin?: string

  /** 默认 "/api" */
  apiBasePath?: string
  /** 默认 "/v1" */
  v1BasePath?: string
  /** 默认 "/outputs" */
  outputsBasePath?: string
  /** 默认 "/ws" */
  wsBasePath?: string
}

let cachedConfig: RuntimeConfig | null = null

function isNonEmptyString(v: any): v is string {
  return typeof v === 'string' && v.trim().length > 0
}

function normalizeBasePath(v: any, fallback: string): string {
  const s = typeof v === 'string' ? v.trim() : ''
  const raw = s || fallback
  const withLeading = raw.startsWith('/') ? raw : `/${raw}`
  return withLeading.replace(/\/+$/, '') || fallback
}

function normalizeOrigin(v: any): string {
  if (!isNonEmptyString(v)) return ''
  const s = v.trim().replace(/\/+$/, '')
  try {
    // Validate
    // eslint-disable-next-line no-new
    new URL(s)
    return s
  } catch {
    return ''
  }
}

function sanitizeConfig(raw: any): RuntimeConfig {
  const backendOrigin = normalizeOrigin(raw?.backendOrigin)
  return {
    backendOrigin,
    apiBasePath: normalizeBasePath(raw?.apiBasePath, '/api'),
    v1BasePath: normalizeBasePath(raw?.v1BasePath, '/v1'),
    outputsBasePath: normalizeBasePath(raw?.outputsBasePath, '/outputs'),
    wsBasePath: normalizeBasePath(raw?.wsBasePath, '/ws'),
  }
}

/**
 * 运行时加载配置（用于“已 build 后用户只改一个文件”场景）
 * - 默认从站点根目录读取 `/config.json`
 * - 读取失败则回退到空配置（同源部署）
 */
export async function loadRuntimeConfig(): Promise<RuntimeConfig> {
  if (cachedConfig) return cachedConfig
  try {
    const base = import.meta.env.BASE_URL || '/'
    const configUrl = `${base.endsWith('/') ? base : `${base}/`}config.json`
    const resp = await fetch(configUrl, { cache: 'no-store' })
    if (!resp.ok) {
      cachedConfig = sanitizeConfig({})
      return cachedConfig
    }
    const json = await resp.json()
    cachedConfig = sanitizeConfig(json)
    return cachedConfig
  } catch {
    cachedConfig = sanitizeConfig({})
    return cachedConfig
  }
}

export function getRuntimeConfigSync(): RuntimeConfig {
  return cachedConfig ?? sanitizeConfig({})
}

export function getBackendOrigin(): string {
  return getRuntimeConfigSync().backendOrigin || ''
}

export function getBackendRequestBaseURL(): string {
  // axios 的 baseURL：如果配置了后端 origin，就用它；否则为空（保持同源相对路径）
  return getBackendOrigin()
}

export function getApiBaseURL(): string {
  const cfg = getRuntimeConfigSync()
  const apiPath = normalizeBasePath(cfg.apiBasePath, '/api')
  const origin = cfg.backendOrigin || ''
  // 支持两种模式：
  // 1) 同源部署/反代：baseURL="/api"
  // 2) 前后端分离：baseURL="http(s)://host:port/api"
  return origin ? `${origin}${apiPath}` : apiPath
}

export function getV1BasePath(): string {
  return normalizeBasePath(getRuntimeConfigSync().v1BasePath, '/v1')
}

export function getOutputsBaseURL(): string {
  const cfg = getRuntimeConfigSync()
  const outputsPath = normalizeBasePath(cfg.outputsBasePath, '/outputs')
  const origin = cfg.backendOrigin || ''
  return origin ? `${origin}${outputsPath}` : outputsPath
}

export function getWsHost(): string {
  // WebSocketClient 需要的是 host（不含协议），例如 "127.0.0.1:8888"
  const origin = getBackendOrigin()
  if (!origin) return ''
  try {
    return new URL(origin).host
  } catch {
    return ''
  }
}

export function getWsBasePath(): string {
  return normalizeBasePath(getRuntimeConfigSync().wsBasePath, '/ws')
}

/**
 * 将后端返回的相对 URL（例如 "/outputs/xxx.png"）在“前后端分离”场景下补全为绝对 URL。
 * - 若已是绝对 URL，则原样返回
 * - 若以 "/" 开头且配置了 backendOrigin，则拼接 backendOrigin
 */
export function resolveBackendPublicUrl(input: string): string {
  const s = (input || '').trim()
  if (!s) return s
  if (/^(https?:)?\/\//i.test(s)) return s
  if (/^data:/i.test(s)) return s
  if (s.startsWith('/')) {
    const origin = getBackendOrigin()
    return origin ? `${origin}${s}` : s
  }
  return s
}


