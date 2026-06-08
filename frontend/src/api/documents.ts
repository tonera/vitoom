import { get, post } from '../utils/api'
import { getBackendRequestBaseURL } from '../utils/runtimeConfig'

export type DocumentToTextResponse = {
  text: string
  source_url?: string
  provider?: string
  status?: string
}

export type DocumentConvertConfig = {
  timeout_seconds: number
  pdf_ocr_model: string
}

const DEFAULT_CONVERT_TIMEOUT_MS = 660_000

export async function getDocumentConvertConfig(): Promise<DocumentConvertConfig> {
  const resp = await get<DocumentConvertConfig>('/v1/documents/convert-config', {
    baseURL: getBackendRequestBaseURL(),
  })
  return resp
}

export async function documentToText(url: string, timeoutMs?: number): Promise<DocumentToTextResponse> {
  let effectiveTimeout = timeoutMs
  if (effectiveTimeout == null) {
    try {
      const cfg = await getDocumentConvertConfig()
      const seconds = Number(cfg?.timeout_seconds)
      effectiveTimeout = Number.isFinite(seconds) && seconds > 0
        ? Math.ceil(seconds * 1000 + 60_000)
        : DEFAULT_CONVERT_TIMEOUT_MS
    } catch {
      effectiveTimeout = DEFAULT_CONVERT_TIMEOUT_MS
    }
  }

  const resp = await post<DocumentToTextResponse>(
    '/v1/documents/to-text',
    { url: String(url || '').trim() },
    {
      baseURL: getBackendRequestBaseURL(),
      timeout: effectiveTimeout,
    },
  )
  return resp
}
