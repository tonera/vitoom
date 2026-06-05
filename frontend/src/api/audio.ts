import type { AxiosRequestConfig } from 'axios'
import { get } from '../utils/api'
import { getBackendRequestBaseURL, getV1BasePath } from '../utils/runtimeConfig'

function v1HttpConfig(): AxiosRequestConfig {
  const origin = getBackendRequestBaseURL().trim()
  return { baseURL: origin ? origin.replace(/\/+$/, '') : '' }
}

function v1Path(suffix: string): string {
  const root = getV1BasePath().replace(/\/+$/, '') || '/v1'
  const s = suffix.startsWith('/') ? suffix : `/${suffix}`
  return `${root}${s}`
}

export interface TtsSpeakerOption {
  name: string
  label?: string
  description?: string
  language?: string
  reference_audio?: string
}

export interface TtsSpeakerFamily {
  default_speaker?: string
  speakers?: TtsSpeakerOption[]
}

export interface TtsSpeakerCatalog {
  version?: number
  families?: Record<string, TtsSpeakerFamily>
}

export function listTtsSpeakers(): Promise<TtsSpeakerCatalog> {
  return get<TtsSpeakerCatalog>(v1Path('/audio/tts-speakers'), v1HttpConfig())
}
