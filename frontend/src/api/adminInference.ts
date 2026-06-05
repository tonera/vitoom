import { del, get, post, put } from '../utils/api'

export type InferenceRuntimeState = 'online' | 'degraded' | 'offline' | 'agent_unreachable' | 'unknown'

export interface AdminInferenceService {
  id: string
  name: string
  type: string
  service_type?: string | null
  status: string
  host?: string | null
  port?: number | null
  client_ip?: string | null
  config?: Record<string, unknown>
  program_name?: string
  supervisor_url?: string
  control_available?: boolean
  control_unavailable_reason?: string | null
  agent_reachable?: boolean
  agent_detail?: string | null
  agent_programs_error?: unknown
  agent_programs_error_status?: number | null
  supervisor_program?: {
    name?: string
    state?: string
    description?: string
    pid?: number | null
    uptime?: string | null
  } | null
  ws_online?: boolean
  heartbeat_fresh?: boolean
  runtime_state?: InferenceRuntimeState
  last_heartbeat_at?: string | null
}

export interface AdminInferenceServiceListResponse {
  items: AdminInferenceService[]
  total: number
}

export interface AdminInferenceLogsResponse {
  name: string
  stream: 'stdout' | 'stderr'
  lines: string[]
}

export interface AdminInferenceConfigDocument {
  path?: string | null
  config: Record<string, unknown>
}

export function getAdminInferenceServiceConfig(id: string): Promise<AdminInferenceConfigDocument> {
  return get<AdminInferenceConfigDocument>(`/admin/inference/services/${encodeURIComponent(id)}/config`)
}

export function updateAdminInferenceServiceConfig(
  id: string,
  config: Record<string, unknown>
): Promise<AdminInferenceConfigDocument> {
  return put<AdminInferenceConfigDocument>(`/admin/inference/services/${encodeURIComponent(id)}/config`, { config })
}

export function getAdminInferenceGlobalConfig(id: string): Promise<AdminInferenceConfigDocument> {
  return get<AdminInferenceConfigDocument>(`/admin/inference/services/${encodeURIComponent(id)}/global-config`)
}

export function updateAdminInferenceGlobalConfig(
  id: string,
  config: Record<string, unknown>
): Promise<AdminInferenceConfigDocument> {
  return put<AdminInferenceConfigDocument>(`/admin/inference/services/${encodeURIComponent(id)}/global-config`, {
    config,
  })
}

export function listAdminInferenceServices(): Promise<AdminInferenceServiceListResponse> {
  return get<AdminInferenceServiceListResponse>('/admin/inference/services')
}

export function deleteAdminInferenceService(id: string): Promise<{ service_id: string }> {
  return del<{ service_id: string }>(`/admin/inference/services/${encodeURIComponent(id)}`)
}

export function startAdminInferenceService(id: string) {
  return post(`/admin/inference/services/${encodeURIComponent(id)}/start`)
}

export function stopAdminInferenceService(id: string) {
  return post(`/admin/inference/services/${encodeURIComponent(id)}/stop`)
}

export function restartAdminInferenceService(id: string) {
  return post(`/admin/inference/services/${encodeURIComponent(id)}/restart`)
}

export function getAdminInferenceServiceLogs(
  id: string,
  params?: { stream?: 'stdout' | 'stderr'; tail?: number }
): Promise<AdminInferenceLogsResponse> {
  return get<AdminInferenceLogsResponse>(`/admin/inference/services/${encodeURIComponent(id)}/logs`, { params })
}

