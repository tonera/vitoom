import { get, post } from '../utils/api'
import { getV1BasePath } from '../utils/runtimeConfig'

export interface AgentRecord {
  id: string
  name: string
  description?: string | null
  type: string
  config: Record<string, any>
  status: string
  is_preset?: boolean
  created_at?: string | null
  updated_at?: string | null
}

export interface AgentRunRecord {
  id: string
  user_id: string
  agent_id: string
  task_id: string
  source_type: string
  source_ref?: string | null
  status: string
  input_payload: Record<string, any>
  runtime_config: Record<string, any>
  result_summary?: string | null
  error_message?: string | null
  started_at?: string | null
  completed_at?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export interface AgentRunCreateRequest {
  agent_id: string
  message: string
  source_type?: string
  source_ref?: string | null
  attachments?: Array<Record<string, any>>
  context?: Record<string, any>
  runtime_config?: Record<string, any>
}

export interface AgentRunCreateResponse {
  run_id: string
  task_id: string
  status: string
}

function getAgentsBasePath(): string {
  return `${getV1BasePath()}/agents`
}

export async function listAgents(params: { status?: string; is_preset?: boolean } = {}) {
  const qs = new URLSearchParams()
  if (params.status) qs.set('status', params.status)
  if (params.is_preset !== undefined) qs.set('is_preset', String(params.is_preset))
  const suffix = qs.toString() ? `?${qs.toString()}` : ''
  return await get<{ agents: AgentRecord[]; total: number }>(`${getAgentsBasePath()}${suffix}`)
}

export async function getAgent(agentId: string) {
  const normalized = encodeURIComponent(String(agentId || '').trim())
  return await get<AgentRecord>(`${getAgentsBasePath()}/${normalized}`)
}

export async function createAgentRun(payload: AgentRunCreateRequest) {
  return await post<AgentRunCreateResponse>(`${getAgentsBasePath()}/runs`, {
    source_type: 'web',
    attachments: [],
    context: {},
    runtime_config: {},
    ...payload,
  })
}

export async function getAgentRun(runId: string) {
  const normalized = encodeURIComponent(String(runId || '').trim())
  return await get<AgentRunRecord>(`${getAgentsBasePath()}/runs/${normalized}`)
}

export async function listAgentRuns(params: { status?: string; limit?: number; offset?: number } = {}) {
  const qs = new URLSearchParams()
  if (params.status) qs.set('status', params.status)
  if (params.limit !== undefined) qs.set('limit', String(params.limit))
  if (params.offset !== undefined) qs.set('offset', String(params.offset))
  const suffix = qs.toString() ? `?${qs.toString()}` : ''
  return await get<{ runs: AgentRunRecord[]; total: number }>(`${getAgentsBasePath()}/runs${suffix}`)
}

export async function cancelAgentRun(runId: string) {
  const normalized = encodeURIComponent(String(runId || '').trim())
  return await post<AgentRunCreateResponse>(`${getAgentsBasePath()}/runs/${normalized}/cancel`)
}
