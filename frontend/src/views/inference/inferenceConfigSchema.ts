export type ConfigFieldType = 'string' | 'number' | 'password' | 'select' | 'tags'

export interface ConfigFieldDef {
  path: string
  type: ConfigFieldType
  labelKey: string
  options?: string[]
  placeholder?: string
  visibleWhen?: (values: Record<string, unknown>) => boolean
}

export const GLOBAL_CONFIG_FIELDS: ConfigFieldDef[] = [
  { path: 'api_base_url', type: 'string', labelKey: 'fields.apiBaseUrl' },
  { path: 'ws_url', type: 'string', labelKey: 'fields.wsUrl' },
  { path: 'supervisor_url', type: 'string', labelKey: 'fields.supervisorUrl' },
  { path: 'pipeline_cache_ttl_seconds', type: 'number', labelKey: 'fields.pipelineCacheTtlSeconds' },
  { path: 'models_dir', type: 'string', labelKey: 'fields.modelsDir' },
  { path: 'weights_dir', type: 'string', labelKey: 'fields.weightsDir' },
  { path: 'loras_dir', type: 'string', labelKey: 'fields.lorasDir' },
  {
    path: 'storage.default',
    type: 'select',
    labelKey: 'fields.storageDefault',
    options: ['local', 'server', 's3', 'oss'],
  },
  { path: 'storage.server.timeout_seconds', type: 'number', labelKey: 'fields.storageServerTimeoutSeconds' },
  { path: 'storage.server.auth.secret', type: 'password', labelKey: 'fields.storageServerAuthSecret' },
  { path: 'storage.s3.endpoint', type: 'string', labelKey: 'fields.storageS3Endpoint' },
  { path: 'storage.s3.region', type: 'string', labelKey: 'fields.storageS3Region' },
  { path: 'storage.s3.bucket', type: 'string', labelKey: 'fields.storageS3Bucket' },
  { path: 'storage.s3.access_key_id', type: 'string', labelKey: 'fields.storageS3AccessKeyId' },
  { path: 'storage.s3.secret_access_key', type: 'password', labelKey: 'fields.storageS3SecretAccessKey' },
  { path: 'storage.s3.public_base_url', type: 'string', labelKey: 'fields.storageS3PublicBaseUrl' },
  { path: 'storage.oss.endpoint', type: 'string', labelKey: 'fields.storageOssEndpoint' },
  { path: 'storage.oss.bucket', type: 'string', labelKey: 'fields.storageOssBucket' },
  { path: 'storage.oss.access_key_id', type: 'string', labelKey: 'fields.storageOssAccessKeyId' },
  { path: 'storage.oss.access_key_secret', type: 'password', labelKey: 'fields.storageOssAccessKeySecret' },
  { path: 'storage.oss.public_base_url', type: 'string', labelKey: 'fields.storageOssPublicBaseUrl' },
]

export const SERVICE_CONFIG_FIELDS: Record<string, ConfigFieldDef[]> = {
  download: [{ path: 'config.civitai_token', type: 'password', labelKey: 'fields.civitaiToken' }],
  image: [],
  video: [],
  mini: [
    {
      path: 'config.runtime.backend',
      type: 'select',
      labelKey: 'fields.runtimeBackend',
      options: ['transformers', 'vllm'],
    },
    { path: 'config.runtime.max_new_tokens', type: 'number', labelKey: 'fields.maxNewTokens' },
    {
      path: 'config.runtime.vllm.gpu_memory_utilization',
      type: 'number',
      labelKey: 'fields.vllmGpuMemoryUtilization',
      visibleWhen: (values) => values['config.runtime.backend'] === 'vllm',
    },
    {
      path: 'config.runtime.vllm.max_model_len',
      type: 'number',
      labelKey: 'fields.vllmMaxModelLen',
      visibleWhen: (values) => values['config.runtime.backend'] === 'vllm',
    },
  ],
  audio: [
    { path: 'config.capabilities', type: 'tags', labelKey: 'fields.capabilities' },
    { path: 'config.fixed_model', type: 'string', labelKey: 'fields.fixedModel' },
    { path: 'config.fixed_family', type: 'string', labelKey: 'fields.fixedFamily' },
    { path: 'config.runtime.vllm.gpu_memory_utilization', type: 'number', labelKey: 'fields.vllmGpuMemoryUtilization' },
    { path: 'config.runtime.vllm.max_model_len', type: 'number', labelKey: 'fields.vllmMaxModelLen' },
  ],
  text: [
    {
      path: 'config.runtime.backend',
      type: 'select',
      labelKey: 'fields.runtimeBackend',
      options: ['vllm', 'transformers', 'ollama'],
    },
    { path: 'config.runtime.max_model_len', type: 'number', labelKey: 'fields.maxModelLen' },
    { path: 'config.runtime.max_tokens', type: 'number', labelKey: 'fields.maxTokens' },
    {
      path: 'config.runtime.enable_thinking',
      type: 'select',
      labelKey: 'fields.enableThinking',
      options: ['true', 'false'],
    },
    {
      path: 'config.runtime.vllm.gpu_memory_utilization',
      type: 'number',
      labelKey: 'fields.vllmGpuMemoryUtilization',
      visibleWhen: (values) => values['config.runtime.backend'] === 'vllm',
    },
    {
      path: 'config.runtime.ollama.base_url',
      type: 'string',
      labelKey: 'fields.ollamaBaseUrl',
      visibleWhen: (values) => values['config.runtime.backend'] === 'ollama',
    },
    {
      path: 'config.runtime.ollama.keep_alive',
      type: 'string',
      labelKey: 'fields.ollamaKeepAlive',
      visibleWhen: (values) => values['config.runtime.backend'] === 'ollama',
    },
  ],
}

export function getNestedValue(source: Record<string, unknown>, path: string): unknown {
  return path.split('.').reduce<unknown>((current, key) => {
    if (!key || !current || typeof current !== 'object') return undefined
    return (current as Record<string, unknown>)[key]
  }, source)
}

export function setNestedValue(target: Record<string, unknown>, path: string, value: unknown): void {
  const keys = path.split('.')
  let cursor: Record<string, unknown> = target
  for (let index = 0; index < keys.length; index += 1) {
    const key = keys[index]!
    if (index === keys.length - 1) {
      cursor[key] = value
      return
    }
    const next = cursor[key]
    if (!next || typeof next !== 'object' || Array.isArray(next)) {
      cursor[key] = {}
    }
    cursor = cursor[key] as Record<string, unknown>
  }
}

export function buildPatchFromForm(
  fields: ConfigFieldDef[],
  values: Record<string, unknown>,
  original: Record<string, unknown>
): Record<string, unknown> {
  const patch: Record<string, unknown> = {}
  for (const field of fields) {
    if (field.visibleWhen && !field.visibleWhen(values)) continue
    const nextValue = values[field.path]
    const prevValue = getNestedValue(original, field.path)
    if (field.type === 'password' && (nextValue === '' || nextValue === '***')) {
      if (prevValue === '***') continue
      if (nextValue === '' && prevValue != null && prevValue !== '') continue
    }
    if (field.type === 'tags') {
      const normalized = Array.isArray(nextValue) ? nextValue : []
      setNestedValue(patch, field.path, normalized)
      continue
    }
    if (field.path === 'config.runtime.enable_thinking') {
      setNestedValue(patch, field.path, nextValue === 'true' || nextValue === true)
      continue
    }
    if (nextValue === undefined) continue
    setNestedValue(patch, field.path, nextValue)
  }
  return patch
}

export function formValuesFromConfig(
  fields: ConfigFieldDef[],
  config: Record<string, unknown>
): Record<string, unknown> {
  const values: Record<string, unknown> = {}
  for (const field of fields) {
    const raw = getNestedValue(config, field.path)
    if (field.type === 'tags') {
      values[field.path] = Array.isArray(raw) ? raw : []
    } else if (field.path === 'config.runtime.enable_thinking') {
      values[field.path] = raw === true ? 'true' : 'false'
    } else if (field.type === 'password') {
      values[field.path] = raw === '***' ? '' : raw ?? ''
    } else {
      values[field.path] = raw ?? ''
    }
  }
  return values
}

export function serviceConfigFields(serviceType?: string | null): ConfigFieldDef[] {
  const key = (serviceType || '').trim().toLowerCase()
  return SERVICE_CONFIG_FIELDS[key] || []
}
