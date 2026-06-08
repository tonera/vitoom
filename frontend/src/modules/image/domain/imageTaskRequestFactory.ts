import type { TaskCreateRequest } from '../../../utils/taskRunner'

export type Upscale = 1 | 2 | 4
export type ImageJobType = 'MK' | 'ED' | 'ID' | 'POSE' | 'FS'

export type LoraPayloadItem = { name: string; weight: number; trigger_word?: string }

export type BaseImageRequest = TaskCreateRequest & {
  task_type: 'image'
  job_type: ImageJobType

  prompt: string
  negative_prompt: string
  width: number
  height: number
  generate_num: number

  model_key?: string
  load_name?: string
  family?: string

  fast_mode: boolean
  guidance_scale: number
  num_inference_steps: number
  schedulerName: string
  seed: number
  remove_bg: boolean
  face_enhance: boolean
  upscale: Upscale

  // Optional extension fields used by backend for some job types
  loras?: string
}

export type MkImageRequest = BaseImageRequest & {
  job_type: 'MK'
}

export type EdImageRequest = BaseImageRequest & {
  job_type: 'ED'
  tpl_list: string[]
}

export type IdImageRequest = BaseImageRequest & {
  job_type: 'ID'
  url: string
}

export type PoseImageRequest = BaseImageRequest & {
  job_type: 'POSE'
  url: string
  image_file2: string
  edit_act: string
}

export type FsImageRequest = BaseImageRequest & {
  job_type: 'FS'
  url: string
  tpl_list: string[]
}

export type AnyImageTaskRequest = MkImageRequest | EdImageRequest | IdImageRequest | PoseImageRequest | FsImageRequest

function clampNum(v: number, min: number, max: number) {
  if (!Number.isFinite(v)) return min
  return Math.min(max, Math.max(min, v))
}

function asUpscale(v: any): Upscale {
  const n = Number(v)
  if (n === 1 || n === 2 || n === 4) return n
  return 1
}

export function serializeLoras(loras: LoraPayloadItem[] | undefined | null): string | undefined {
  if (!Array.isArray(loras) || loras.length === 0) return undefined
  // Keep stable payload shape: only include known fields
  const out = loras
    .map((x) => ({
      name: String(x?.name || '').trim(),
      weight: Number(x?.weight),
      trigger_word: x?.trigger_word !== undefined ? String(x.trigger_word || '').trim() : undefined,
    }))
    .filter((x) => x.name && Number.isFinite(x.weight))
    .map((x) => ({
      name: x.name,
      weight: Math.round(clampNum(x.weight, -1, 2) * 10) / 10,
      ...(x.trigger_word ? { trigger_word: x.trigger_word } : {}),
    }))

  if (!out.length) return undefined
  return JSON.stringify(out)
}

export type CommonImageRequestInput = {
  prompt: string
  negativePrompt?: string
  width: number
  height: number
  generateNum: number

  modelKey?: string
  loadName?: string
  family?: string

  guidanceScale: number
  numInferenceSteps: number
  schedulerName: string
  seed: number
  removeBg: boolean
  faceEnhance: boolean
  fastMode?: boolean
  upscale: Upscale | number

  lorasPayload?: LoraPayloadItem[]
}

export function buildMkImageRequest(input: CommonImageRequestInput): MkImageRequest {
  const loras = serializeLoras(input.lorasPayload)
  const rawSeed = Math.floor(Number(input.seed))
  const seed = Number.isFinite(rawSeed) ? Math.max(-1, rawSeed) : -1
  const req: MkImageRequest = {
    task_type: 'image',
    job_type: 'MK',
    prompt: String(input.prompt || ''),
    negative_prompt: String(input.negativePrompt || ''),
    width: Math.floor(clampNum(Number(input.width), 64, 4096)),
    height: Math.floor(clampNum(Number(input.height), 64, 4096)),
    generate_num: Math.floor(clampNum(Number(input.generateNum), 1, 9)),
    model_key: input.modelKey ? String(input.modelKey) : undefined,
    load_name: input.loadName ? String(input.loadName) : undefined,
    family: input.family ? String(input.family) : undefined,
    fast_mode: input.fastMode !== false,
    guidance_scale: Number.isFinite(Number(input.guidanceScale)) ? Number(input.guidanceScale) : 7.5,
    num_inference_steps: Math.floor(clampNum(Number(input.numInferenceSteps), 1, 100)),
    schedulerName: String(input.schedulerName || ''),
    // 约定：-1 表示随机 seed；>=0 表示固定 seed
    seed,
    remove_bg: Boolean(input.removeBg),
    face_enhance: Boolean(input.faceEnhance),
    upscale: asUpscale(input.upscale),
    ...(loras ? { loras } : {}),
  }
  return req
}

export function buildEdImageRequest(input: CommonImageRequestInput & { tplList: string[] }): EdImageRequest {
  const loras = serializeLoras(input.lorasPayload)
  const req: EdImageRequest = {
    ...buildMkImageRequest(input),
    job_type: 'ED',
    tpl_list: Array.isArray(input.tplList) ? input.tplList.filter(Boolean).map((x) => String(x)) : [],
    ...(loras ? { loras } : {}),
  }
  return req
}

export function buildIdImageRequest(input: CommonImageRequestInput & { url: string }): IdImageRequest {
  const req: IdImageRequest = {
    ...buildMkImageRequest(input),
    job_type: 'ID',
    url: String(input.url || ''),
  }
  return req
}

export function buildPoseImageRequest(
  input: CommonImageRequestInput & { url: string; image_file2: string; edit_act: string }
): PoseImageRequest {
  const loras = serializeLoras(input.lorasPayload)
  const req: PoseImageRequest = {
    ...buildMkImageRequest(input),
    job_type: 'POSE',
    url: String(input.url || ''),
    image_file2: String(input.image_file2 || ''),
    edit_act: String(input.edit_act || ''),
    ...(loras ? { loras } : {}),
  }
  return req
}

export function buildFsImageRequest(input: CommonImageRequestInput & { url: string; tplList: string[] }): FsImageRequest {
  const req: FsImageRequest = {
    ...buildMkImageRequest(input),
    job_type: 'FS',
    url: String(input.url || ''),
    tpl_list: Array.isArray(input.tplList) ? input.tplList.filter(Boolean).map((x) => String(x)) : [],
  }
  return req
}
