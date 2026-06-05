export type ModelConfigDefaultsApplier = {
  setGuidanceScale?: (v: number) => void
  setNumInferenceSteps?: (v: number) => void
  setSchedulerName?: (v: string) => void
}

function clampNum(v: number, min: number, max: number) {
  if (!Number.isFinite(v)) return min
  return Math.min(max, Math.max(min, v))
}

function safeParseModelConfig(input: any): Record<string, any> | null {
  if (!input) return null
  if (typeof input === 'object') return input as Record<string, any>
  if (typeof input === 'string') {
    try {
      const v = JSON.parse(input)
      if (v && typeof v === 'object') return v as Record<string, any>
      return null
    } catch {
      return null
    }
  }
  return null
}

/**
 * 从模型的 model_config 里“按字段存在与否”覆盖 UI 的默认表单值。
 * 约定字段：
 * - guidance_scale: number
 * - num_inference_steps: number
 * - schedulerName: string | null
 */
export function applyModelConfigDefaults(input: any, applier: ModelConfigDefaultsApplier) {
  const cfg = safeParseModelConfig(input)
  if (!cfg) return

  if (Object.prototype.hasOwnProperty.call(cfg, 'guidance_scale')) {
    const v = Number((cfg as any).guidance_scale)
    if (Number.isFinite(v) && applier.setGuidanceScale) {
      applier.setGuidanceScale(Math.round(clampNum(v, 0, 20) * 10) / 10)
    }
  }

  if (Object.prototype.hasOwnProperty.call(cfg, 'num_inference_steps')) {
    const v = Number((cfg as any).num_inference_steps)
    if (Number.isFinite(v) && applier.setNumInferenceSteps) {
      applier.setNumInferenceSteps(Math.floor(clampNum(v, 1, 100)))
    }
  }

  if (Object.prototype.hasOwnProperty.call(cfg, 'schedulerName')) {
    const v = (cfg as any).schedulerName
    if (!applier.setSchedulerName) return
    if (typeof v === 'string') applier.setSchedulerName(v)
    else if (v === null) applier.setSchedulerName('')
  }
}

