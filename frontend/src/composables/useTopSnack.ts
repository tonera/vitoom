import { readonly, ref } from 'vue'

export type TopSnackVariant = 'default' | 'error' | 'success' | 'info'

type TopSnackState = {
  open: boolean
  message: string
  variant: TopSnackVariant
}

const state = ref<TopSnackState>({ open: false, message: '', variant: 'default' })
let timer: number | undefined

export type TopSnackOptions = {
  durationMs?: number
  variant?: TopSnackVariant
}

export function showTopSnack(message: string, opts?: TopSnackOptions) {
  const msg = String(message || '').trim()
  if (!msg) return

  const variant = opts?.variant ?? 'default'
  state.value = { open: true, message: msg, variant }

  if (timer) window.clearTimeout(timer)
  const durationMs = Number(opts?.durationMs)
  const defaultDur = variant === 'error' ? 5200 : 3500
  const dur = Number.isFinite(durationMs) ? Math.max(300, Math.floor(durationMs)) : defaultDur
  timer = window.setTimeout(() => {
    closeTopSnack()
  }, dur)
}

export function showTopSnackError(message: string, opts?: Omit<TopSnackOptions, 'variant'>) {
  showTopSnack(message, { ...opts, variant: 'error' })
}

export function showTopSnackSuccess(message: string, opts?: Omit<TopSnackOptions, 'variant'>) {
  showTopSnack(message, { ...opts, variant: 'success' })
}

export function closeTopSnack() {
  state.value = { ...state.value, open: false }
  if (timer) {
    window.clearTimeout(timer)
    timer = undefined
  }
}

/**
 * 只读状态给宿主组件渲染使用（单例）
 */
export function useTopSnack() {
  return {
    snackbar: readonly(state),
    closeTopSnack,
  }
}
