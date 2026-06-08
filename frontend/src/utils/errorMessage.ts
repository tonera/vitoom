import { i18n } from '../i18n'

export type MessageParams = Record<string, unknown>

export type TranslatablePayload = {
  message_code?: string
  message_params?: MessageParams
  message?: string
  msg?: string
  error?: string
  code?: number
}

function translateKey(key: string, params?: MessageParams): string | null {
  const translated = i18n.global.t(key, params || {})
  return translated !== key ? translated : null
}

export function resolveMessageCode(
  messageCode?: string | null,
  params?: MessageParams
): string | null {
  const code = String(messageCode || '').trim()
  if (!code) return null
  return translateKey(`errors.${code}`, params)
}

export function resolveErrorCode(code?: number | null, params?: MessageParams): string | null {
  if (code == null || Number.isNaN(Number(code))) return null
  return translateKey(`errors.codes.${String(code)}`, params)
}

export function resolveTranslatableMessage(payload: TranslatablePayload): string {
  const params = payload.message_params || {}

  const fromMessageCode = resolveMessageCode(payload.message_code, params)
  if (fromMessageCode) return fromMessageCode

  const fromErrorCode = resolveErrorCode(payload.code, params)
  if (fromErrorCode) return fromErrorCode

  const fallback =
    payload.msg ||
    payload.message ||
    payload.error ||
    ''

  if (fallback) return String(fallback)

  return i18n.global.t('common.unknownError')
}

export function resolveWsMessage(message: TranslatablePayload): string {
  return resolveTranslatableMessage(message)
}
