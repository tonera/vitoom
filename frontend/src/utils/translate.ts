import { i18n } from '../i18n'

export type TranslateParams = Record<string, unknown>

/** 在非 Vue 组件上下文（composable / util）中使用 i18n */
export function translate(key: string, params?: TranslateParams): string {
  return i18n.global.t(key, params || {})
}

export function currentLocaleTag(): string {
  return String(i18n.global.locale.value || 'en-US')
}
