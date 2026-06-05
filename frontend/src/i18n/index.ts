import { createI18n } from 'vue-i18n'
import {
  applyDocumentLocale,
  detectInitialLocale,
  FALLBACK_LOCALE,
  LOCALE_STORAGE_KEY,
  type SupportedLocale,
} from './locale'

export {
  applyDocumentLocale,
  detectInitialLocale,
  FALLBACK_LOCALE,
  LOCALE_STORAGE_KEY,
  normalizeLocale,
  type SupportedLocale,
} from './locale'

const loaders: Record<SupportedLocale, () => Promise<{ default: Record<string, unknown> }>> = {
  'zh-CN': () => import('./messages/zh-CN'),
  'en-US': () => import('./messages/en-US'),
  'ja-JP': () => import('./messages/ja-JP'),
}

const loadedLocales = new Set<SupportedLocale>()

const initialLocale = detectInitialLocale()
applyDocumentLocale(initialLocale)

export const i18n = createI18n({
  legacy: false,
  locale: initialLocale,
  fallbackLocale: FALLBACK_LOCALE,
  messages: {},
})

export function getCurrentLocale(): SupportedLocale {
  return i18n.global.locale.value as SupportedLocale
}

export async function loadLocaleMessages(locale: SupportedLocale) {
  if (loadedLocales.has(locale)) return
  const messages = await loaders[locale]()
  i18n.global.setLocaleMessage(locale, messages.default)
  loadedLocales.add(locale)
}

export async function setLocale(locale: SupportedLocale) {
  if (locale === getCurrentLocale()) return
  await loadLocaleMessages(locale)
  i18n.global.locale.value = locale
  window.localStorage.setItem(LOCALE_STORAGE_KEY, locale)
  applyDocumentLocale(locale)
}

export async function initI18n() {
  await loadLocaleMessages(initialLocale)
}
