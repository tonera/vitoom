export type SupportedLocale = 'zh-CN' | 'en-US' | 'ja-JP'

export const LOCALE_STORAGE_KEY = 'vitoom.locale'
export const DEFAULT_LOCALE: SupportedLocale = 'en-US'
export const FALLBACK_LOCALE: SupportedLocale = 'en-US'

export function normalizeLocale(value: string | null | undefined): SupportedLocale {
  if (!value) return DEFAULT_LOCALE
  const raw = value.trim().toLowerCase()
  if (raw.startsWith('zh')) return 'zh-CN'
  if (raw.startsWith('en')) return 'en-US'
  if (raw.startsWith('ja')) return 'ja-JP'
  return DEFAULT_LOCALE
}

export function detectInitialLocale(): SupportedLocale {
  try {
    const stored = window.localStorage.getItem(LOCALE_STORAGE_KEY)
    if (stored) return normalizeLocale(stored)
  } catch {
    // ignore storage errors
  }

  const browserLocales = window.navigator.languages?.length
    ? window.navigator.languages
    : [window.navigator.language]

  for (const lang of browserLocales) {
    const normalized = normalizeLocale(lang)
    if (normalized === 'zh-CN' || normalized === 'en-US' || normalized === 'ja-JP') {
      return normalized
    }
  }

  return DEFAULT_LOCALE
}

export function applyDocumentLocale(locale: SupportedLocale) {
  document.documentElement.lang = locale
}
