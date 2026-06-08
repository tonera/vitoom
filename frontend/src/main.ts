import './style.css'
import { loadRuntimeConfig } from './utils/runtimeConfig'

const THEME_STORAGE_KEY = 'vitoom.theme'

const applyInitialTheme = () => {
  const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY)
  const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)').matches
  const shouldUseDark = storedTheme === 'dark' || (!storedTheme && prefersDark)
  document.documentElement.classList.toggle('dark', shouldUseDark)
}

async function bootstrap() {
  applyInitialTheme()

  // 运行时配置：支持“build 后用户只改 dist/config.json”
  await loadRuntimeConfig()

  const { createApp } = await import('vue')
  const { default: App } = await import('./App.vue')
  const { default: router } = await import('./router')
  const { i18n, initI18n } = await import('./i18n')

  await initI18n()

  const app = createApp(App)
  app.use(router)
  app.use(i18n)
  app.mount('#app')
}

bootstrap()
