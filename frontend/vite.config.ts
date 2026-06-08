import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

function readAppVersion(): string {
  try {
    return readFileSync(resolve(__dirname, '../VERSION'), 'utf-8').trim()
  } catch {
    return ''
  }
}

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const appVersion = readAppVersion()
  // 仅开发态使用：生产 build 后不会走 vite proxy
  const env = loadEnv(mode, process.cwd(), '')
  const devBackend = env.VITE_DEV_BACKEND || 'http://127.0.0.1:8888'
  const devBackendWs = env.VITE_DEV_BACKEND_WS || devBackend.replace(/^http/i, 'ws')

  return {
    plugins: [vue()],
    define: {
      'import.meta.env.VITE_APP_VERSION': JSON.stringify(appVersion),
    },
    /* markdown-it-katex 为 CJS + 依赖 katex；显式纳入预构建，避免 dev 下 504 Outdated Optimize Dep 导致动态路由整页挂掉 */
    optimizeDeps: {
      include: ['markdown-it-katex', 'katex', 'lowlight', 'hast-util-to-html'],
    },
    server: {
      port: 5173,
      proxy: {
        '^/v1/.*': {
          target: devBackend,
          changeOrigin: true,
          secure: false,
        },
        '^/api/.*': {
          target: devBackend,
          changeOrigin: true,
          secure: false,
        },
        '^/outputs/.*': {
          target: devBackend,
          changeOrigin: true,
          secure: false,
        },
        '^/ws/.*': {
          target: devBackendWs,
          ws: true,
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: 'dist',
      assetsDir: 'assets',
    },
  }
})
