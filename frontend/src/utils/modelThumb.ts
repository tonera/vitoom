import { getBackendOrigin, resolveBackendPublicUrl } from './runtimeConfig'

function isAbsolutePublicUrl(s: string): boolean {
  return /^(https?:)?\/\//i.test(s) || /^data:/i.test(s)
}

/** 转为 outputs 根目录下的相对 key，例如 models/xxx.webp（与后端入库格式一致） */
export function stripToOutputsRelativeKey(value: string): string {
  let u = String(value || '').trim()
  if (!u) return ''

  if (isAbsolutePublicUrl(u)) {
    try {
      const url = new URL(u)
      const origin = getBackendOrigin()
      if (origin) {
        try {
          if (url.origin === new URL(origin).origin) {
            u = url.pathname
          } else {
            return u
          }
        } catch {
          return u
        }
      } else {
        u = url.pathname
      }
    } catch {
      return u
    }
  }

  u = u.replace(/^\/+/, '')
  if (u.startsWith('resources/outputs/')) {
    return u.slice('resources/outputs/'.length)
  }
  if (u.startsWith('outputs/')) {
    return u.slice('outputs/'.length)
  }
  const match = u.match(/(?:^|\/)outputs\/(.+)$/)
  if (match?.[1]) return match[1]
  return u
}

/** 提交前归一化：相对 outputs 路径或外部 http(s) URL */
export function normalizeModelThumbForStore(input: string): string {
  const s = String(input || '').trim()
  if (!s) return ''

  if (isAbsolutePublicUrl(s)) {
    try {
      const url = new URL(s)
      const origin = getBackendOrigin()
      if (origin) {
        try {
          if (url.origin === new URL(origin).origin) {
            return stripToOutputsRelativeKey(url.pathname)
          }
        } catch {
          /* ignore */
        }
      }
      return s
    } catch {
      return s
    }
  }

  return stripToOutputsRelativeKey(s)
}

/** 编辑表单回填：优先展示相对 outputs 的短路径 */
export function formatModelThumbForForm(input: string): string {
  const s = String(input || '').trim()
  if (!s) return ''
  const key = stripToOutputsRelativeKey(s)
  return key || s
}

/** 解析为浏览器可加载的 URL（磁盘 resources/outputs → HTTP /outputs） */
export function resolveModelThumbUrl(input: string): string {
  const s = String(input || '').trim()
  if (!s) return ''
  if (isAbsolutePublicUrl(s)) return s

  if (s.startsWith('/')) {
    if (s.startsWith('/resources/outputs/')) {
      return resolveBackendPublicUrl(s.replace(/^\/resources\//, '/'))
    }
    return resolveBackendPublicUrl(s)
  }
  if (s.startsWith('resources/outputs/')) {
    return resolveBackendPublicUrl(`/${s.replace(/^resources\//, '')}`)
  }
  if (s.startsWith('outputs/')) {
    return resolveBackendPublicUrl(`/${s}`)
  }
  return resolveBackendPublicUrl(`/outputs/${s.replace(/^\/+/, '')}`)
}
