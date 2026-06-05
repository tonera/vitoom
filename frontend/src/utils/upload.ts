/**
 * 文件上传和下载工具函数
 * 提供统一的上传和下载接口
 */
import axios from 'axios'
import { getAccessToken } from './auth'
import { getBackendRequestBaseURL } from './runtimeConfig'
import { translate } from './translate'

/**
 * 上传单个文件到后端
 * @param file 要上传的文件
 * @returns Promise<string> 返回上传后的文件 URL
 * @throws Error 上传失败时抛出错误
 */
export async function uploadFile(file: File): Promise<string> {
  const fd = new FormData()
  fd.append('file', file)
  const token = getAccessToken()
  const resp = await axios.post('/v1/uploads', fd, {
    baseURL: getBackendRequestBaseURL(),
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      'Content-Type': 'multipart/form-data',
    },
  })
  const url = resp?.data?.data?.url as string | undefined
  if (!url) {
    throw new Error(translate('components.upload.missingUrl'))
  }
  // 后端现在直接返回绝对 URL，前端无需再拼接
  return url
}

/**
 * 验证文件是否为图片类型
 * @param file 要验证的文件
 * @returns boolean 是否为图片
 */
export function isImageFile(file: File): boolean {
  return file.type.startsWith('image/')
}

/**
 * 验证文件是否为视频类型
 * @param file 要验证的文件
 * @returns boolean 是否为视频
 */
export function isVideoFile(file: File): boolean {
  return file.type.startsWith('video/')
}

/**
 * 下载文件选项
 */
export interface DownloadFileOptions {
  /** 文件 URL（必需） */
  url: string
  /** 下载文件名（可选，如果不提供会根据 URL 和类型自动生成） */
  filename?: string
  /** 媒体类型，用于生成文件扩展名（可选） */
  mediaType?: 'image' | 'video' | 'audio' | 'text'
  /** 标题，用于生成文件名（可选） */
  title?: string
}

/**
 * 从 URL 中提取文件扩展名
 * @param url 文件 URL
 * @returns 文件扩展名（不含点号），如果无法提取则返回空字符串
 */
function extractExtensionFromUrl(url: string): string {
  try {
    const urlObj = new URL(url, window.location.href)
    const pathname = urlObj.pathname
    const match = pathname.match(/\.([a-zA-Z0-9]+)(?:\?|$)/)
    return match ? match[1]!.toLowerCase() : ''
  } catch {
    // 如果不是有效的 URL，尝试从路径中提取
    const match = url.match(/\.([a-zA-Z0-9]+)(?:\?|$)/)
    return match ? match[1]!.toLowerCase() : ''
  }
}

/**
 * 生成下载文件名
 * @param options 下载选项
 * @returns 生成的文件名
 */
function generateFilename(options: DownloadFileOptions): string {
  // 如果提供了文件名，直接使用
  if (options.filename) {
    return options.filename
  }

  // 尝试从 URL 中提取扩展名
  let ext = extractExtensionFromUrl(options.url)
  
  // 如果没有扩展名，根据媒体类型设置默认扩展名
  if (!ext) {
    if (options.mediaType === 'video') {
      ext = 'mp4'
    } else if (options.mediaType === 'audio') {
      ext = 'wav'
    } else if (options.mediaType === 'text') {
      ext = 'txt'
    } else {
      ext = 'jpg'
    }
  }

  // 如果有标题，使用标题作为文件名（清理标题中的特殊字符）
  if (options.title) {
    const cleanTitle = options.title.replace(/[^\w\s-]/g, '').trim().replace(/\s+/g, '-')
    if (cleanTitle) {
      return `${cleanTitle}.${ext}`
    }
  }

  // 默认文件名：根据类型和时间戳生成
  const prefix = options.mediaType === 'video' ? 'video' : options.mediaType === 'audio' ? 'audio' : options.mediaType === 'text' ? 'text' : 'image'
  return `${prefix}-${Date.now()}.${ext}`
}

/**
 * 检查 URL 是否与当前页面同源
 * @param url 要检查的 URL
 * @returns boolean 是否同源
 */
function isSameOrigin(url: string): boolean {
  try {
    const urlObj = new URL(url, window.location.href)
    return urlObj.origin === window.location.origin
  } catch {
    // 如果 URL 解析失败，假设是相对路径，视为同源
    return true
  }
}

/**
 * 下载文件到本地
 * @param options 下载选项
 * @returns Promise<boolean> 是否成功触发下载
 */
export async function downloadFile(options: DownloadFileOptions): Promise<boolean> {
  const { url } = options
  
  // 验证 URL
  if (!url || typeof url !== 'string') {
    console.error('下载失败：无效的 URL')
    return false
  }

  try {
    // 生成文件名
    const filename = generateFilename(options)
    
    // 检查是否同源
    const sameOrigin = isSameOrigin(url)
    
    if (sameOrigin) {
      // 同源：直接使用 download 属性
      const a = document.createElement('a')
      a.href = url
      a.download = filename
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      return true
    } else {
      // 跨域：需要先 fetch 获取文件内容，然后创建 Blob URL
      const token = getAccessToken()
      const response = await fetch(url, {
        headers: {
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const blob = await response.blob()
      const blobUrl = URL.createObjectURL(blob)
      
      try {
        const a = document.createElement('a')
        a.href = blobUrl
        a.download = filename
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        return true
      } finally {
        // 清理 Blob URL，避免内存泄漏
        URL.revokeObjectURL(blobUrl)
      }
    }
  } catch (error) {
    console.error('下载失败:', error)
    return false
  }
}

