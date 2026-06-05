/**
 * HTTP客户端封装
 * 提供统一的API调用接口，包括请求拦截、响应处理、错误处理等
 */
import axios, { type AxiosInstance, type AxiosRequestConfig, type AxiosResponse, type AxiosError } from 'axios'
import { clearAuthTokens, getAccessToken, getRefreshToken, setAccessToken } from './auth'
import { getApiBaseURL } from './runtimeConfig'
import { getCurrentLocale, i18n } from '../i18n'
import { resolveTranslatableMessage } from './errorMessage'

// API基础URL：
// - 同源部署/反代："/api"
// - 前后端分离（运行时配置）："http(s)://host:port/api"
// 注意：main.ts 会先 loadRuntimeConfig() 再加载应用，确保这里读取到运行时配置。
const API_BASE_URL = getApiBaseURL() || import.meta.env.VITE_API_BASE_URL || '/api'

// 创建Axios实例
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    config.headers = config.headers || {}
    config.headers['Accept-Language'] = getCurrentLocale()

    // 从localStorage获取token并添加到请求头
    const token = getAccessToken()
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error: AxiosError) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response
  },
  async (error: AxiosError) => {
    const originalRequest = error.config as AxiosRequestConfig & { _retry?: boolean }

    // 401错误：token过期或无效，尝试刷新token
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true

      try {
        const refreshToken = getRefreshToken()
        if (refreshToken) {
          const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
            refresh_token: refreshToken,
          })

          const accessToken = response.data?.data?.access_token ?? response.data?.access_token
          if (!accessToken) {
            throw new Error('Refresh response missing access token')
          }
          setAccessToken(accessToken)
          originalRequest.headers = originalRequest.headers || {}
          originalRequest.headers.Authorization = `Bearer ${accessToken}`

          return apiClient(originalRequest)
        }
        clearAuthTokens()
        window.location.href = '/login'
      } catch (refreshError) {
        // 刷新token失败，清除本地token并跳转到登录页
        clearAuthTokens()
        window.location.href = '/login'
        return Promise.reject(refreshError)
      }
    }

    // 403：权限变更后清缓存；若当前在管理员页面则回首页
    if (error.response?.status === 403) {
      try {
        const { clearCurrentUserCache } = await import('./currentUser')
        clearCurrentUserCache()
        const path = window.location.pathname.replace(/\/+$/, '') || '/'
        if (path === '/users' || path === '/inference' || path === '/models' || path === '/settings') {
          window.location.replace('/')
        }
      } catch {
        // ignore cache clear failures
      }
    }

    return Promise.reject(error)
  }
)

// API响应类型
export interface ApiResponse<T = any> {
  code?: number
  msg?: string
  message?: string
  message_code?: string
  message_params?: Record<string, unknown>
  data?: T
  meta?: any
  [key: string]: any
}

// API错误类型
export interface ApiError {
  code: number
  message: string
  message_code?: string
  message_params?: Record<string, unknown>
  details?: any
}

/**
 * 处理API错误
 */
export function handleApiError(error: any): ApiError {
  if (error.response) {
    const response = error.response.data || {}
    const message = resolveTranslatableMessage({
      code: response.code ?? error.response.status,
      message_code: response.message_code,
      message_params: response.message_params,
      msg: response.msg,
      message: response.message || response.detail,
    })

    return {
      code: response.code || error.response.status,
      message,
      message_code: response.message_code,
      message_params: response.message_params,
      details: response.details || response.data,
    }
  }

  if (error.request) {
    return {
      code: -1,
      message: i18n.global.t('common.networkError'),
    }
  }

  return {
    code: -1,
    message: error.message || i18n.global.t('common.unknownError'),
  }
}

/**
 * GET请求
 */
export async function get<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
  const response = await apiClient.get<ApiResponse<T>>(url, config)
  return (response.data.data ?? response.data) as T
}

/**
 * GET请求（返回完整响应：包含 meta 等）
 */
export async function getRaw<T = any>(url: string, config?: AxiosRequestConfig): Promise<ApiResponse<T>> {
  const response = await apiClient.get<ApiResponse<T>>(url, config)
  return response.data as ApiResponse<T>
}

/**
 * POST请求
 */
export async function post<T = any>(
  url: string,
  data?: any,
  config?: AxiosRequestConfig
): Promise<T> {
  const response = await apiClient.post<ApiResponse<T>>(url, data, config)
  return (response.data.data ?? response.data) as T
}

/**
 * POST请求（返回完整响应：包含 msg 等）
 */
export async function postRaw<T = any>(
  url: string,
  data?: any,
  config?: AxiosRequestConfig
): Promise<ApiResponse<T>> {
  const response = await apiClient.post<ApiResponse<T>>(url, data, config)
  return response.data as ApiResponse<T>
}

/**
 * PUT请求
 */
export async function put<T = any>(
  url: string,
  data?: any,
  config?: AxiosRequestConfig
): Promise<T> {
  const response = await apiClient.put<ApiResponse<T>>(url, data, config)
  return (response.data.data ?? response.data) as T
}

/**
 * DELETE请求
 */
export async function del<T = any>(url: string, config?: AxiosRequestConfig): Promise<T> {
  const response = await apiClient.delete<ApiResponse<T>>(url, config)
  return (response.data.data ?? response.data) as T
}

export default apiClient

