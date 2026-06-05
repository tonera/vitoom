/**
 * WebSocket客户端封装
 * 提供任务进度推送、任务状态更新等功能
 */
import { getAccessToken } from './auth'
import { getWsBasePath, getWsHost } from './runtimeConfig'
import { getCurrentLocale } from '../i18n'
import { resolveWsMessage } from './errorMessage'

export interface WebSocketMessage {
  task_id: string
  type?: string
  progress?: number
  message?: string
  message_code?: string
  message_params?: Record<string, unknown>
  status?: string
  timestamp?: string
  error?: string
  [key: string]: any
}

export type WebSocketMessageHandler = (message: WebSocketMessage) => void

export type WebSocketBinaryHandler = (data: ArrayBuffer) => void

export type WebSocketCloseInfo = { code: number; reason: string; wasClean: boolean }

export type WebSocketCloseHandler = (info: WebSocketCloseInfo) => void

function appendQuery(url: string, key: string, value: string): string {
  const separator = url.includes('?') ? '&' : '?'
  return `${url}${separator}${encodeURIComponent(key)}=${encodeURIComponent(value)}`
}

function appendLocale(url: string): string {
  return appendQuery(url, 'locale', getCurrentLocale())
}

export interface WebSocketClientOptions {
  /** 默认 true；聊天等场景应传 false 避免自动重连 */
  reconnect?: boolean
  maxReconnectAttempts?: number
}

export class WebSocketClient {
  private ws: WebSocket | null = null
  private url: string
  private reconnectInterval: number = 3000
  private maxReconnectAttempts: number = 5
  private reconnectEnabled: boolean = true
  private reconnectAttempts: number = 0
  /** ``binary`` 事件使用 ``WebSocketBinaryHandler``，其余为 ``WebSocketMessageHandler`` */
  private handlers: Map<string, Set<WebSocketMessageHandler | WebSocketBinaryHandler>> = new Map()
  private closeHandlers: Set<WebSocketCloseHandler> = new Set()
  private isManualClose: boolean = false
  private connectionSeq: number = 0

  constructor(baseUrl: string = '', options?: WebSocketClientOptions) {
    // WebSocket URL
    // - 同源部署/反代：默认走 window.location.host + "/ws"
    // - 前后端分离（运行时配置）：走 backendOrigin 的 host + "/ws"
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsHost = baseUrl || getWsHost() || window.location.host
    const wsBasePath = getWsBasePath()
    this.url = `${wsProtocol}//${wsHost}${wsBasePath}`
    if (options?.maxReconnectAttempts != null) {
      this.maxReconnectAttempts = Math.max(0, options.maxReconnectAttempts)
    }
    this.reconnectEnabled = options?.reconnect !== false
  }

  onClose(handler: WebSocketCloseHandler): () => void {
    this.closeHandlers.add(handler)
    return () => this.closeHandlers.delete(handler)
  }

  private emitClose(info: WebSocketCloseInfo): void {
    this.closeHandlers.forEach((h) => {
      try {
        h(info)
      } catch (e) {
        console.error('Error in WebSocket onClose handler:', e)
      }
    })
  }

  /**
   * 连接到WebSocket服务器
   */
  connect(taskId?: string, token?: string): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        // 新连接开始时重置手动关闭标记，确保异常断线可重连
        this.isManualClose = false
        let wsUrl = this.url
        if (taskId) {
          wsUrl = `${wsUrl}/task/${taskId}`
        }

        // 如果有token，添加到URL参数
        if (token) {
          wsUrl = appendQuery(wsUrl, 'token', token)
        }
        wsUrl = appendLocale(wsUrl)

        const connectionId = ++this.connectionSeq
        const ws = new WebSocket(wsUrl)
        let settled = false
        this.ws = ws
        ws.binaryType = 'arraybuffer'

        ws.onopen = () => {
          if (this.connectionSeq !== connectionId || this.isManualClose) {
            if (!settled) {
              settled = true
              resolve()
            }
            ws.close(1000, 'client disconnect')
            return
          }
          console.log('WebSocket connected')
          this.reconnectAttempts = 0
          settled = true
          resolve()
        }

        ws.onmessage = (event) => {
          if (this.connectionSeq !== connectionId || this.isManualClose) return
          if (typeof event.data === 'string') {
            try {
              const message: WebSocketMessage = JSON.parse(event.data)
              this.handleMessage(message)
            } catch (error) {
              console.error('Failed to parse WebSocket message:', error)
            }
            return
          }
          if (event.data instanceof ArrayBuffer) {
            this.emitBinary(event.data)
            return
          }
          if (event.data instanceof Blob) {
            void event.data.arrayBuffer().then((buf) => this.emitBinary(buf)).catch((err) => {
              console.error('Failed to read binary Blob from WebSocket:', err)
            })
          }
        }

        ws.onerror = () => {
          if (this.connectionSeq !== connectionId || this.isManualClose) {
            if (!settled) {
              settled = true
              resolve()
            }
            return
          }
          console.error('WebSocket error')
          settled = true
          reject(new Error('WebSocket connection failed'))
        }

        ws.onclose = (ev: CloseEvent) => {
          if (this.connectionSeq !== connectionId) {
            if (!settled) {
              settled = true
              resolve()
            }
            return
          }
          console.log('WebSocket closed')
          this.emitClose({ code: ev.code, reason: String(ev.reason || ''), wasClean: ev.wasClean })
          if (this.ws === ws) this.ws = null
          if (!settled && this.isManualClose) {
            settled = true
            resolve()
          }

          // 如果不是手动关闭，尝试重连
          if (
            this.reconnectEnabled &&
            !this.isManualClose &&
            this.reconnectAttempts < this.maxReconnectAttempts
          ) {
            this.reconnectAttempts++
            console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`)
            setTimeout(() => {
              this.connect(taskId, token).catch(console.error)
            }, this.reconnectInterval)
          }
        }
      } catch (error) {
        reject(error)
      }
    })
  }

  /**
   * 连接到指定路径（用于非 task 的 WS，比如 /model/{model_key}）
   */
  connectPath(path: string, token?: string): Promise<void> {
    const p = String(path || '').trim()
    if (!p) return Promise.reject(new Error('path is required'))
    // path should start with '/'
    const normalized = p.startsWith('/') ? p : `/${p}`

    return new Promise((resolve, reject) => {
      try {
        // 新连接开始时重置手动关闭标记，确保异常断线可重连
        this.isManualClose = false
        let wsUrl = `${this.url}${normalized}`
        if (token) {
          wsUrl = appendQuery(wsUrl, 'token', token)
        }
        wsUrl = appendLocale(wsUrl)

        const connectionId = ++this.connectionSeq
        const ws = new WebSocket(wsUrl)
        let settled = false
        this.ws = ws
        ws.binaryType = 'arraybuffer'

        ws.onopen = () => {
          if (this.connectionSeq !== connectionId || this.isManualClose) {
            if (!settled) {
              settled = true
              resolve()
            }
            ws.close(1000, 'client disconnect')
            return
          }
          console.log('WebSocket connected')
          this.reconnectAttempts = 0
          settled = true
          resolve()
        }

        ws.onmessage = (event) => {
          if (this.connectionSeq !== connectionId || this.isManualClose) return
          if (typeof event.data === 'string') {
            try {
              const message: WebSocketMessage = JSON.parse(event.data)
              this.handleMessage(message)
            } catch (error) {
              console.error('Failed to parse WebSocket message:', error)
            }
            return
          }
          if (event.data instanceof ArrayBuffer) {
            this.emitBinary(event.data)
            return
          }
          if (event.data instanceof Blob) {
            void event.data.arrayBuffer().then((buf) => this.emitBinary(buf)).catch((err) => {
              console.error('Failed to read binary Blob from WebSocket:', err)
            })
          }
        }

        ws.onerror = () => {
          if (this.connectionSeq !== connectionId || this.isManualClose) {
            if (!settled) {
              settled = true
              resolve()
            }
            return
          }
          console.error('WebSocket error')
          settled = true
          reject(new Error('WebSocket connection failed'))
        }

        ws.onclose = (ev: CloseEvent) => {
          if (this.connectionSeq !== connectionId) {
            if (!settled) {
              settled = true
              resolve()
            }
            return
          }
          console.log('WebSocket closed')
          this.emitClose({ code: ev.code, reason: String(ev.reason || ''), wasClean: ev.wasClean })
          if (this.ws === ws) this.ws = null
          if (!settled && this.isManualClose) {
            settled = true
            resolve()
          }

          if (
            this.reconnectEnabled &&
            !this.isManualClose &&
            this.reconnectAttempts < this.maxReconnectAttempts
          ) {
            this.reconnectAttempts++
            console.log(`Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`)
            setTimeout(() => {
              this.connectPath(normalized, token).catch(console.error)
            }, this.reconnectInterval)
          }
        }
      } catch (error) {
        reject(error)
      }
    })
  }

  /**
   * 断开WebSocket连接
   */
  disconnect(): void {
    this.isManualClose = true
    if (this.ws) {
      if (this.ws.readyState === WebSocket.CONNECTING) {
        // Calling close() while CONNECTING makes Chromium print a false
        // connection error. Let onopen/onerror settle the pending socket.
        return
      }
      this.ws.close()
      this.ws = null
    }
  }

  /**
   * 发送消息
   */
  send(data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data))
    } else {
      console.warn('WebSocket is not connected')
    }
  }

  /** 发送一帧二进制（与上一条 JSON 配对，如 ``audio_chunk`` 上行 PCM） */
  sendBinary(data: ArrayBuffer | ArrayBufferView): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket is not connected')
      return
    }
    const buf =
      data instanceof ArrayBuffer
        ? data
        : data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)
    this.ws.send(buf)
  }

  /**
   * 订阅消息
   */
  on(event: string, handler: WebSocketMessageHandler | WebSocketBinaryHandler): void {
    if (!this.handlers.has(event)) this.handlers.set(event, new Set())
    this.handlers.get(event)!.add(handler)
  }

  /**
   * 取消订阅
   */
  off(event: string, handler: WebSocketMessageHandler | WebSocketBinaryHandler): void {
    const handlers = this.handlers.get(event)
    if (handlers) {
      handlers.delete(handler)
    }
  }

  /**
   * 处理接收到的消息
   */
  private emitBinary(data: ArrayBuffer): void {
    const handlers = this.handlers.get('binary')
    handlers?.forEach((handler) => {
      try {
        ;(handler as unknown as WebSocketBinaryHandler)(data)
      } catch (error) {
        console.error('Error in WebSocket binary handler:', error)
      }
    })
  }

  private handleMessage(message: WebSocketMessage): void {
    // Always emit "message"
    const allHandlers = this.handlers.get('message')
    allHandlers?.forEach((handler) => {
      try {
        ;(handler as WebSocketMessageHandler)(message)
      } catch (error) {
        console.error('Error in WebSocket message handler:', error)
      }
    })

    // Also emit typed event if provided (e.g. "task_status", "result")
    if (message.type) {
      const typedHandlers = this.handlers.get(message.type)
      typedHandlers?.forEach((handler) => {
        try {
          ;(handler as WebSocketMessageHandler)(message)
        } catch (error) {
          console.error('Error in WebSocket message handler:', error)
        }
      })
    }
  }

  /**
   * 获取连接状态
   */
  get readyState(): number {
    return this.ws?.readyState ?? WebSocket.CLOSED
  }

  /**
   * 是否已连接
   */
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

// Factory (preferred): create a dedicated client per task/feature
export function createWebSocketClient(baseUrl: string = '', options?: WebSocketClientOptions): WebSocketClient {
  return new WebSocketClient(baseUrl, options)
}

// 便捷方法：连接任务进度WebSocket
export function connectTaskProgress(
  taskId: string,
  onProgress: (progress: number, message?: string) => void,
  onStatusChange?: (status: string) => void
): WebSocketClient {
  const token = getAccessToken()
  const wsClient = createWebSocketClient()

  wsClient.connect(taskId, token || undefined).catch((error) => {
    console.error('Failed to connect task progress WebSocket:', error)
  })

  wsClient.on('message', (message: WebSocketMessage) => {
    if (message.progress !== undefined) {
      const progressMessage = resolveWsMessage(message)
      onProgress(message.progress, progressMessage || message.message)
    }
    if (message.status && onStatusChange) {
      onStatusChange(message.status)
    }
  })

  return wsClient
}

export function connectModelDownload(
  modelKey: string,
  onMessage: (message: WebSocketMessage) => void
): WebSocketClient {
  const token = getAccessToken()
  const wsClient = createWebSocketClient()

  const key = encodeURIComponent(String(modelKey || '').trim())
  wsClient.connectPath(`/model/${key}`, token || undefined).catch((error) => {
    console.error('Failed to connect model download WebSocket:', error)
  })

  wsClient.on('message', onMessage)
  return wsClient
}

export default WebSocketClient

export { resolveWsMessage } from './errorMessage'

