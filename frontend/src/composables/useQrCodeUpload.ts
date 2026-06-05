import { nextTick, onBeforeUnmount, reactive, ref, type Ref } from 'vue'
import { useI18n } from 'vue-i18n'
import axios from 'axios'
import QRCode from 'qrcode'
import { getAccessToken } from '../utils/auth'
import { getBackendRequestBaseURL } from '../utils/runtimeConfig'
import { showTopSnack } from './useTopSnack'

export type QrUploadResult = {
  url: string
  file_name?: string
  mime_type?: string
  file_size?: number
  id?: string
}

type QrState = {
  open: boolean
  loading: boolean
  polling: boolean
  error: string
  token: string
  uploadUrl: string
  pollUrl: string
  expiresAt: string
  ttlSeconds: number
}

function parseUploadResult(data: Record<string, unknown> | undefined): QrUploadResult | null {
  if (!data) return null
  const url = String(data.url || data.http_url || '').trim()
  if (!url) return null
  return {
    url,
    file_name: data.file_name != null ? String(data.file_name) : undefined,
    mime_type: data.mime_type != null ? String(data.mime_type) : undefined,
    file_size: data.file_size != null ? Number(data.file_size) : undefined,
    id: data.id != null ? String(data.id) : undefined,
  }
}

export function useQrCodeUpload(options: {
  accept: () => string
  onUploaded?: (result: QrUploadResult) => void
}) {
  const { t } = useI18n()
  const qrCanvasRef = ref<HTMLCanvasElement | null>(null) as Ref<HTMLCanvasElement | null>
  const qr = reactive<QrState>({
    open: false,
    loading: false,
    polling: false,
    error: '',
    token: '',
    uploadUrl: '',
    pollUrl: '',
    expiresAt: '',
    ttlSeconds: 0,
  })
  let qrPollRunId = 0

  function resetQrFields() {
    qr.loading = false
    qr.polling = false
    qr.error = ''
    qr.token = ''
    qr.uploadUrl = ''
    qr.pollUrl = ''
    qr.expiresAt = ''
    qr.ttlSeconds = 0
  }

  function close() {
    qrPollRunId += 1
    resetQrFields()
    qr.open = false
  }

  async function drawQrCanvas() {
    await nextTick()
    if (!qrCanvasRef.value || !qr.uploadUrl) return
    await QRCode.toCanvas(qrCanvasRef.value, qr.uploadUrl, {
      width: 220,
      margin: 1,
    })
  }

  async function startQrPolling(runId: number) {
    if (!qr.pollUrl) return
    qr.polling = true
    const startedAt = Date.now()
    const timeoutMs = Math.max(30, Number(qr.ttlSeconds || 600)) * 1000
    const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms))

    while (qr.open && qr.polling && runId === qrPollRunId && Date.now() - startedAt < timeoutMs) {
      try {
        const token = getAccessToken()
        const resp = await axios.get(qr.pollUrl, {
          headers: {
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
        })
        const msg = String(resp?.data?.msg || '')
        const result = parseUploadResult(resp?.data?.data as Record<string, unknown> | undefined)
        if (resp?.data?.code === 1 && result) {
          options.onUploaded?.(result)
          close()
          return
        }
        if (msg && msg !== 'PENDING') {
          qr.error = msg
          qr.polling = false
          return
        }
      } catch (e: unknown) {
        const err = e as { response?: { data?: { detail?: string; msg?: string } }; message?: string }
        qr.error =
          err?.response?.data?.detail ||
          err?.response?.data?.msg ||
          err?.message ||
          t('components.upload.qrPollFailed')
        qr.polling = false
        return
      }
      await sleep(2000)
    }
    if (runId === qrPollRunId && qr.open) {
      qr.polling = false
      if (!qr.error) qr.error = t('components.upload.qrExpired')
    }
  }

  async function open() {
    qr.open = true
    qr.loading = true
    qr.error = ''
    qr.polling = false
    qrPollRunId += 1
    const runId = qrPollRunId
    try {
      const token = getAccessToken()
      const resp = await axios.post(
        '/v1/uploads/qrcode/init',
        { accept: options.accept() },
        {
          baseURL: getBackendRequestBaseURL(),
          headers: {
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
        },
      )
      const data = resp?.data?.data || {}
      qr.token = String(data.token || '')
      qr.uploadUrl = String(data.uploadUrl || data.upload_url || '')
      qr.pollUrl = String(data.pollUrl || data.poll_url || '')
      qr.expiresAt = String(data.expiresAt || data.expires_at || '')
      qr.ttlSeconds = Number(data.ttlSeconds || data.ttl_seconds || 600)
      if (!qr.uploadUrl || !qr.pollUrl) throw new Error(t('components.upload.qrInitFailed'))
      qr.loading = false
      await drawQrCanvas()
      void startQrPolling(runId)
    } catch (e: unknown) {
      qr.loading = false
      const err = e as { response?: { data?: { detail?: string; msg?: string } }; message?: string }
      qr.error =
        err?.response?.data?.detail ||
        err?.response?.data?.msg ||
        err?.message ||
        t('components.upload.qrInitFailed')
    }
  }

  async function copyQrLink() {
    if (!qr.uploadUrl) return
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(qr.uploadUrl)
        showTopSnack(t('components.upload.linkCopied'))
        return
      }
      const ta = document.createElement('textarea')
      ta.value = qr.uploadUrl
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      ta.style.left = '-9999px'
      document.body.appendChild(ta)
      ta.focus()
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
      showTopSnack(t('components.upload.linkCopied'))
    } catch {
      showTopSnack(t('components.upload.copyFailed'))
    }
  }

  onBeforeUnmount(() => {
    close()
  })

  return {
    qr,
    qrCanvasRef,
    open,
    close,
    copyQrLink,
    refresh: open,
  }
}
