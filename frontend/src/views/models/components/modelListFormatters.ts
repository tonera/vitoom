import { i18n } from '../../../i18n'

export function storageValueToLabel(v: string) {
  const s = String(v || '').toLowerCase()
  if (s === 'local') return i18n.global.t('models.storageLocal')
  if (s === 'cloud') return i18n.global.t('models.storageCloud')
  return v
}

export function abbr(s: string) {
  const t = String(s || '').trim()
  if (!t) return 'M'
  const parts = t.split(/[\s_\-]+/).filter(Boolean)
  const head = parts.slice(0, 2).map((x) => x.charAt(0).toUpperCase())
  return head.join('') || t.slice(0, 2).toUpperCase()
}

export function formatTime(iso?: string | null) {
  const v = String(iso || '').trim()
  if (!v) return ''
  const d = new Date(v)
  if (Number.isNaN(d.getTime())) return v
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

function hasDownloadBlock(desc: any) {
  const s = String(desc || '')
  return s.includes('--- download ---') && s.includes('--- /download ---')
}

export function isNotDownloadCompleted(m: any) {
  const st = String(m?.download_status || '').trim().toLowerCase()
  const hasMeta = Boolean(String(m?.resource_id || '').trim()) || hasDownloadBlock(m?.description)
  if (!hasMeta) return false
  return st !== 'completed'
}

export function extractDownloadPercent(m: any): number | null {
  // 优先使用实时字段（来自 /ws/model 的 download_status），保证卡片进度条与终端一致
  const liveP = Number(m?.download_progress ?? m?.progress)
  if (Number.isFinite(liveP) && liveP >= 0 && liveP <= 100) {
    return Math.max(0, Math.min(100, liveP))
  }
  const bd = Number(m?.download_bytes_downloaded)
  const bt = Number(m?.download_bytes_total)
  if (Number.isFinite(bd) && Number.isFinite(bt) && bt > 0) {
    const p = Math.round((bd / bt) * 100)
    return Math.max(0, Math.min(100, p))
  }

  const desc = String(m?.description || '')
  const m1 = desc.match(/progress:\s*.*?\(\s*(\d{1,3})%\s*\)/i)
  if (m1?.[1]) {
    const p = Math.max(0, Math.min(100, Number(m1[1])))
    if (!Number.isNaN(p)) return p
  }
  // 兜底：任意 %（避免 progress_text 格式略有差异）
  const m2 = desc.match(/(\d{1,3})%/i)
  if (m2?.[1]) {
    const p = Math.max(0, Math.min(100, Number(m2[1])))
    if (!Number.isNaN(p)) return p
  }
  return null
}

export function downloadStatusLabel(m: any) {
  const st = String(m?.download_status || '').trim().toLowerCase() || 'pending'
  const p = extractDownloadPercent(m)
  if (p !== null) return `${st} ${p}%`
  return st
}

