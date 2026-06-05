export type VideoResolutionP = 480 | 720 | number

export type VideoSize = {
  width: number
  height: number
}

function clampInt(v: number, min: number, max: number) {
  const n = Math.floor(Number(v))
  if (!Number.isFinite(n)) return min
  return Math.min(max, Math.max(min, n))
}

function roundDownToMultiple(v: number, m: number) {
  const n = Math.floor(Number(v))
  if (!Number.isFinite(n) || m <= 1) return n
  return Math.floor(n / m) * m
}

/**
 * Parse "16:9" / "9:16" / "1:1" etc into [w, h].
 * Falls back to 1:1 on invalid input.
 */
export function parseAspectRatio(val: string): [number, number] {
  const s = String(val || '').trim()
  const m = s.match(/^(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)$/)
  if (!m) return [1, 1]
  const w = Number(m[1])
  const h = Number(m[2])
  if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return [1, 1]
  return [w, h]
}

/**
 * Compute common video width/height based on "480p/720p" and aspect ratio.
 *
 * Rule:
 * - Treat `p` as the SHORTER side of the frame (works for landscape & portrait).
 * - The other side is derived from aspect ratio.
 * - Round dimensions DOWN to multiples of 16 (common encoder-friendly sizes).
 *
 * Examples:
 * - 480p, 16:9 => 848x480
 * - 480p, 9:16 => 480x848
 * - 720p, 16:9 => 1280x720
 * - 720p, 4:3  => 960x720
 */
export function computeVideoSizeByResolutionAndAspect(
  resolutionP: VideoResolutionP,
  aspectRatio: string,
  options: { multiple?: number; min?: number; max?: number } = {}
): VideoSize {
  const multiple = typeof options.multiple === 'number' ? Math.max(2, Math.floor(options.multiple)) : 16
  const min = typeof options.min === 'number' ? Math.floor(options.min) : 64
  const max = typeof options.max === 'number' ? Math.floor(options.max) : 4096

  const p = clampInt(Number(resolutionP), 240, 2160) // reasonable default bounds
  const [aw, ah] = parseAspectRatio(aspectRatio)
  const ratio = aw / ah

  // Use p as the shorter side to naturally support portrait ratios (e.g. 9:16).
  const isLandscape = ratio >= 1
  const shortSide = clampInt(p, min, max)

  if (isLandscape) {
    const h = roundDownToMultiple(shortSide, multiple)
    const w = roundDownToMultiple(h * ratio, multiple)
    return {
      width: clampInt(w, min, max),
      height: clampInt(h, min, max),
    }
  } else {
    const w = roundDownToMultiple(shortSide, multiple)
    const h = roundDownToMultiple(w / ratio, multiple)
    return {
      width: clampInt(w, min, max),
      height: clampInt(h, min, max),
    }
  }
}

