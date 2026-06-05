import { onBeforeUnmount, reactive, readonly, ref, shallowRef } from 'vue'
import { translate } from '../utils/translate'

/**
 * useAudioPlayback
 *
 * 消费后端 `audio_delta` 的音频分片（PCM16 LE 或 WAV），串联无缝播放。
 *
 * 方案：事件驱动短水位 jitter buffer
 *   - 状态机：
 *     · `scheduled`：已排进 AudioContext 时间线的 AudioBufferSourceNode FIFO
 *     · `pending`：原始解码数据 FIFO
 *   - `enqueue(chunk)`：同步解码成 Float32，入 `pending`；再把 pending 补入时间线，
 *     尽量维持短播放水位，避免首包/前几包 jitter 直接变成断续。
 *
 *   默认首播缓冲 220ms，正常目标水位 350ms；延迟仍足够低，但开头更不容易饿死。
 *
 * 注意：
 *   - AudioContext 需要用户手势激活；首次 `enqueue` 前若未激活，会调 `resume()`；
 *     若浏览器仍拒绝（无手势），解码将被丢弃并设 `error`，后续片同理。
 *   - chunk.sampleRate 直接用于 AudioBuffer，浏览器自动向 ctx.sampleRate 重采样。
 *   - `flush()` 等整条队列（pending + scheduled）自然播完（非阻塞 Promise）。
 *   - `cancel(reason)`：立即 stop scheduled，清空 pending（barge-in / teardown 语义）。
 *
 * 数字人音视频对齐模式（方案 A，docs/数字人口型音视频对齐接力文档.md）：
 *   - `setRemoteAudioActive(true)`：上层（AgentChat.vue）侦测到 sidecar 推
 *     remote audio track 后调用。本模块进入"音频权威源在 WebRTC 远端"模式：
 *     · `enqueue` 收到的 PCM 不再调度到 AudioContext（避免双倍出声）
 *     · 立即 `cancel('remote-audio-active')` 清掉已排队的本地播放
 *     · `echoGate` 直接放行所有 mic PCM（remote audio 时序不在本模块掌控,
 *        无法做精确 cross-correlation；上层应建议用户用耳机）
 *   - `setRemoteAudioActive(false)`：恢复默认行为，音频走本地解码出声
 *   - 切换瞬间会有 ~200ms 静音空隙（远端 jitter buffer 起播延迟），属预期
 */

export type PlaybackState = 'idle' | 'buffering' | 'playing' | 'error'

export interface AudioPlaybackChunk {
  /**
   * 裸音频字节。支持两种格式（由 ``mime`` 区分）：
   * - `mime: audio/pcm;rate=N`：裸 Int16 LE PCM
   * - `mime: audio/wav`：完整 WAV 文件（RIFF header + PCM16 / IEEE float32 样本）
   */
  bytes: ArrayBuffer
  /** 形如 `audio/pcm;rate=24000` 或 `audio/wav`；默认按 PCM 24000 解析 */
  mime: string
  /** 本 Turn/本段最后一片 */
  isFinal: boolean
  /** 可选 run_id，便于排障/barge-in 目标过滤 */
  runId?: string | null
  /** 可选，后端 `audio_delta.payload.sample_rate`，WAV 解析失败时兜底 */
  sampleRate?: number | null
}

export interface PlaybackStats {
  enqueued: number
  played: number
  cancelled: number
  droppedDecodeError: number
  echoSuppressed: number
  queueLength: number
  lastSampleRate: number
  lastChunkMs: number
}

export interface EchoGateDecision {
  suppress: boolean
  reason: 'echo' | 'quiet_during_playback' | 'not_playing' | 'no_reference' | 'voice_candidate'
  correlation: number
  micRms: number
  refRms: number
}

export interface UseAudioPlaybackOptions {
  /** 首片启动时相对 `now` 的缓冲（秒）；过小会爆音/断续，过大会延迟。默认 0.22 */
  latencyBudgetSec?: number
  /** 正常播放时尽量提前排进 AudioContext 的音频水位（秒）。默认 0.35 */
  targetBufferSec?: number
  /**
   * 两段拼接时的最小安全 gap（秒）：若上一段 endTime 已落后 `ctx.currentTime`，
   * 从 `now + gapSec` 开始播下一段；用于挂起/恢复后的回补，避免 `start()` 抛异常。
   * 默认 0.005。
   */
  joinGapSec?: number
}

function parseMimeRate(mime: string, fallback = 24000): number {
  const m = /rate\s*=\s*(\d+)/i.exec(mime || '')
  if (!m) return fallback
  const n = Number(m[1])
  return Number.isFinite(n) && n > 0 ? n : fallback
}

function pcm16ToFloat32(bytes: Uint8Array): Float32Array {
  // Int16 LE
  const samples = bytes.byteLength >>> 1
  if (samples === 0) return new Float32Array(0)
  // byteOffset 需 2 字节对齐，否则 Int16Array 会抛；兜底 copy 一遍
  const aligned =
    (bytes.byteOffset & 1) === 0
      ? new Int16Array(bytes.buffer, bytes.byteOffset, samples)
      : new Int16Array(bytes.slice().buffer)
  const out = new Float32Array(samples)
  for (let i = 0; i < samples; i++) out[i] = aligned[i]! / 32768
  return out
}

function isWavMime(mime: string): boolean {
  const s = (mime || '').toLowerCase()
  return s.startsWith('audio/wav') || s.startsWith('audio/x-wav') || s.startsWith('audio/wave')
}

/**
 * 解析 WAV（RIFF/WAVE）字节流为 mono Float32 + sampleRate。
 *
 * 支持：
 *   - PCM int16 / int24 (audioFormat=1)
 *   - IEEE float32 (audioFormat=3)
 *   - WAVE_FORMAT_EXTENSIBLE (0xFFFE，按 SubFormat 再分派)
 *
 * 多声道时取首声道（当前业务链路 TTS 都是 mono）。
 */
function parseWav(bytes: Uint8Array): { sampleRate: number; data: Float32Array } | null {
  if (bytes.byteLength < 44) return null
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  // 'RIFF' / 'WAVE' 用 big-endian 读便于字面匹配
  if (dv.getUint32(0, false) !== 0x52494646) return null
  if (dv.getUint32(8, false) !== 0x57415645) return null

  let offset = 12
  let audioFormat = 0
  let numChannels = 1
  let sampleRate = 0
  let bitsPerSample = 0
  let dataOffset = -1
  let dataSize = 0

  while (offset + 8 <= bytes.byteLength) {
    const chunkId = dv.getUint32(offset, false)
    const chunkSize = dv.getUint32(offset + 4, true)
    const dataStart = offset + 8

    if (chunkId === 0x666d7420) {
      audioFormat = dv.getUint16(dataStart, true)
      numChannels = Math.max(1, dv.getUint16(dataStart + 2, true))
      sampleRate = dv.getUint32(dataStart + 4, true)
      bitsPerSample = dv.getUint16(dataStart + 14, true)
      // WAVE_FORMAT_EXTENSIBLE: 真正的 format code 在 SubFormat GUID 前 2 字节
      if (audioFormat === 0xfffe && chunkSize >= 24) {
        audioFormat = dv.getUint16(dataStart + 24, true)
      }
    } else if (chunkId === 0x64617461) {
      dataOffset = dataStart
      dataSize = Math.min(chunkSize, bytes.byteLength - dataStart)
      break
    }
    offset = dataStart + chunkSize + (chunkSize & 1) // RIFF chunks 偶数字节对齐
  }

  if (dataOffset < 0 || sampleRate <= 0 || bitsPerSample <= 0) return null

  const bytesPerSample = bitsPerSample >>> 3
  const bytesPerFrame = bytesPerSample * numChannels
  if (bytesPerFrame <= 0) return null
  const frameCount = Math.floor(dataSize / bytesPerFrame)
  if (frameCount <= 0) return { sampleRate, data: new Float32Array(0) }

  const out = new Float32Array(frameCount)

  if (audioFormat === 1 && bitsPerSample === 16) {
    for (let i = 0; i < frameCount; i++) {
      out[i] = dv.getInt16(dataOffset + i * bytesPerFrame, true) / 32768
    }
  } else if (audioFormat === 3 && bitsPerSample === 32) {
    for (let i = 0; i < frameCount; i++) {
      out[i] = dv.getFloat32(dataOffset + i * bytesPerFrame, true)
    }
  } else if (audioFormat === 1 && bitsPerSample === 24) {
    for (let i = 0; i < frameCount; i++) {
      const o = dataOffset + i * bytesPerFrame
      const b0 = dv.getUint8(o)
      const b1 = dv.getUint8(o + 1)
      const b2 = dv.getUint8(o + 2)
      let v = b0 | (b1 << 8) | (b2 << 16)
      if (v & 0x800000) v |= 0xff000000 // sign extend
      out[i] = v / 8388608 // 2^23
    }
  } else if (audioFormat === 1 && bitsPerSample === 8) {
    // 8-bit WAV 是 unsigned
    for (let i = 0; i < frameCount; i++) {
      out[i] = (dv.getUint8(dataOffset + i * bytesPerFrame) - 128) / 128
    }
  } else {
    return null
  }

  return { sampleRate, data: out }
}

interface Decoded {
  float: Float32Array
  sampleRate: number
  durationSec: number
}

interface Scheduled {
  src: AudioBufferSourceNode
  startTime: number
  endTime: number
  durationSec: number
  float: Float32Array
  sampleRate: number
}

export function useAudioPlayback(options: UseAudioPlaybackOptions = {}) {
  const latencyBudgetSec = options.latencyBudgetSec ?? 0.15
  // 把所有 pending 一次性排满时间线（≈无穷大），关键：避免依赖 ``onended`` 回调
  // 触发后续段的 schedule。浏览器的 ``onended`` 实测有 50–200ms 滞后，如果限制
  // 水位（如旧值 0.25s）让 pump 在第一次 enqueue 时只 schedule 一段，后续段必须
  // 等 onended 才补——而那时 ``ctx.currentTime > lastEnd``，``startAt`` 会被推到
  // ``ctx.currentTime + joinGapSec``，每段切换都听到 50ms+ 静音 gap，听感是
  // "说一会，卡一下，再说一会，又卡一下"。
  // 服务端把整段切片 emit 后，时间线一次性排满，AudioContext 内部按 startAt 精确
  // 衔接相邻段；onended 何时来都不影响连续性。代价仅是 cancel 时多调几次 stop()。
  const targetBufferSec = options.targetBufferSec ?? 600
  const joinGapSec = options.joinGapSec ?? 0.005

  const state = ref<PlaybackState>('idle')
  const errorMessage = ref<string | null>(null)
  const stats = reactive<PlaybackStats>({
    enqueued: 0,
    played: 0,
    cancelled: 0,
    droppedDecodeError: 0,
    echoSuppressed: 0,
    queueLength: 0,
    lastSampleRate: 0,
    lastChunkMs: 0,
  })
  /**
   * 方案 A 对齐模式开关。`true` 时：
   * - `enqueue` 不调度 AudioContext（音频走 WebRTC remote track）
   * - `echoGate` 短路放行（无法对远端音频做 correlation）
   *
   * 详见模块顶部 docstring。
   */
  const remoteAudioActive = ref<boolean>(false)

  const audioContext = shallowRef<AudioContext | null>(null)
  /** 已排进 ctx 时间线的音频片段（FIFO） */
  const scheduled: Scheduled[] = []
  /** 解码完成但尚未入调度槽位的原始数据；严格 FIFO */
  const pending: Decoded[] = []
  /**
   * 已自然播完的最近 ref 数据，仅供 `echoGate` 做 tail-echo 抑制用。
   *
   * 物理回声链路（扬声器 → 空气 → 麦克风）有 100~300ms 延迟。如果 echoGate 只
   * 看 `scheduled`，TTS 最后一段在 `onended` 触发那一刻就被弹出 → echoGate 判
   * `not_playing` 整段放行，结果接下来 200~400ms 麦克风录到的"扬声器尾音回声"
   * 会原样上行进 ASR，被识别成新一轮用户输入，触发 LLM → TTS → 又被自己听到，
   * 进入自我对话死循环（issue: TTS 关了浏览器 AEC 后出现）。
   *
   * 所以播完的 sc 不立即丢，而是在这里再保留 `RECENT_REF_TAIL_MS`（覆盖 lag
   * 上限 500ms + frame 64ms + 安全余量），让 echoGate 在 TTS 自然结束后的几
   * 百毫秒里仍能做 correlation。用户真说话时和 ref 不相关，照常放行。
   *
   * 注意：这个值必须 ≥ echoGate lag 搜索上限（当前 500ms）+ frame 长度，否则
   * tail 期内 echoGate 查 ref 会查不到对应时刻的样本，correlation 退化到 0。
   */
  const recentRefPlayed: Scheduled[] = []
  const RECENT_REF_TAIL_MS = 900
  /** 'idle' → 'playing' 的切换定时器 */
  let firstStartTimer: ReturnType<typeof setTimeout> | null = null

  const recomputeQueueLength = () => {
    stats.queueLength = pending.length + scheduled.length
  }

  const ensureContext = async (): Promise<AudioContext> => {
    const existing = audioContext.value
    if (existing && existing.state !== 'closed') {
      if (existing.state === 'suspended') {
        try {
          await existing.resume()
        } catch (err) {
          console.warn('[useAudioPlayback] resume failed:', err)
        }
      }
      return existing
    }
    const Ctx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!Ctx) throw new Error(translate('agents.mic.noAudioContext'))
    const ctx = new Ctx()
    if (ctx.state === 'suspended') {
      try {
        await ctx.resume()
      } catch (err) {
        console.warn('[useAudioPlayback] initial resume failed:', err)
      }
    }
    audioContext.value = ctx
    return ctx
  }

  const scheduleFirstStartIndicator = (at: number) => {
    if (firstStartTimer) {
      clearTimeout(firstStartTimer)
      firstStartTimer = null
    }
    const ctx = audioContext.value
    if (!ctx) return
    const delayMs = Math.max(0, (at - ctx.currentTime) * 1000)
    firstStartTimer = setTimeout(() => {
      firstStartTimer = null
      if (scheduled.length > 0) state.value = 'playing'
    }, delayMs)
  }

  const decodeChunk = (chunk: AudioPlaybackChunk): Decoded | null => {
    let bytes: Uint8Array
    try {
      const raw = chunk.bytes
      if (!raw || raw.byteLength === 0) {
        bytes = new Uint8Array(0)
      } else {
        bytes = new Uint8Array(raw)
      }
    } catch (err) {
      stats.droppedDecodeError += 1
      console.warn('[useAudioPlayback] bytes decode error:', err)
      return null
    }
    if (bytes.byteLength < 2) return null

    let sampleRate: number
    let float: Float32Array
    if (isWavMime(chunk.mime)) {
      const parsed = parseWav(bytes)
      if (!parsed || parsed.data.length === 0) {
        stats.droppedDecodeError += 1
        console.warn(
          '[useAudioPlayback] WAV decode failed: mime=%s bytes=%d',
          chunk.mime,
          bytes.byteLength,
        )
        return null
      }
      sampleRate = parsed.sampleRate || chunk.sampleRate || parseMimeRate(chunk.mime, 24000)
      float = parsed.data
    } else {
      sampleRate = parseMimeRate(chunk.mime, chunk.sampleRate ?? 24000)
      float = pcm16ToFloat32(bytes)
      if (float.length === 0) return null
    }
    const durationSec = float.length / sampleRate
    stats.lastSampleRate = sampleRate
    stats.lastChunkMs = Math.round(durationSec * 1000)
    return { float, sampleRate, durationSec }
  }

  /** 把一个已解码片段调度到 ctx 时间线的 `startAt` 时刻。失败返回 null。 */
  const scheduleDecoded = (
    ctx: AudioContext,
    d: Decoded,
    startAt: number,
  ): Scheduled | null => {
    let buffer: AudioBuffer
    try {
      buffer = ctx.createBuffer(1, d.float.length, d.sampleRate)
      buffer.getChannelData(0).set(d.float)
    } catch (err) {
      stats.droppedDecodeError += 1
      console.warn('[useAudioPlayback] createBuffer failed:', err)
      return null
    }
    const src = ctx.createBufferSource()
    src.buffer = buffer
    src.connect(ctx.destination)

    const scheduled: Scheduled = {
      src,
      startTime: startAt,
      endTime: startAt + d.durationSec,
      durationSec: d.durationSec,
      float: d.float,
      sampleRate: d.sampleRate,
    }
    src.onended = () => {
      try {
        src.disconnect()
      } catch {
        /* ignore */
      }
      handleEnded(scheduled)
    }
    try {
      src.start(startAt)
    } catch (err) {
      stats.droppedDecodeError += 1
      try {
        src.disconnect()
      } catch {
        /* ignore */
      }
      console.warn('[useAudioPlayback] start failed:', err)
      return null
    }
    return scheduled
  }

  const scheduledEndTime = (): number | null => {
    const last = scheduled[scheduled.length - 1]
    return last ? last.endTime : null
  }

  /** 把 pending 填进时间线，尽量维持 targetBufferSec 的已排播放水位。 */
  const pumpSlots = (ctx: AudioContext): void => {
    while (pending.length > 0) {
      const lastEnd = scheduledEndTime()
      const scheduledAhead = lastEnd === null ? 0 : Math.max(0, lastEnd - ctx.currentTime)
      if (lastEnd !== null && scheduledAhead >= targetBufferSec) break

      const d = pending.shift()!
      if (lastEnd === null) {
        const startAt = ctx.currentTime + latencyBudgetSec
        const sc = scheduleDecoded(ctx, d, startAt)
        if (!sc) continue
        scheduled.push(sc)
        if (state.value === 'idle' || state.value === 'error') {
          state.value = 'buffering'
          errorMessage.value = null
          scheduleFirstStartIndicator(startAt)
        }
      } else {
        const startAt = Math.max(lastEnd, ctx.currentTime + joinGapSec)
        const sc = scheduleDecoded(ctx, d, startAt)
        if (!sc) continue
        scheduled.push(sc)
      }
    }
    recomputeQueueLength()
  }

  /** evict `recentRefPlayed` 里 endTime 早于 `now - RECENT_REF_TAIL_MS` 的条目。 */
  const trimRecentRefPlayed = (nowSec: number): void => {
    const cutoff = nowSec - RECENT_REF_TAIL_MS / 1000
    while (recentRefPlayed.length > 0 && recentRefPlayed[0]!.endTime < cutoff) {
      recentRefPlayed.shift()
    }
  }

  /** `src.onended` 回调；移除已播片段并继续补足播放水位。 */
  const handleEnded = (finished: Scheduled): void => {
    const idx = scheduled.indexOf(finished)
    if (idx >= 0) {
      scheduled.splice(idx, 1)
      stats.played += 1
      // 把 ref 数据搬到 tail 缓存，供 echoGate 在 TTS 刚结束的几百毫秒继续做
      // correlation；见 `recentRefPlayed` 的注释。
      recentRefPlayed.push(finished)
    } else {
      // finished 已不在队列里（例如 cancel 后残留的 stop 回调）：丢弃
      recomputeQueueLength()
      return
    }
    const ctx = audioContext.value
    if (ctx) {
      trimRecentRefPlayed(ctx.currentTime)
      pumpSlots(ctx)
    }
    if (scheduled.length === 0 && pending.length === 0) {
      state.value = 'idle'
    }
    recomputeQueueLength()
  }

  const enqueue = (chunk: AudioPlaybackChunk): void => {
    stats.enqueued += 1
    // 方案 A 对齐模式：音频权威源在 WebRTC remote track，本地不再解码出声。
    // 仍然 +1 enqueued 用于上层观测与一致性，但跳过 AudioContext 调度。
    // 不在这里做 echo ref 缓存：mock 一份"假装在 ctx 时间线上"的 ref 反而会
    // 让 `echoGate` 算出错的 correlation；echoGate 在对齐模式下走短路分支。
    if (remoteAudioActive.value) {
      return
    }
    const decoded = decodeChunk(chunk)
    if (!decoded) return
    pending.push(decoded)
    recomputeQueueLength()
    void (async () => {
      let ctx: AudioContext
      try {
        ctx = await ensureContext()
      } catch (err) {
        stats.droppedDecodeError += 1
        state.value = 'error'
        errorMessage.value = err instanceof Error ? err.message : translate('agents.mic.audioContextInitFailed')
        return
      }
      pumpSlots(ctx)
    })()
  }

  /**
   * 切换音频权威源。详见模块顶部 docstring 与
   * docs/数字人口型音视频对齐接力文档.md 第 4.2 节步骤 4。
   *
   * - `active=true`：方案 A 对齐模式启用。立即 cancel 已排队的本地播放,
   *   后续 `enqueue` no-op、`echoGate` 短路放行。
   * - `active=false`：恢复默认本地解码出声。
   *
   * 幂等：重复传同一值无副作用。
   */
  const setRemoteAudioActive = (active: boolean): void => {
    const next = !!active
    if (remoteAudioActive.value === next) return
    remoteAudioActive.value = next
    if (next) {
      // 切到远端音频前清掉本地已排队播放，避免本地+远端双倍出声
      cancel('remote-audio-active')
    }
  }

  const cancel = (reason?: string): void => {
    if (firstStartTimer) {
      clearTimeout(firstStartTimer)
      firstStartTimer = null
    }
    let dropped = 0
    for (const sc of scheduled) {
      try {
        sc.src.onended = null
        sc.src.stop()
      } catch {
        /* ignore */
      }
      try {
        sc.src.disconnect()
      } catch {
        /* ignore */
      }
      dropped += 1
    }
    dropped += pending.length
    scheduled.length = 0
    pending.length = 0
    // cancel 通常意味着 barge-in / teardown：用户已经在说话或会话不再需要 echo
    // 抑制；继续保留老的 tail ref 反而可能让随后用户的真实人声被误判为 TTS 余响
    // 而被吞。所以一并清掉。
    recentRefPlayed.length = 0
    stats.cancelled += dropped
    stats.queueLength = 0
    if (dropped > 0) {
      console.log('[useAudioPlayback] cancel:', reason || '(no reason)', 'dropped', dropped)
    }
    state.value = 'idle'
  }

  const sampleScheduledAt = (timeSec: number): number | null => {
    for (const sc of scheduled) {
      if (timeSec < sc.startTime || timeSec >= sc.endTime) continue
      const idx = Math.floor((timeSec - sc.startTime) * sc.sampleRate)
      if (idx < 0 || idx >= sc.float.length) return null
      return sc.float[idx] ?? 0
    }
    // 同样查最近已播完的 ref 缓存（覆盖 TTS 自然结束后 ~700ms 的物理回声尾巴）。
    for (const sc of recentRefPlayed) {
      if (timeSec < sc.startTime || timeSec >= sc.endTime) continue
      const idx = Math.floor((timeSec - sc.startTime) * sc.sampleRate)
      if (idx < 0 || idx >= sc.float.length) return null
      return sc.float[idx] ?? 0
    }
    return null
  }

  const echoGate = (pcm: Int16Array): EchoGateDecision => {
    // 方案 A 对齐模式：本地没有调度任何 TTS PCM 到 AudioContext（audio 走
    // WebRTC remote track），无法对远端音频做 cross-correlation。
    // 直接放行所有 mic PCM——上层根据业务策略决定是否给用户一个"建议带耳机"
    // 的 banner，避免外放回声进 ASR。
    if (remoteAudioActive.value) {
      return { suppress: false, reason: 'not_playing', correlation: 0, micRms: 0, refRms: 0 }
    }
    const ctx = audioContext.value
    if (ctx) trimRecentRefPlayed(ctx.currentTime)
    // tail ref 保留期内（最后一段 TTS 在 RECENT_REF_TAIL_MS 内自然结束）也要继续
    // 做 correlation —— 否则会把刚结束 TTS 的物理回声尾巴当成新一轮用户输入。
    const hasRef = scheduled.length > 0 || recentRefPlayed.length > 0
    if (!hasRef || state.value === 'error') {
      return { suppress: false, reason: 'not_playing', correlation: 0, micRms: 0, refRms: 0 }
    }
    if (!ctx || pcm.length === 0) {
      return { suppress: false, reason: 'no_reference', correlation: 0, micRms: 0, refRms: 0 }
    }

    let micEnergy = 0
    for (let i = 0; i < pcm.length; i++) {
      const v = pcm[i]! / 32768
      micEnergy += v * v
    }
    const micRms = Math.sqrt(micEnergy / pcm.length)

    // 关键：在 TTS 外放期间，**不能**用"micRms < 阈值"早退判 quiet 直接丢帧。
    // 浏览器 WebRTC AEC + AGC 在 double-talk（用户和远端同时讲）场景下会自适应
    // 地把麦克输入往下"压"——我们曾用过 0.006 的阈值，结果实测里 AEC 把人声
    // 衰减到 RMS < 0.006，echoGate 整帧丢，barge-in VAD 拼不出连续 200ms PCM、
    // FunASR 全判 silence，900ms 阈值永远到不了，导致"TTS 生成阶段说 2-3 秒
    // 都打不断"。把"是不是有效语音"留给后端 FunASR 决定。
    // 这里只对**真正零信号**（mic 整帧基本就是数字 0，比如 device 没数据）做最
    // 后兜底，避免空包浪费带宽——0.0005 远低于任何可见环境噪声地板。
    if (micRms < 0.0005) {
      stats.echoSuppressed += 1
      return { suppress: true, reason: 'quiet_during_playback', correlation: 0, micRms, refRms: 0 }
    }

    let bestCorrelation = 0
    let bestRefRms = 0
    const micSampleRate = 16000
    const frameDurationSec = pcm.length / micSampleRate
    // lag 搜索范围：覆盖"扬声器机械延迟 + 浏览器 outputLatency + 空气传播"全部典型值。
    //   - Mac 内置扬声器/麦克风：物理 + AudioContext outputLatency 实测 200~400ms
    //   - 蓝牙耳机/外置音箱混响重的场景偶尔到 450ms
    //   早期用 280ms 上限，结果"李白的字是"这种 1.5s 的 TTS 在播时，物理回声 lag
    //   落在 [300, 500]ms 区间的整段都被漏判，correlation 不到 0.52，echo PCM
    //   被原样上行进后端，触发 `energy_speech_floor` 强判 speech → barge-in 误触。
    //   500ms 是"同时覆盖 95% 设备 + 不让搜索成本爆炸"的折中（24 个 lag * 20ms 步）。
    for (let lagMs = 40; lagMs <= 500; lagMs += 20) {
      const refEnd = ctx.currentTime - lagMs / 1000
      const refStart = refEnd - frameDurationSec
      let dot = 0
      let refEnergy = 0
      let used = 0
      for (let i = 0; i < pcm.length; i++) {
        const refTime = refStart + (i / Math.max(1, pcm.length - 1)) * frameDurationSec
        const ref = sampleScheduledAt(refTime)
        if (ref == null) continue
        const mic = pcm[i]! / 32768
        dot += mic * ref
        refEnergy += ref * ref
        used += 1
      }
      if (used < pcm.length * 0.75 || refEnergy <= 1e-9) continue
      const corr = Math.abs(dot / Math.sqrt(micEnergy * refEnergy))
      if (corr > bestCorrelation) {
        bestCorrelation = corr
        bestRefRms = Math.sqrt(refEnergy / used)
      }
    }

    if (bestCorrelation >= 0.52) {
      stats.echoSuppressed += 1
      return {
        suppress: true,
        reason: 'echo',
        correlation: bestCorrelation,
        micRms,
        refRms: bestRefRms,
      }
    }
    return {
      suppress: false,
      reason: bestRefRms > 0 ? 'voice_candidate' : 'no_reference',
      correlation: bestCorrelation,
      micRms,
      refRms: bestRefRms,
    }
  }

  /** 等整条队列（pending + scheduled）自然播完。非阻塞 polling 实现。 */
  const flush = async (): Promise<void> => {
    if (scheduled.length === 0 && pending.length === 0) return
    await new Promise<void>((resolve) => {
      const check = () => {
        if (scheduled.length === 0 && pending.length === 0) {
          resolve()
          return
        }
        setTimeout(check, 50)
      }
      check()
    })
  }

  const teardown = (): void => {
    cancel('teardown')
    recentRefPlayed.length = 0
    const ctx = audioContext.value
    audioContext.value = null
    if (ctx && ctx.state !== 'closed') {
      ctx.close().catch(() => {
        /* ignore */
      })
    }
    state.value = 'idle'
    errorMessage.value = null
  }

  onBeforeUnmount(() => {
    teardown()
  })

  return {
    state,
    errorMessage,
    stats,
    enqueue,
    cancel,
    echoGate,
    flush,
    teardown,
    setRemoteAudioActive,
    /** 当前是否处于"音频权威源在 WebRTC remote 端"模式（方案 A） */
    remoteAudioActive: readonly(remoteAudioActive),
  }
}
