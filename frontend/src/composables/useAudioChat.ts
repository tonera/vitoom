import { onBeforeUnmount, reactive, ref, shallowRef } from 'vue'
import { translate } from '../utils/translate'

/**
 * useAudioChat
 *
 * 交付范围：
 *   - PR-1：AudioWorklet 采集 16kHz / mono / Int16 PCM，20ms/帧（保持不变）
 *   - PR-2（本次）：能量 VAD（hangover + pre-roll + min-speech + 段丢弃）
 *                  + onUplinkFrame / onVadEvent 订阅 + stats
 *
 * 后续 PR：
 *   - PR-3：audio_delta PCM 播放队列
 *   - PR-4：上层把 onUplinkFrame / onVadEvent 接到 useAgentChatSession 的 WS 帧
 *   - PR-5：barge-in
 *
 * VAD 行为：
 *   - 所有帧 → preRollRing（环形 buffer，长度 = preRollFrames），供段开始回溯
 *   - idle 状态：RMS ≥ thresholdOn 时进入 speaking，触发帧 + preRollRing 内容一起
 *     进 minSpeechBuf（暂不上行，等 min-duration 判定）
 *   - speaking/buffer 阶段：
 *     · RMS < thresholdOff 累计 silenceStreak；到 hangoverFrames 时结段
 *     · buffer 帧数 ≥ minSpeechFrames → emit 'speech-start' + 批量 flush minSpeechBuf
 *       （前 initialPreRollCount 条标 kind='pre-roll'，其余 'speech'），转直推流
 *   - 结段：
 *     · segmentConfirmed=true → emit 'speech-end'
 *     · segmentConfirmed=false → emit 'segment-dropped'，buffer 内帧永不上行
 *   - vadOptions.enabled=false：所有帧直接 emit('speech', segmentId='vad-off')
 */

export type MicState = 'idle' | 'requesting' | 'recording' | 'error'

export interface PcmFrame {
  pcm: Int16Array
  sampleRate: number
  samples: number
  frameMs: number
  seq: number
  captureTs: number
}

export type PcmFrameListener = (frame: PcmFrame) => void

export interface UplinkFrame extends PcmFrame {
  kind: 'pre-roll' | 'speech'
  segmentIndex: number
  segmentId: string
}

export type UplinkFrameListener = (frame: UplinkFrame) => void

export type VadEventType = 'speech-start' | 'speech-end' | 'segment-dropped'

export interface VadEvent {
  type: VadEventType
  at: number
  segmentId: string
  frameSeq: number
  durationMs?: number
  framesBuffered?: number
  reason?: 'silence-timeout' | 'stop-requested' | 'min-not-reached'
}

export type VadEventListener = (event: VadEvent) => void

export interface VadOptions {
  enabled: boolean
  /** 进入 speaking 的 RMS 阈值（linear，归一化 0~1） */
  thresholdOn: number
  /** 静音判定阈值（RMS 低于此计入 silenceStreak） */
  thresholdOff: number
  /** 连续静音多少毫秒判段结束 */
  hangoverMs: number
  /** 段开始时回溯多少毫秒历史帧 */
  preRollMs: number
  /** 段最短时长（帧数），不足丢弃整段 */
  minSpeechMs: number
}

export interface AudioStats {
  totalFrames: number
  uplinkFrames: number
  droppedFrames: number
  segments: number
  lastRmsNorm: number
}

export interface UseAudioChatOptions {
  targetSampleRate?: number
  frameMs?: number
  workletUrl?: string
  vad?: Partial<VadOptions>
}

const DEFAULT_VAD: VadOptions = {
  enabled: true,
  thresholdOn: 0.02,
  thresholdOff: 0.012,
  hangoverMs: 500,
  preRollMs: 300,
  minSpeechMs: 300,
}

function resolveWorkletUrl(override?: string): string {
  if (override) return override
  const base = (import.meta.env.BASE_URL || '/').replace(/\/+$/, '/') || '/'
  return `${base}audio-worklets/pcm-recorder.worklet.js`
}

function computeRmsNormalized(pcm: Int16Array): number {
  if (pcm.length === 0) return 0
  let sum = 0
  for (let i = 0; i < pcm.length; i++) {
    const s = pcm[i]! / 32768
    sum += s * s
  }
  return Math.sqrt(sum / pcm.length)
}

function newSegmentId(): string {
  return `seg-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

/** 时间顺序环形缓冲；仅持引用，不 copy PcmFrame.pcm（它本身已是独立 Int16Array） */
class RingBuffer<T> {
  readonly capacity: number
  private buf: Array<T | undefined>
  private head = 0
  private count = 0
  constructor(capacity: number) {
    this.capacity = capacity
    this.buf = new Array(Math.max(0, capacity))
  }
  push(item: T): void {
    if (this.capacity <= 0) return
    this.buf[this.head] = item
    this.head = (this.head + 1) % this.capacity
    if (this.count < this.capacity) this.count += 1
  }
  toArray(): T[] {
    if (this.count === 0) return []
    const out: T[] = new Array(this.count)
    const start = (this.head - this.count + this.capacity) % this.capacity
    for (let i = 0; i < this.count; i++) {
      out[i] = this.buf[(start + i) % this.capacity] as T
    }
    return out
  }
  clear(): void {
    this.buf.fill(undefined)
    this.head = 0
    this.count = 0
  }
}

type GetUserMediaCompat = (constraints: MediaStreamConstraints) => Promise<MediaStream>

/** 优先 mediaDevices.getUserMedia；极少数环境仅有旧版前缀回调 API */
function getCompatGetUserMedia(): GetUserMediaCompat | null {
  const md = navigator.mediaDevices
  if (typeof md?.getUserMedia === 'function') {
    return (constraints) => md.getUserMedia(constraints)
  }
  type LegacyCb = (
    constraints: MediaStreamConstraints,
    success: (stream: MediaStream) => void,
    error: (err: DOMException) => void,
  ) => void
  const nav = navigator as Navigator & {
    getUserMedia?: LegacyCb
    webkitGetUserMedia?: LegacyCb
    mozGetUserMedia?: LegacyCb
  }
  const legacy = nav.getUserMedia ?? nav.webkitGetUserMedia ?? nav.mozGetUserMedia
  if (typeof legacy !== 'function') return null
  return (constraints) =>
    new Promise((resolve, reject) => {
      legacy.call(navigator, constraints, resolve, reject)
    })
}

export function useAudioChat(options: UseAudioChatOptions = {}) {
  const targetSampleRate = options.targetSampleRate ?? 16000
  const frameMs = options.frameMs ?? 20
  const workletUrl = resolveWorkletUrl(options.workletUrl)

  const micState = ref<MicState>('idle')
  const errorMessage = ref<string | null>(null)
  const recordStartedAt = ref<number | null>(null)

  const audioContext = shallowRef<AudioContext | null>(null)
  const mediaStream = shallowRef<MediaStream | null>(null)
  const mediaTrackSettings = ref<MediaTrackSettings | null>(null)
  const sourceNode = shallowRef<MediaStreamAudioSourceNode | null>(null)
  const workletNode = shallowRef<AudioWorkletNode | null>(null)

  // ---------- listeners ----------
  const frameListeners = new Set<PcmFrameListener>()
  const uplinkListeners = new Set<UplinkFrameListener>()
  const vadListeners = new Set<VadEventListener>()
  let frameSeq = 0

  const onPcmFrame = (cb: PcmFrameListener): (() => void) => {
    frameListeners.add(cb)
    return () => void frameListeners.delete(cb)
  }
  const onUplinkFrame = (cb: UplinkFrameListener): (() => void) => {
    uplinkListeners.add(cb)
    return () => void uplinkListeners.delete(cb)
  }
  const onVadEvent = (cb: VadEventListener): (() => void) => {
    vadListeners.add(cb)
    return () => void vadListeners.delete(cb)
  }

  const emitPcm = (f: PcmFrame) => {
    for (const cb of frameListeners) {
      try {
        cb(f)
      } catch (err) {
        console.error('[useAudioChat] pcm listener error:', err)
      }
    }
  }
  const emitUplink = (f: UplinkFrame) => {
    stats.uplinkFrames += 1
    for (const cb of uplinkListeners) {
      try {
        cb(f)
      } catch (err) {
        console.error('[useAudioChat] uplink listener error:', err)
      }
    }
  }
  const emitVad = (e: VadEvent) => {
    for (const cb of vadListeners) {
      try {
        cb(e)
      } catch (err) {
        console.error('[useAudioChat] vad listener error:', err)
      }
    }
  }

  // ---------- VAD state ----------
  const vadOptions = ref<VadOptions>({ ...DEFAULT_VAD, ...(options.vad ?? {}) })
  const vadSpeaking = ref(false)
  const stats = reactive<AudioStats>({
    totalFrames: 0,
    uplinkFrames: 0,
    droppedFrames: 0,
    segments: 0,
    lastRmsNorm: 0,
  })

  const vadDerived = () => {
    const v = vadOptions.value
    const framesFromMs = (ms: number) => Math.max(0, Math.round(ms / frameMs))
    return {
      hangoverFrames: framesFromMs(v.hangoverMs),
      preRollFrames: framesFromMs(v.preRollMs),
      minSpeechFrames: framesFromMs(v.minSpeechMs),
    }
  }

  let preRollRing = new RingBuffer<PcmFrame>(vadDerived().preRollFrames)
  let minSpeechBuf: PcmFrame[] = []
  let silenceStreak = 0
  let segmentConfirmed = false
  let segmentId = ''
  let segmentStartSeq = -1
  let segmentIndex = 0
  let preRollEmitCount = 0
  let initialPreRollCount = 0

  const resetVadRuntime = () => {
    preRollRing = new RingBuffer<PcmFrame>(vadDerived().preRollFrames)
    minSpeechBuf = []
    silenceStreak = 0
    segmentConfirmed = false
    segmentId = ''
    segmentStartSeq = -1
    segmentIndex = 0
    preRollEmitCount = 0
    initialPreRollCount = 0
    vadSpeaking.value = false
  }

  const configureVad = (partial: Partial<VadOptions>): void => {
    vadOptions.value = { ...vadOptions.value, ...partial }
    if (micState.value !== 'recording') {
      resetVadRuntime()
    } else {
      const d = vadDerived()
      if (d.preRollFrames !== preRollRing.capacity) {
        preRollRing = new RingBuffer<PcmFrame>(d.preRollFrames)
      }
    }
  }

  const confirmAndFlushSegment = (preRollCount: number) => {
    segmentConfirmed = true
    vadSpeaking.value = true
    stats.segments += 1
    emitVad({
      type: 'speech-start',
      at: performance.now(),
      segmentId,
      frameSeq: segmentStartSeq,
    })
    for (let i = 0; i < minSpeechBuf.length; i++) {
      const f = minSpeechBuf[i]!
      const kind: UplinkFrame['kind'] = i < preRollCount ? 'pre-roll' : 'speech'
      if (kind === 'pre-roll') preRollEmitCount += 1
      emitUplink({ ...f, kind, segmentIndex: segmentIndex++, segmentId })
    }
    minSpeechBuf = []
  }

  const startSegment = (triggerFrame: PcmFrame) => {
    segmentId = newSegmentId()
    segmentStartSeq = triggerFrame.seq
    segmentIndex = 0
    segmentConfirmed = false
    minSpeechBuf = []
    preRollEmitCount = 0
    silenceStreak = 0

    const preRollFrames = preRollRing.toArray()
    for (const pf of preRollFrames) minSpeechBuf.push(pf)
    minSpeechBuf.push(triggerFrame)
    initialPreRollCount = preRollFrames.length

    const { minSpeechFrames } = vadDerived()
    if (minSpeechBuf.length >= minSpeechFrames) {
      confirmAndFlushSegment(initialPreRollCount)
    }
  }

  const closeSegment = (reason: VadEvent['reason'] = 'silence-timeout') => {
    if (!segmentId) return
    if (segmentConfirmed) {
      const speechFrames = Math.max(0, segmentIndex - preRollEmitCount)
      emitVad({
        type: 'speech-end',
        at: performance.now(),
        segmentId,
        frameSeq: frameSeq - 1,
        durationMs: speechFrames * frameMs,
        reason,
      })
    } else {
      stats.droppedFrames += minSpeechBuf.length
      emitVad({
        type: 'segment-dropped',
        at: performance.now(),
        segmentId,
        frameSeq: frameSeq - 1,
        framesBuffered: minSpeechBuf.length,
        reason: reason === 'stop-requested' ? 'stop-requested' : 'min-not-reached',
      })
    }
    minSpeechBuf = []
    silenceStreak = 0
    segmentConfirmed = false
    segmentId = ''
    segmentStartSeq = -1
    segmentIndex = 0
    preRollEmitCount = 0
    initialPreRollCount = 0
    vadSpeaking.value = false
  }

  const processFrame = (frame: PcmFrame) => {
    stats.totalFrames += 1
    const rms = computeRmsNormalized(frame.pcm)
    stats.lastRmsNorm = rms

    const v = vadOptions.value
    if (!v.enabled) {
      emitUplink({
        ...frame,
        kind: 'speech',
        segmentIndex: stats.uplinkFrames,
        segmentId: 'vad-off',
      })
      return
    }

    const { hangoverFrames, minSpeechFrames } = vadDerived()

    // idle：未启动段
    if (segmentId === '') {
      if (rms >= v.thresholdOn) {
        startSegment(frame)
      } else {
        preRollRing.push(frame)
      }
      return
    }

    // segmentConfirmed=false：仍在 min-duration buffer 阶段
    if (!segmentConfirmed) {
      minSpeechBuf.push(frame)
      if (rms < v.thresholdOff) silenceStreak += 1
      else silenceStreak = 0
      if (minSpeechBuf.length >= minSpeechFrames) {
        confirmAndFlushSegment(initialPreRollCount)
      } else if (silenceStreak >= hangoverFrames) {
        closeSegment('silence-timeout')
      }
      return
    }

    // live streaming 阶段
    if (rms < v.thresholdOff) silenceStreak += 1
    else silenceStreak = 0
    emitUplink({ ...frame, kind: 'speech', segmentIndex: segmentIndex++, segmentId })
    if (silenceStreak >= hangoverFrames) {
      closeSegment('silence-timeout')
    }
  }

  // ---------- core media plumbing ----------
  const isAudioWorkletSupported = (): boolean => {
    if (typeof window === 'undefined') return false
    if (!window.AudioContext && !(window as unknown as { webkitAudioContext?: unknown }).webkitAudioContext) {
      return false
    }
    return typeof (window as unknown as Record<string, unknown>).AudioWorkletNode === 'function'
  }

  const ensureAudioContext = async (): Promise<AudioContext> => {
    if (audioContext.value && audioContext.value.state !== 'closed') {
      if (audioContext.value.state === 'suspended') {
        await audioContext.value.resume()
      }
      return audioContext.value
    }
    const Ctx =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!Ctx) throw new Error(translate('agents.mic.noAudioContext'))
    const ctx = new Ctx()
    if (ctx.state === 'suspended') await ctx.resume()
    await ctx.audioWorklet.addModule(workletUrl)
    audioContext.value = ctx
    return ctx
  }

  const disposeNodes = () => {
    try {
      workletNode.value?.port?.close?.()
    } catch {
      /* ignore */
    }
    try {
      workletNode.value?.disconnect()
    } catch {
      /* ignore */
    }
    workletNode.value = null

    try {
      sourceNode.value?.disconnect()
    } catch {
      /* ignore */
    }
    sourceNode.value = null

    if (mediaStream.value) {
      for (const track of mediaStream.value.getTracks()) {
        try {
          track.stop()
        } catch {
          /* ignore */
        }
      }
      mediaStream.value = null
    }
    mediaTrackSettings.value = null
  }

  const resetStats = () => {
    stats.totalFrames = 0
    stats.uplinkFrames = 0
    stats.droppedFrames = 0
    stats.segments = 0
    stats.lastRmsNorm = 0
  }

  const startRecord = async (): Promise<void> => {
    if (micState.value === 'recording' || micState.value === 'requesting') return
    if (!isAudioWorkletSupported()) {
      micState.value = 'error'
      errorMessage.value = translate('agents.mic.noAudioWorklet')
      throw new Error(errorMessage.value)
    }
    const gum = getCompatGetUserMedia()
    if (!gum) {
      micState.value = 'error'
      errorMessage.value =
        typeof globalThis !== 'undefined' && globalThis.isSecureContext === false
          ? translate('agents.mic.insecureContext')
          : translate('agents.mic.noGetUserMedia')
      throw new Error(errorMessage.value)
    }

    micState.value = 'requesting'
    errorMessage.value = null
    frameSeq = 0
    resetStats()
    resetVadRuntime()

    let stream: MediaStream
    try {
      // 麦克风采集约束的关键取舍（与 barge-in 链路强相关，改前先看 docs/实时语音和文本聊天全生命周期流程.md §3.3）：
      // - echoCancellation: true —— **必须开**。Mac 内置扬声器/麦克风这种物理
      //   直耦合场景下，扬声器 → 空气 → 麦克风的链路有显著**非线性失真**（中等
      //   音量下扬声器谐波 3-5%、塑料外壳震动、室内混响叠加），mic 信号 ≠ 线性
      //   变换的 ref 信号。我们手搓的 `useAudioPlayback.echoGate` 是 sample-
      //   level 线性 correlation，对非线性失真无能为力，实测整段 echo
      //   correlation 只有 0.2-0.4 过不了 0.52 阈值，整段 echo 被原样上行进
      //   ASR，TTS 自己说的话被识别成新一轮用户输入触发自我对话死循环。
      //   WebRTC AEC3 的 NLP（non-linear processing）+ DTD（double-talk
      //   detector）专门处理这类场景，是行业标准，必须借助它做主防。
      //   早期版本曾用 false，是因为当时 AGC=true 联合 ducking 导致"打不断"，
      //   归因到 AEC 是误判 —— AGC 关了之后单独 AEC 不会把人声压到 0.0005。
      // - autoGainControl: false —— **保持关**。AGC 才是真正在 double-talk 时
      //   把麦克 ducking 30–40 dB 的元凶。AEC3 自己内部不做激进 AGC。
      // - noiseSuppression: true —— 保留。对环境底噪有用，目前没观察到副作用。
      //
      // 防御层次（任何一条要回退请同步更新文档 §3.3 并写回归测试）：
      //   主防 浏览器 AEC3            →处理 95%+ 物理回声（含非线性）
      //   备防一 useAudioPlayback.echoGate  → AEC 漏的残余线性 echo
      //   备防二 后端 streak=3        → 脉冲式 echo（连续 < 600ms 高能量不算）
      //   备防三 后端 floor=0.05      → FunASR silence + 高能量时强判 speech
      stream = await gum({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: false,
          sampleRate: targetSampleRate,
        },
        video: false,
      })
    } catch (err) {
      micState.value = 'error'
      const name = err instanceof Error ? err.name : ''
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        errorMessage.value = translate('agents.mic.permissionDenied')
      } else if (name === 'NotFoundError' || name === 'OverconstrainedError') {
        errorMessage.value = translate('agents.mic.noMicFound')
      } else {
        errorMessage.value = err instanceof Error ? err.message : translate('agents.mic.startFailed')
      }
      throw err
    }
    mediaStream.value = stream
    mediaTrackSettings.value = stream.getAudioTracks()[0]?.getSettings?.() ?? null

    let ctx: AudioContext
    try {
      ctx = await ensureAudioContext()
    } catch (err) {
      disposeNodes()
      micState.value = 'error'
      errorMessage.value = err instanceof Error ? err.message : translate('agents.mic.audioContextInitFailed')
      throw err
    }

    const src = ctx.createMediaStreamSource(stream)
    sourceNode.value = src

    const node = new AudioWorkletNode(ctx, 'pcm-recorder', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
    })
    node.port.postMessage({ type: 'config', targetSampleRate, frameMs })
    node.port.onmessage = (ev: MessageEvent) => {
      const data = ev.data as {
        type?: string
        buffer?: ArrayBuffer
        sampleRate?: number
        frameMs?: number
        samples?: number
      } | null
      if (!data || data.type !== 'pcm' || !(data.buffer instanceof ArrayBuffer)) return
      const frame: PcmFrame = {
        pcm: new Int16Array(data.buffer),
        sampleRate: data.sampleRate ?? targetSampleRate,
        frameMs: data.frameMs ?? frameMs,
        samples: data.samples ?? 0,
        seq: frameSeq++,
        captureTs: performance.now(),
      }
      emitPcm(frame)
      processFrame(frame)
    }
    workletNode.value = node
    src.connect(node)
    const silent = ctx.createGain()
    silent.gain.value = 0
    node.connect(silent).connect(ctx.destination)

    recordStartedAt.value = performance.now()
    micState.value = 'recording'
  }

  const stopRecord = (): void => {
    if (micState.value !== 'recording' && micState.value !== 'requesting') return
    try {
      workletNode.value?.port?.postMessage({ type: 'flush' })
    } catch {
      /* ignore */
    }
    if (segmentId) {
      closeSegment('stop-requested')
    }
    disposeNodes()
    recordStartedAt.value = null
    micState.value = 'idle'
  }

  const teardown = (): void => {
    if (segmentId) {
      closeSegment('stop-requested')
    }
    disposeNodes()
    const ctx = audioContext.value
    audioContext.value = null
    if (ctx && ctx.state !== 'closed') {
      ctx.close().catch(() => {
        /* ignore */
      })
    }
    frameListeners.clear()
    uplinkListeners.clear()
    vadListeners.clear()
    recordStartedAt.value = null
    micState.value = 'idle'
    errorMessage.value = null
    resetVadRuntime()
  }

  onBeforeUnmount(() => {
    teardown()
  })

  return {
    // PR-1 API
    micState,
    errorMessage,
    recordStartedAt,
    mediaTrackSettings,
    targetSampleRate,
    frameMs,
    startRecord,
    stopRecord,
    teardown,
    onPcmFrame,
    isAudioWorkletSupported,

    // PR-2 API
    vadOptions,
    vadSpeaking,
    stats,
    configureVad,
    onUplinkFrame,
    onVadEvent,
  }
}
