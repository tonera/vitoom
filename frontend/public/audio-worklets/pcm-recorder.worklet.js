/**
 * PCM Recorder AudioWorklet
 *
 * 把浏览器输入的 Float32 音频流重采样到 16 kHz 单声道 Int16 PCM，
 * 每 `frameMs` 毫秒成帧 (默认 20ms = 320 samples @16kHz)，通过
 * `port.postMessage` 把 `ArrayBuffer(Int16LE)` 发给主线程：
 *
 *   { type: 'pcm', buffer: ArrayBuffer, sampleRate: 16000, frameMs: 20, samples: 320 }
 *
 * 支持的控制消息（主线程 → worklet）：
 *   { type: 'config', targetSampleRate?: number, frameMs?: number }
 *   { type: 'flush' }   // 立即把尚未凑齐的尾巴发出去
 *
 * 重采样：线性插值，下/上采样通用。输入多通道时只取 channel 0。
 */

const DEFAULT_TARGET_RATE = 16000
const DEFAULT_FRAME_MS = 20

function floatToInt16(sample) {
  const clamped = sample < -1 ? -1 : sample > 1 ? 1 : sample
  return clamped < 0 ? Math.round(clamped * 0x8000) : Math.round(clamped * 0x7fff)
}

class PcmRecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this.targetSampleRate = DEFAULT_TARGET_RATE
    this.frameMs = DEFAULT_FRAME_MS
    this.frameSamples = Math.round((this.targetSampleRate * this.frameMs) / 1000)

    this.outBuffer = new Int16Array(this.frameSamples)
    this.outFill = 0

    // 线性插值状态：srcPos 是"下一个输出样点对应输入流中的浮点位置"，
    // 以"自本次 process() 输入数组 index 0"为原点。每次 process 结束时
    // 把 srcPos 减去 input.length 以继续指向未来流中的位置；prevSample
    // 保存上一次 process 的最后一个输入样点，用作跨块插值的左端点。
    this.srcPos = 0
    this.prevSample = 0
    this.hasPrev = false

    this.port.onmessage = (ev) => {
      const data = ev.data || {}
      if (data.type === 'config') {
        if (Number.isFinite(data.targetSampleRate) && data.targetSampleRate > 0) {
          this.targetSampleRate = data.targetSampleRate
        }
        if (Number.isFinite(data.frameMs) && data.frameMs > 0) {
          this.frameMs = data.frameMs
        }
        this.frameSamples = Math.round((this.targetSampleRate * this.frameMs) / 1000)
        this.outBuffer = new Int16Array(this.frameSamples)
        this.outFill = 0
        this.srcPos = 0
        this.hasPrev = false
      } else if (data.type === 'flush') {
        this._flush()
      }
    }
  }

  _flush() {
    if (this.outFill === 0) return
    const out = new Int16Array(this.outFill)
    out.set(this.outBuffer.subarray(0, this.outFill))
    this.port.postMessage(
      {
        type: 'pcm',
        buffer: out.buffer,
        sampleRate: this.targetSampleRate,
        frameMs: this.frameMs,
        samples: out.length,
      },
      [out.buffer],
    )
    this.outFill = 0
  }

  _pushSample(int16) {
    this.outBuffer[this.outFill++] = int16
    if (this.outFill >= this.frameSamples) {
      const out = new Int16Array(this.frameSamples)
      out.set(this.outBuffer)
      this.port.postMessage(
        {
          type: 'pcm',
          buffer: out.buffer,
          sampleRate: this.targetSampleRate,
          frameMs: this.frameMs,
          samples: this.frameSamples,
        },
        [out.buffer],
      )
      this.outFill = 0
    }
  }

  process(inputs) {
    const input = inputs[0]
    if (!input || input.length === 0) return true
    const ch0 = input[0]
    if (!ch0 || ch0.length === 0) return true

    // sampleRate 是 AudioWorkletGlobalScope 暴露的全局常量
    const srcRate = sampleRate
    const step = srcRate / this.targetSampleRate // >1 表示下采样（如 48000/16000=3）

    // 同采样率 → 直接量化
    if (step === 1) {
      for (let i = 0; i < ch0.length; i++) this._pushSample(floatToInt16(ch0[i]))
      this.prevSample = ch0[ch0.length - 1]
      this.hasPrev = true
      return true
    }

    // 线性插值：srcPos 是下一个输出对应的源流浮点位置（以本块输入 index 0 为 0）
    const n = ch0.length
    while (this.srcPos < n) {
      const idxR = this.srcPos
      const i1 = Math.floor(idxR)
      const frac = idxR - i1
      let s0
      let s1
      if (i1 <= 0) {
        s0 = this.hasPrev ? this.prevSample : ch0[0]
        s1 = ch0[0]
      } else if (i1 >= n) {
        // 理论上 while 已阻断，这里仅防御
        break
      } else {
        s0 = ch0[i1 - 1]
        s1 = ch0[i1]
      }
      const s = s0 + (s1 - s0) * frac
      this._pushSample(floatToInt16(s))
      this.srcPos += step
    }
    // 块结束：srcPos 重定位到"距离本块末尾 -n" 继续累进
    this.srcPos -= n
    this.prevSample = ch0[n - 1]
    this.hasPrev = true
    return true
  }
}

registerProcessor('pcm-recorder', PcmRecorderProcessor)
