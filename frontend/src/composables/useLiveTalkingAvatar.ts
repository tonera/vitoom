import { onBeforeUnmount, onMounted, readonly, ref, shallowRef, watch, type Ref } from 'vue'
import { getAccessToken as getProjectAccessToken } from '../utils/auth'
import { translate } from '../utils/translate'

/**
 * LiveTalking 数字人 composable。
 *
 * 详见 .cursor/plans/livetalking_装饰接入_*.plan.md
 * 与 docs/数字人口型音视频对齐接力文档.md（方案 A）。
 *
 * 职责严格收敛：
 * - mount 时调 `GET /api/avatar/status` 预查 sidecar 是否在线 + 拿到
 *   `webrtc_offer_url`（sidecar 对外可达的 /offer 入口，由后端
 *   `config/app.yaml` 的 `server.livetalking_url` 声明）
 * - **直接 POST 到 `webrtc_offer_url`** 建 WebRTC（不再经过后端反向代理，
 *   少一跳后端 SDP 中转，部署形态与 LiveTalking 官方 demo 一致）
 * - SDP 一律 offer audio + video 两路 recvonly 通道；sidecar 是否真的
 *   addTrack(audio) 由其 env `VITOOM_LIVETALKING_AV_SYNC_MODE` 决定，
 *   前端用 `ontrack` 回调里 `event.track.kind` 判断是否有 remote audio
 * - **D 方案 / 装饰模式**（默认）：sidecar 不 addTrack(audio) → 前端 audio
 *   element 永远 muted，本地 `useAudioPlayback` 是音频权威源
 * - **方案 A / 对齐模式**：sidecar addTrack(audio) → 前端 audio element
 *   解除静音，浏览器自动 lip-sync。本组件通过 `unmuteRemoteAudio`
 *   reactive ref 决定是否解除静音；上层根据 `hasRemoteAudio` 与业务策略
 *   联动 `useAudioPlayback.setRemoteAudioActive`
 * - 通过传入的 `sendToggle` 回调通知 chat WS 后端 enabled 变化
 * - 提供 `cancel()` / `teardown()` 给 interrupt / unmount 路径调用
 *
 * **不做的事**：
 * - 不转发 PCM（PCM 由后端 livetalking_client 旁路推到 sidecar）
 * - 不重连（sidecar 挂掉后由用户手动 toggle / 重新检测恢复）
 * - 不解释后端错误码（一律展示成 `error` 状态）
 *
 * 状态机（5 态）：
 *
 *   unavailable ←─────────────────────── (status 接口报 available=false)
 *      │   ▲
 *      │   │ refresh 后仍不可用
 *      │   │
 *      │ toggle()→refresh→若可用则继续连
 *      ▼
 *    idle ──toggle()──▶ connecting ──握手成功──▶ live
 *      ▲                    │
 *      │                    └─握手失败─▶ error  (用户点 toggle 重试)
 *      │                                  │
 *      └──────────toggle() / teardown()───┘
 *
 * 用户体验约定（plan: frontend-merge-buttons）：
 * - 单一按钮"开启/关闭/重试"，不暴露独立"重新检测"。
 * - `toggle()` 在 `unavailable` / `error` 态下会先做一次 `refreshAvailability` 兜底，
 *   成功则继续走连接流程，失败则维持原 `unavailable` / 切回 `unavailable`。
 *   这样用户从未上线 → 上线后只要点一次按钮就能直接进入 live，不需要先点检测再点开启。
 */

export type AvatarState = 'unavailable' | 'idle' | 'connecting' | 'live' | 'error'

const STATUS_ENDPOINT = '/api/avatar/status'

export interface UseLiveTalkingAvatarOptions {
  /** 当前 chat session id，决定 sidecar 把视频流绑到哪个 AvatarSession。 */
  sessionId: Ref<string | null>
  /**
   * 数字人开关变化回调，用来通过 chat WS 给后端发 `avatar_toggle`。
   * 实现侧应该 fire-and-forget，失败 swallow，绝不抛异常。
   */
  sendToggle: (enabled: boolean) => void
  /**
   * 携带给 `/api/avatar/status` 的 Bearer token 读取器（可选）。
   * 默认走 `frontend/src/utils/auth.ts` 的 `getAccessToken()`。
   * 注意：直接 POST 到 sidecar `/offer` 时**不带** token（sidecar 不校验）。
   */
  getAccessToken?: () => string | null
  /**
   * 是否解除 remote audio 元素静音。reactive：上层（AgentChat.vue）根据
   * sidecar 是否真的推 audio track（`hasRemoteAudio`）+ 业务策略决定。
   *
   * - `false` / 未传：D 方案兼容，audio element 永远 muted（默认）
   * - `true`：方案 A 对齐模式，audio element 取消静音让浏览器播 remote
   *   audio；上层必须**同时**调 `useAudioPlayback.setRemoteAudioActive(true)`
   *   禁掉本地 PCM 解码出声，否则会双倍出声
   */
  unmuteRemoteAudio?: Ref<boolean>
}

export interface UseLiveTalkingAvatar {
  state: Ref<AvatarState>
  errorMessage: Ref<string | null>
  enabled: Ref<boolean>
  /**
   * 当前 PeerConnection 是否收到了 sidecar 推过来的 audio track。
   * - `false`（D 方案 / sidecar 未启用 aligned 模式）：上层应保持本地
   *   `useAudioPlayback` 出声
   * - `true`（方案 A）：上层应在 state==='live' 时调
   *   `useAudioPlayback.setRemoteAudioActive(true)` 切换权威源
   */
  hasRemoteAudio: Ref<boolean>
  /** 由模板绑定到 `<video ref="videoEl">`。设为 null 解绑；非 null 时立即 attach 已存在的 stream。 */
  attachVideo: (el: HTMLVideoElement | null) => void
  /**
   * 单击头像区域的统一入口：
   * - `live` → idle 断流
   * - `idle` → connecting → live
   * - `error` → 重置错误后立刻再次尝试连接
   * - `unavailable` → 先 refreshAvailability 兜底，可用则直接连，不可用则维持 unavailable
   *   （前端不再暴露独立的"重新检测"按钮，避免两步操作的差体验）
   * - `connecting` → no-op
   */
  toggle: () => Promise<void>
  /** 主动重新探测 sidecar 是否上线（兼容老调用方；常规路径走 toggle 即可）。 */
  refreshAvailability: () => Promise<void>
  /** interrupt 路径快速 hook：暂停视频显示但不断 WebRTC。 */
  cancel: () => void
  /** unmount / 切会话：彻底清掉。 */
  teardown: () => void
}

function defaultGetAccessToken(): string | null {
  // 与 utils/api.ts 拦截器对齐
  try {
    return getProjectAccessToken()
  } catch {
    return null
  }
}

export function useLiveTalkingAvatar(
  opts: UseLiveTalkingAvatarOptions,
): UseLiveTalkingAvatar {
  // 初始 unavailable，由 onMounted 拉 status 后再决定 → idle / 维持 unavailable
  const state = ref<AvatarState>('unavailable')
  const errorMessage = ref<string | null>(null)
  const enabled = ref<boolean>(false)
  const hasRemoteAudio = ref<boolean>(false)
  // sidecar /offer 的对外可达 URL，由 GET /api/avatar/status 提供
  const webrtcOfferUrl = ref<string | null>(null)

  const pcRef = shallowRef<RTCPeerConnection | null>(null)
  const remoteStreamRef = shallowRef<MediaStream | null>(null)
  const videoElRef = shallowRef<HTMLVideoElement | null>(null)
  // 防止竞态：toggle 期间用户再次点击，旧的 connect 流程不能再覆盖新状态
  let connectGen = 0

  const accessTokenGetter = () =>
    (opts.getAccessToken ?? defaultGetAccessToken)()

  const _setError = (msg: string) => {
    errorMessage.value = msg
    state.value = 'error'
  }

  const _setUnavailable = (reason: string | null) => {
    errorMessage.value = reason
    state.value = 'unavailable'
    enabled.value = false
    hasRemoteAudio.value = false
    webrtcOfferUrl.value = null
  }

  /**
   * 计算当前应否解除 audio element 静音：
   *
   * 必须三个条件同时满足才解除：
   * 1. 调用方明确要求（`unmuteRemoteAudio.value === true`）
   * 2. sidecar 真的推了 audio track（`hasRemoteAudio.value`）；否则解除静音
   *    没有意义（甚至可能让浏览器把 D 方案下本地 audioPlayback 输出的声音
   *    经 video 元素混入而出现回声）
   * 3. WebRTC 已 live；非 live 状态下 srcObject 可能是上一轮残留
   *
   * 任一条件不满足 → muted=true，回退 D 方案行为。
   */
  const _shouldUnmuteRemoteAudio = (): boolean => {
    return (
      opts.unmuteRemoteAudio?.value === true
      && hasRemoteAudio.value
      && state.value === 'live'
    )
  }

  const _applyMutedFlag = () => {
    const el = videoElRef.value
    if (!el) return
    const targetMuted = !_shouldUnmuteRemoteAudio()
    if (el.muted !== targetMuted) {
      el.muted = targetMuted
    }
  }

  const _attachStreamIfReady = () => {
    const el = videoElRef.value
    const stream = remoteStreamRef.value
    if (el && stream && el.srcObject !== stream) {
      el.srcObject = stream
      el.play?.().catch(() => {
        /* autoplay restricted, will play on user gesture */
      })
    }
  }

  const attachVideo = (el: HTMLVideoElement | null) => {
    videoElRef.value = el
    if (el === null) return
    // 默认 muted；_applyMutedFlag() 根据 unmuteRemoteAudio + hasRemoteAudio
    // + state 决定真实值（D 方案下保持 muted；方案 A live 时解除静音）
    el.muted = true
    el.autoplay = true
    el.playsInline = true
    _attachStreamIfReady()
    _applyMutedFlag()
  }

  // unmuteRemoteAudio / hasRemoteAudio / state 任一变化都重新评估静音标志，
  // 让上层切换 av-sync 模式时不需要重连 WebRTC。
  if (opts.unmuteRemoteAudio) {
    watch(opts.unmuteRemoteAudio, _applyMutedFlag)
  }
  watch(hasRemoteAudio, _applyMutedFlag)
  watch(state, _applyMutedFlag)

  const _detachStream = () => {
    const el = videoElRef.value
    if (el) {
      try {
        el.srcObject = null
      } catch {
        /* ignore */
      }
    }
  }

  const _closePc = (pc: RTCPeerConnection | null) => {
    if (!pc) return
    try {
      pc.getReceivers().forEach((r) => {
        try {
          r.track?.stop()
        } catch {
          /* ignore */
        }
      })
    } catch {
      /* ignore */
    }
    try {
      pc.close()
    } catch {
      /* ignore */
    }
  }

  const _hardReset = () => {
    _detachStream()
    remoteStreamRef.value = null
    hasRemoteAudio.value = false
    _closePc(pcRef.value)
    pcRef.value = null
  }

  /**
   * 拉 sidecar 注册状态。available=true → 切到 idle + 缓存 webrtc_offer_url；
   * available=false / 接口失败 → 维持/回到 unavailable + reason 文案。
   *
   * 不影响已经在 live 的连接（连接断开会由 connectionstatechange 单独处理）。
   */
  const refreshAvailability = async (): Promise<void> => {
    if (state.value === 'live' || state.value === 'connecting') {
      // 进行中的会话不要被状态查询打断
      return
    }
    let response: Response
    try {
      const token = accessTokenGetter()
      response = await fetch(STATUS_ENDPOINT, {
        method: 'GET',
        headers: {
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      })
    } catch (exc) {
      const msg = exc instanceof Error ? exc.message : translate('agents.avatarRuntime.networkError')
      _setUnavailable(translate('agents.avatarRuntime.statusFetchFailed', { msg }))
      return
    }
    if (!response.ok) {
      _setUnavailable(translate('agents.avatarRuntime.statusHttpError', { status: response.status }))
      return
    }
    let body: { available?: boolean; reason?: string; webrtc_offer_url?: string } = {}
    try {
      body = (await response.json()) as typeof body
    } catch {
      _setUnavailable(translate('agents.avatarRuntime.statusNotJson'))
      return
    }
    if (body.available === true) {
      const url = (body.webrtc_offer_url || '').trim()
      if (!url) {
        _setUnavailable(translate('agents.avatarRuntime.missingOfferUrl'))
        return
      }
      webrtcOfferUrl.value = url
      // 第一次从 unavailable 切到 idle；如果当前已是 idle/error 也无害
      if (state.value === 'unavailable') {
        state.value = 'idle'
        errorMessage.value = null
      }
      return
    }
    _setUnavailable(body.reason ?? 'sidecar_unavailable')
  }

  const _doConnect = async (): Promise<void> => {
    const sid = opts.sessionId.value
    if (!sid) {
      _setError(translate('agents.avatarRuntime.noChatSession'))
      return
    }
    const offerUrl = webrtcOfferUrl.value
    if (!offerUrl) {
      // 理论上 toggle 拦截了 unavailable，这里是 belt-and-suspenders
      _setUnavailable(translate('agents.avatarRuntime.missingSidecarUrl'))
      return
    }

    _hardReset()
    const myGen = ++connectGen
    state.value = 'connecting'
    errorMessage.value = null

    let pc: RTCPeerConnection
    try {
      pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
      })
    } catch (exc) {
      const msg = exc instanceof Error ? exc.message : 'RTCPeerConnection unsupported'
      _setError(msg)
      return
    }
    pcRef.value = pc

    pc.addEventListener('track', (ev) => {
      if (myGen !== connectGen) return
      const stream = ev.streams[0] ?? new MediaStream([ev.track])
      remoteStreamRef.value = stream
      // 方案 A 检测：任一 audio kind 的 track 进入 → 标记 hasRemoteAudio=true，
      // _applyMutedFlag 再根据 unmuteRemoteAudio 决定是否真的解除静音。
      // sidecar 在 decorative 模式下不 addTrack(audio) → 永远不会进这里 → 保持
      // hasRemoteAudio=false → 保持 muted=true → 完全等价 D 方案。
      if (ev.track?.kind === 'audio') {
        hasRemoteAudio.value = true
      }
      // AV-sync 微调：浏览器 audio jitter buffer 默认 80~200ms 起播延迟（保证不
      // 爆音），video jitter buffer 默认 30~50ms。两边不一致会导致用户主观感
      // 受"video 早于 audio 100~150ms"。playoutDelayHint 是 W3C 标准 API
      // (Chrome 92+)，告诉浏览器"我能容忍多少播放延迟"，浏览器会尽量靠近这个
      // 值。两路都设 0 让浏览器最小化双向延迟，audio/video 起播延迟趋于一致 →
      // lip-sync 改善（实测能把 desync 从 ~150ms 降到 ~30ms）。
      // 这是 hint 不是强制，浏览器有权拒绝；不影响功能，老浏览器没这个属性
      // 直接 swallow。
      try {
        const receiver = ev.receiver as RTCRtpReceiver & { playoutDelayHint?: number }
        if (receiver && 'playoutDelayHint' in receiver) {
          receiver.playoutDelayHint = 0
        }
      } catch {
        /* 老浏览器不支持，忽略 */
      }
      _attachStreamIfReady()
      _applyMutedFlag()
    })
    pc.addEventListener('connectionstatechange', () => {
      if (myGen !== connectGen) return
      const cs = pc.connectionState
      if (cs === 'connected') {
        if (state.value !== 'live') state.value = 'live'
      } else if (cs === 'failed' || cs === 'disconnected') {
        _setError(`WebRTC ${cs}`)
        _hardReset()
      } else if (cs === 'closed') {
        if (state.value === 'live' || state.value === 'connecting') {
          state.value = 'idle'
        }
      }
    })

    // recvonly：浏览器只接收 sidecar 推过来的视频/音频，不上传任何 track。
    try {
      pc.addTransceiver('video', { direction: 'recvonly' })
      pc.addTransceiver('audio', { direction: 'recvonly' })
    } catch {
      /* 部分浏览器可能不支持 addTransceiver，忽略让 ontrack 兜底 */
    }

    let offer: RTCSessionDescriptionInit
    try {
      offer = await pc.createOffer()
      await pc.setLocalDescription(offer)
    } catch (exc) {
      if (myGen !== connectGen) return
      const msg = exc instanceof Error ? exc.message : 'createOffer failed'
      _setError(msg)
      _hardReset()
      return
    }

    let response: Response
    try {
      // 直接 POST 到 sidecar：不带 Authorization（sidecar 不懂 vitoom JWT），
      // 走 sidecar CORS allow * 路径
      response = await fetch(offerUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sdp: pc.localDescription?.sdp ?? offer.sdp,
          type: pc.localDescription?.type ?? offer.type,
          session_id: sid,
        }),
      })
    } catch (exc) {
      if (myGen !== connectGen) return
      const msg = exc instanceof Error ? exc.message : 'network error'
      // 直连 sidecar 的网络错误（DNS / TCP refused / CORS preflight 失败等）
      // 通常意味着 sidecar 真的挂了或 webrtc_offer_url 配错。回退到 unavailable
      // 让用户点"重新检测"重新走 status 接口
      _setUnavailable(translate('agents.avatarRuntime.sidecarDirectFailed', { msg }))
      return
    }

    if (myGen !== connectGen) {
      return
    }

    if (!response.ok) {
      let detail = `${response.status}`
      try {
        const body = await response.json()
        detail = body?.reason || body?.detail || JSON.stringify(body)
      } catch {
        /* ignore */
      }
      _hardReset()
      // sidecar 拒绝（500 模型加载失败 / 503 缺 aiortc 等）→ error 让用户重试
      _setError(translate('agents.avatarRuntime.serviceUnavailable', { detail }))
      return
    }

    let answer: RTCSessionDescriptionInit
    try {
      const body = await response.json()
      if (!body?.sdp || !body?.type) {
        throw new Error('missing sdp/type in response')
      }
      answer = { sdp: body.sdp, type: body.type }
    } catch (exc) {
      const msg = exc instanceof Error ? exc.message : 'invalid response'
      _setError(msg)
      _hardReset()
      return
    }

    try {
      await pc.setRemoteDescription(answer)
    } catch (exc) {
      const msg = exc instanceof Error ? exc.message : 'setRemoteDescription failed'
      _setError(msg)
      _hardReset()
      return
    }
    if (myGen === connectGen && state.value === 'connecting' && remoteStreamRef.value) {
      state.value = 'live'
    }
  }

  const toggle = async (): Promise<void> => {
    if (state.value === 'connecting') return

    // 关闭路径：live/connected → idle，最高优先级
    if (enabled.value) {
      enabled.value = false
      ++connectGen
      _hardReset()
      state.value = 'idle'
      errorMessage.value = null
      try {
        opts.sendToggle(false)
      } catch {
        /* ignore */
      }
      return
    }

    // 开启路径：unavailable / error / idle 全都走"按需预检 → 连接"统一流程，
    // 不再暴露单独的"重新检测"按钮（plan: frontend-merge-buttons）。
    const entryState = state.value
    if (entryState === 'unavailable') {
      // sidecar 之前不在线：先拉一次 status，能拿到 webrtc_offer_url 才往下走
      await refreshAvailability()
      // refresh 后状态被同步改写：可能切到 idle，也可能维持 unavailable
      if ((state.value as AvatarState) !== 'idle') {
        // 仍不可用：状态/错误信息已由 refreshAvailability 落盘，不继续连接
        return
      }
    } else if (entryState === 'error') {
      // 上一次握手失败：清掉错误回到 idle，然后继续走 _doConnect
      errorMessage.value = null
      state.value = 'idle'
    }

    enabled.value = true
    try {
      opts.sendToggle(true)
    } catch {
      /* ignore */
    }
    await _doConnect()
  }

  const cancel = () => {
    // **故意 no-op**。装饰性数字人 D 方案下视频是 always-on idle 显示：
    // - 用户开始说话 / interrupt / auto-commit 时本地 audio 走自己的 cancel
    //   通路（useAudioPlayback），与数字人视频解耦
    // - sidecar 端的"打断当前句子重新生成口型"由后端 InterruptCoordinator
    //   旁路推 ``flush_talk`` 给 sidecar 完成，前端不参与
    // 之前这里 ``el.pause()`` 是错的：浏览器 <video> 一旦 pause，WebRTC
    // 后续推过来的视频帧不会自动播放，结果就是"开始聊天后数字人卡住静止"。
  }

  const teardown = () => {
    ++connectGen
    enabled.value = false
    _hardReset()
    if (state.value !== 'unavailable') {
      state.value = 'idle'
    }
    errorMessage.value = null
  }

  onMounted(() => {
    void refreshAvailability()
  })

  onBeforeUnmount(() => {
    teardown()
  })

  return {
    state: readonly(state) as Ref<AvatarState>,
    errorMessage: readonly(errorMessage) as Ref<string | null>,
    enabled: readonly(enabled) as Ref<boolean>,
    hasRemoteAudio: readonly(hasRemoteAudio) as Ref<boolean>,
    attachVideo,
    toggle,
    refreshAvailability,
    cancel,
    teardown,
  }
}
