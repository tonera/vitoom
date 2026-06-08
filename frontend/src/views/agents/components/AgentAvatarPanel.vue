<template>
  <section
    class="agent-avatar-panel relative w-full overflow-hidden rounded-xl border border-gray-200 bg-gray-50 shadow-sm [.dark_&]:border-gray-800/80 [.dark_&]:bg-gray-950/70"
    :aria-label="t('agents.avatar.label') + ' ' + state"
  >
    <!-- 视频区：固定 16:9，非 live 状态遮一层占位避免黑屏闪烁 -->
    <div class="relative w-full" style="aspect-ratio: 1 / 1">
      <video
        ref="videoElRef"
        class="block h-full w-full bg-gray-900 object-cover"
        :class="{ 'opacity-0': state !== 'live' }"
        muted
        playsinline
        autoplay
      />

      <!-- 状态遮罩 -->
      <div
        v-if="state !== 'live'"
        class="absolute inset-0 flex items-center justify-center bg-gray-900/95 text-center text-xs"
        :class="overlayToneClass"
      >
        <div class="flex flex-col items-center gap-1.5 px-3">
          <span v-if="state === 'unavailable'" class="flex flex-col items-center gap-1">
            <svg class="h-7 w-7 text-gray-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" d="M18 6 6 18M6 6l12 12" />
              <circle cx="12" cy="12" r="9" stroke-linecap="round" />
            </svg>
            <span class="text-gray-300">{{ t('agents.avatar.serviceOffline') }}</span>
            <span class="text-[10px] text-gray-500">{{ errorMessage || t('agents.avatar.startSidecarHint') }}</span>
          </span>
          <span v-else-if="state === 'idle'" class="flex flex-col items-center gap-1">
            <svg class="h-7 w-7 text-gray-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" d="M16 14a4 4 0 1 0-8 0v3h8v-3Z" />
              <circle cx="12" cy="9" r="3" stroke-linecap="round" />
            </svg>
            <span class="text-gray-300">{{ t('agents.avatar.clickToEnable') }}</span>
            <span class="text-[10px] text-gray-500">{{ t('agents.avatar.lipSyncHint') }}</span>
          </span>
          <span v-else-if="state === 'connecting'" class="flex flex-col items-center gap-1">
            <svg class="h-6 w-6 animate-spin text-sky-300" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <circle cx="12" cy="12" r="9" stroke="currentColor" stroke-opacity="0.25" stroke-width="2.5" />
              <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" />
            </svg>
            <span class="text-sky-200">{{ t('agents.avatar.connecting') }}</span>
          </span>
          <span v-else-if="state === 'error'" class="flex flex-col items-center gap-1">
            <svg class="h-6 w-6 text-red-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3m0 3.01.01-.011M5.07 19h13.86c1.54 0 2.5-1.67 1.73-3L13.73 4a2 2 0 0 0-3.46 0L3.34 16c-.77 1.33.19 3 1.73 3Z" />
            </svg>
            <span class="text-red-200">{{ t('agents.avatar.unavailable') }}</span>
            <span class="text-[10px] text-red-300/80">{{ errorMessage || t('agents.avatar.checkSidecar') }}</span>
          </span>
        </div>
      </div>

      <!-- live 状态的关闭按钮 -->
      <button
        v-if="state === 'live'"
        type="button"
        class="absolute right-2 top-2 inline-flex h-6 w-6 items-center justify-center rounded-full bg-black/60 text-white/90 ring-1 ring-white/20 transition hover:bg-black/80"
        :aria-label="t('agents.avatar.close')"
        @click="onToggleClick"
      >
        <svg class="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" d="M6 6l12 12M18 6 6 18" />
        </svg>
      </button>

      <!-- 状态徽章 -->
      <span
        class="absolute left-2 top-2 inline-flex items-center gap-1 rounded-full bg-black/55 px-2 py-0.5 text-[10px] font-medium text-white/90 ring-1 ring-white/15"
      >
        <span class="inline-block h-1.5 w-1.5 rounded-full" :class="dotClass" />
        {{ stateLabel }}
      </span>
    </div>

    <!-- 操作条：单一按钮入口（plan: frontend-merge-buttons）。
         unavailable / error / idle 都点同一个按钮，由 composable.toggle()
         内部决定是先做 refreshAvailability 还是直接连接，避免"先点检测再点开启"
         的两步差体验。仅 connecting 期间 disabled 防止重复触发。 -->
    <div class="flex items-center justify-between gap-2 border-t border-gray-200/80 bg-white/80 px-2.5 py-1.5 [.dark_&]:border-gray-800/80 [.dark_&]:bg-gray-900/60">
      <span class="truncate text-[11px] text-gray-500 [.dark_&]:text-gray-400">{{ t('agents.avatar.label') }}</span>
      <div class="flex shrink-0 items-center gap-1.5">
        <button
          type="button"
          class="rounded-md px-2 py-0.5 text-[11px] ring-1 disabled:cursor-not-allowed disabled:opacity-60"
          :class="primaryButtonClass"
          :disabled="state === 'connecting'"
          :title="primaryButtonTitle"
          @click="onToggleClick"
        >
          {{ primaryButtonLabel }}
        </button>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
/**
 * AgentAvatarPanel —— 装饰性数字人 5 态展示（unavailable/idle/connecting/live/error）。
 *
 * 关注点分离：本组件只做"展示 + 触发 toggle"，所有 WebRTC 状态机由
 * `useLiveTalkingAvatar` 在父组件持有。本组件用 props 接收状态、用
 * `attach-video` 回调把内部 `<video>` 句柄交回 composable。
 *
 * 操作约定（plan: frontend-merge-buttons）：单一按钮入口。`onToggle` 是唯一
 * 触发点，`useLiveTalkingAvatar.toggle()` 内部根据当前状态自动决定"先预检
 * 再连接"还是"直接断开"。`onRefreshAvailability` 字段保留只为后向兼容，
 * 本组件不再调用。
 */
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import type { AvatarState } from '../../../composables/useLiveTalkingAvatar'

const { t } = useI18n()

const props = defineProps<{
  state: AvatarState
  enabled: boolean
  errorMessage: string | null
  onToggle: () => void
  /** @deprecated 单一按钮模式下不再使用；保留 prop 让老调用方平滑升级。 */
  onRefreshAvailability?: () => void
  attachVideo: (el: HTMLVideoElement | null) => void
}>()

const videoElRef = ref<HTMLVideoElement | null>(null)

watch(
  videoElRef,
  (el) => {
    props.attachVideo(el)
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  // 显式解绑，避免 composable 持有已销毁的 DOM 引用
  props.attachVideo(null)
})

const stateLabel = computed(() => {
  switch (props.state) {
    case 'unavailable':
      return t('agents.avatar.statusOffline')
    case 'idle':
      return t('agents.avatar.statusIdle')
    case 'connecting':
      return t('agents.avatar.statusConnecting')
    case 'live':
      return t('agents.avatar.statusLive')
    case 'error':
      return t('agents.avatar.statusError')
    default:
      return ''
  }
})

const dotClass = computed(() => {
  switch (props.state) {
    case 'live':
      return 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.9)]'
    case 'connecting':
      return 'bg-sky-400 animate-pulse'
    case 'error':
      return 'bg-red-400'
    case 'unavailable':
      return 'bg-gray-500'
    default:
      return 'bg-gray-400'
  }
})

const overlayToneClass = computed(() => {
  if (props.state === 'connecting') return 'text-sky-200'
  if (props.state === 'error') return 'text-red-200'
  if (props.state === 'unavailable') return 'text-gray-300'
  return 'text-gray-300'
})

// 单一按钮文案/样式：以"用户下一步要做什么"为视角，而不是"现在是什么状态"。
// - unavailable: 用户已经知道未上线，按钮要给"再试一次连"的预期 → "启用"
// - error: 上一次失败，明确"再试一次" → "重试"
// - idle: 标准开启
// - live: 关闭
// - connecting: 显示进度文案 + disabled
const primaryButtonLabel = computed(() => {
  switch (props.state) {
    case 'connecting':
      return t('agents.avatar.btnConnecting')
    case 'live':
      return t('agents.avatar.btnClose')
    case 'error':
      return t('agents.avatar.btnRetry')
    case 'unavailable':
      return t('agents.avatar.btnEnable')
    case 'idle':
    default:
      return props.enabled ? t('agents.avatar.btnClose') : t('agents.avatar.btnOpen')
  }
})

const primaryButtonClass = computed(() => {
  // live/已开启：醒目蓝色（呼应"正在直播"）
  if (props.state === 'live' || (props.state === 'idle' && props.enabled)) {
    return 'bg-sky-600/80 text-white ring-sky-300/60 hover:bg-sky-500/80'
  }
  // error：稍深的中性色，区别于普通"开启"
  if (props.state === 'error') {
    return 'bg-gray-800 text-gray-100 ring-gray-600 hover:bg-gray-700'
  }
  // 其余（idle 未启用 / unavailable / connecting）：通用灰色按钮
  return 'bg-gray-200 text-gray-700 ring-gray-300 hover:bg-gray-300 [.dark_&]:bg-gray-800 [.dark_&]:text-gray-100 [.dark_&]:ring-gray-600 [.dark_&]:hover:bg-gray-700'
})

const primaryButtonTitle = computed(() => {
  if (props.state === 'unavailable') {
    return props.errorMessage
      ? t('agents.avatar.enableRetryHint', { message: props.errorMessage })
      : t('agents.avatar.enableHint')
  }
  if (props.state === 'error') {
    return props.errorMessage || t('agents.avatar.handshakeFailed')
  }
  return ''
})

function onToggleClick() {
  if (props.state === 'connecting') return
  props.onToggle()
}
</script>
