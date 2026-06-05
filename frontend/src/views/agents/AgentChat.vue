<template>
  <div
    class="agent-chat vt-surface vt-text-smooth relative flex h-full min-h-0 w-full min-w-0 flex-col overflow-hidden antialiased text-gray-950 [.dark_&]:text-gray-100"
  >
    <!-- 顶部状态条 -->
    <div
      v-if="phase !== 'ready' || errorBanner || recoverableToast"
      class="shrink-0 border-b border-gray-800/80 bg-gray-900/90 px-3 py-2 text-xs sm:px-4"
    >
      <p v-if="phase === 'connecting' || phase === 'opening'" class="text-gray-400">{{ t('agents.chat.connecting') }}</p>
      <p v-else-if="phase === 'closed'" class="text-gray-500">{{ t('agents.session.closed') }}</p>
      <div v-else-if="errorBanner" class="flex flex-wrap items-center justify-between gap-2">
        <span class="text-red-300">{{ errorBanner }}</span>
        <button
          type="button"
          class="rounded-lg bg-gray-800 px-2.5 py-1 text-gray-100 ring-1 ring-gray-600 hover:bg-gray-700"
          @click="onRetryWs"
        >
          {{ t('agents.chat.reconnect') }}
        </button>
      </div>
      <p v-if="recoverableToast" class="mt-1 text-amber-200/90">{{ recoverableToast }}</p>
    </div>

    <!-- P1 PR-1/PR-2 dev-only：麦克风采集 + VAD 调试条（生产构建下整体消失） -->
    <!-- <div
      v-if="isDev"
      class="shrink-0 border-b border-amber-900/60 bg-amber-950/40 px-3 py-2 text-xs text-amber-100 sm:px-4"
    >
      <div class="flex flex-wrap items-center gap-2">
        <span class="font-mono text-[10px] text-amber-300/80">[dev] audio</span>
        <button
          type="button"
          class="rounded-lg bg-amber-600/80 px-2.5 py-1 text-amber-50 ring-1 ring-amber-400/60 hover:bg-amber-500 disabled:opacity-50"
          :disabled="audioDebug.busy"
          @click="onAudioDebugToggle"
        >
          {{ audioDebug.micState === 'recording' ? '停止录音' : '开始录音' }}
        </button>
        <button
          type="button"
          class="rounded-lg px-2 py-1 text-[11px] ring-1 ring-amber-400/60"
          :class="
            audioDebug.vadEnabled
              ? 'bg-emerald-700/70 text-emerald-50 hover:bg-emerald-600/70'
              : 'bg-gray-700/60 text-gray-200 hover:bg-gray-600/70'
          "
          @click="onToggleVad"
        >
          VAD: {{ audioDebug.vadEnabled ? 'on' : 'off' }}
        </button>
        <span
          class="inline-flex items-center gap-1 font-mono text-[11px]"
          :class="audioDebug.speaking ? 'text-emerald-300' : 'text-gray-400'"
        >
          <span
            class="inline-block h-2 w-2 rounded-full"
            :class="audioDebug.speaking ? 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.9)]' : 'bg-gray-600'"
          />
          {{ audioDebug.speaking ? 'speaking' : 'silent' }}
        </span>
        <span class="font-mono text-[11px] text-amber-200/90">
          state={{ audioDebug.micState }} · dur={{ audioDebug.durationMs }}ms · frames={{ audioDebug.frames }} ·
          rms={{ audioDebug.rmsDisplay }}
        </span>
      </div>
      <div class="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-amber-200/80">
        <span>uplink={{ audioDebug.uplinkFrames }} ({{ audioDebug.uplinkBytes }}B)</span>
        <span>dropped={{ audioDebug.droppedFrames }}</span>
        <span>segments={{ audioDebug.segments }}</span>
        <span v-if="audioDebug.lastFrameMs !== null" class="text-amber-200/60">
          sr={{ audioDebug.sampleRate }} · frameMs={{ audioDebug.lastFrameMs }}
        </span>
        <span v-if="audioDebug.lastVadEvent" class="text-sky-300/90">last: {{ audioDebug.lastVadEvent }}</span>
        <span v-if="audioDebug.error" class="text-red-300">{{ audioDebug.error }}</span>
      </div>
      <div class="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px]">
        <span class="text-amber-300/80">[playback]</span>
        <span
          class="inline-flex items-center gap-1"
          :class="
            audioPlaybackDebug.state === 'playing'
              ? 'text-sky-300'
              : audioPlaybackDebug.state === 'buffering'
                ? 'text-amber-300'
                : audioPlaybackDebug.state === 'error'
                  ? 'text-red-300'
                  : 'text-gray-400'
          "
        >
          <span
            class="inline-block h-2 w-2 rounded-full"
            :class="{
              'bg-sky-400 shadow-[0_0_6px_rgba(56,189,248,0.9)]': audioPlaybackDebug.state === 'playing',
              'bg-amber-400': audioPlaybackDebug.state === 'buffering',
              'bg-red-400': audioPlaybackDebug.state === 'error',
              'bg-gray-600': audioPlaybackDebug.state === 'idle',
            }"
          />
          {{ audioPlaybackDebug.state }}
        </span>
        <span class="text-amber-200/80">
          queue={{ audioPlaybackDebug.queueLength }} · enq={{ audioPlaybackDebug.enqueued }} ·
          played={{ audioPlaybackDebug.played }} · cancelled={{ audioPlaybackDebug.cancelled }}
        </span>
        <span v-if="audioPlaybackDebug.lastSampleRate > 0" class="text-amber-200/60">
          sr={{ audioPlaybackDebug.lastSampleRate }} · chunkMs={{ audioPlaybackDebug.lastChunkMs }}
        </span>
        <button
          type="button"
          class="rounded-lg bg-gray-700/70 px-2 py-0.5 text-[11px] text-gray-100 ring-1 ring-gray-500/60 hover:bg-gray-600/70"
          @click="onPlaybackCancel"
        >
          cancel
        </button>
        <span v-if="audioPlaybackDebug.error" class="text-red-300">{{ audioPlaybackDebug.error }}</span>
      </div>
    </div> -->

    <div
      class="agent-chat-layout relative flex min-h-0 min-w-0 flex-1 flex-col"
    >
      <!-- 消息区：全宽滚动，滚动条在内容区最右侧（不再夹在消息与事件卡片之间） -->
      <div ref="chatColumnRef" class="relative flex min-h-0 min-w-0 flex-1 flex-col">
        <div
          ref="scrollRef"
          class="agent-chat-scroll min-h-0 flex-1 overflow-y-auto px-3 pb-3 pt-3 pr-[calc(var(--agent-chat-rail)+0.75rem)] sm:px-4 sm:pb-4 sm:pt-4 sm:pr-[calc(var(--agent-chat-rail)+1rem)]"
        >
          <div
            class="mx-auto flex w-full max-w-3xl flex-col gap-3 xl:max-w-4xl 2xl:max-w-5xl"
          >
            <div v-if="historyHasMore && phase === 'ready'" class="flex justify-center pb-1">
              <button
                type="button"
                class="rounded-full border border-gray-200 bg-white px-3 py-1.5 text-xs text-gray-600 shadow-sm hover:bg-gray-50 hover:text-gray-950 disabled:cursor-not-allowed disabled:opacity-50 [.dark_&]:border-gray-700/80 [.dark_&]:bg-gray-900/80 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-gray-100"
                :disabled="historyLoadingMore"
                @click="onLoadMoreHistory"
              >
                {{ historyLoadingMore ? t('common.loading') : t('agents.chat.loadEarlierMessages') }}
              </button>
            </div>
            <AnimatePresence>
              <motion.article
                v-for="msg in messages"
                :key="msg.id"
                class="flex w-full flex-col"
                :class="isUserSideMessage(msg) ? 'items-end' : 'items-start'"
                :initial="{ opacity: 0, y: 10, scale: 0.98 }"
                :animate="{ opacity: 1, y: 0, scale: 1 }"
                :exit="{ opacity: 0, y: -6, scale: 0.98 }"
                :transition="springSoft"
              >
                <div
                  class="agent-msg-bubble max-w-[min(100%,var(--agent-chat-bubble-max-width))] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed shadow-sm wrap-anywhere sm:px-4 sm:py-3 sm:text-[15px]"
                  :class="bubbleClass(msg)"
                >
                  <div
                    v-if="isAssistantThinkingMessage(msg)"
                    class="flex items-center gap-3 text-gray-500 [.dark_&]:text-gray-300"
                  >
                    <span class="text-gray-500 [.dark_&]:text-gray-400">Agent</span>
                    <span class="flex items-center gap-1" aria-hidden="true">
                      <motion.span
                        class="inline-block h-1.5 w-1.5 rounded-full bg-gray-400"
                        :animate="{ y: [0, -4, 0], opacity: [0.45, 1, 0.45] }"
                        :transition="{ repeat: Infinity, duration: 0.75, ease: 'easeInOut' }"
                      />
                      <motion.span
                        class="inline-block h-1.5 w-1.5 rounded-full bg-gray-400"
                        :animate="{ y: [0, -4, 0], opacity: [0.45, 1, 0.45] }"
                        :transition="{ repeat: Infinity, duration: 0.75, delay: 0.12, ease: 'easeInOut' }"
                      />
                      <motion.span
                        class="inline-block h-1.5 w-1.5 rounded-full bg-gray-400"
                        :animate="{ y: [0, -4, 0], opacity: [0.45, 1, 0.45] }"
                        :transition="{ repeat: Infinity, duration: 0.75, delay: 0.24, ease: 'easeInOut' }"
                      />
                    </span>
                  </div>
                  <div
                    v-else-if="msg.role === 'assistant'"
                    class="agent-md"
                    :class="assistantMarkdownClass"
                    v-html="assistantBubbleHtml(msg)"
                    @click="onAssistantMdClick"
                  />
                  <p v-else class="whitespace-pre-wrap">{{ msg.text }}</p>
                  <ChatMessageArtifacts v-if="msg.artifacts?.length" :items="msg.artifacts" />
                  <AgentAssistantMessageMenu
                    v-if="msg.role === 'assistant' && String(msg.text || '').trim()"
                    :raw-markdown="msg.text"
                  />
                </div>
              </motion.article>
            </AnimatePresence>
          </div>
        </div>

        <!-- 运行事件：脱离文档流贴在主消息区右侧，主列表单独纵向滚动 -->
        <div
          class="pointer-events-none absolute inset-x-0 top-3 z-20 px-3 pr-[calc(var(--agent-chat-rail)+0.75rem)] sm:top-4 sm:px-4 sm:pr-[calc(var(--agent-chat-rail)+1rem)]"
        >
          <div class="relative mx-auto w-full max-w-3xl xl:max-w-4xl 2xl:max-w-5xl">
            <aside
              class="agent-chat-events-card pointer-events-auto absolute left-[calc(100%+0.75rem)] top-0 w-(--agent-chat-rail) max-w-[min(var(--agent-chat-rail),calc(100%-1.5rem))] min-w-0 overflow-x-hidden rounded-2xl border border-gray-200 bg-white/95 p-3 shadow-lg ring-1 ring-gray-100 sm:left-[calc(100%+1rem)] [.dark_&]:border-gray-700/80 [.dark_&]:bg-gray-900/95 [.dark_&]:ring-black/20"
              role="complementary"
              :aria-label="t('agents.chat.runEvents')"
            >
        <!-- 装饰性数字人面板（plan: livetalking_装饰接入）。仅出视频，不出声。
             5 态：unavailable（sidecar 未注册/未运行，按钮置灰）/ idle / connecting / live / error。
             状态来源完全等价于后端 inference_services.status，不再有 ENABLED env 总开关。 -->
        <AgentAvatarPanel v-if="avatar.enabled.value"
          class="mb-2.5"
          :state="avatar.state.value"
          :enabled="avatar.enabled.value"
          :error-message="avatar.errorMessage.value"
          :on-toggle="avatar.toggle"
          :on-refresh-availability="avatar.refreshAvailability"
          :attach-video="avatar.attachVideo"
        />
        <div class="border-b border-gray-200 pb-2.5 [.dark_&]:border-gray-800/80">
          <motion.div
            :initial="{ opacity: 0, x: 8 }"
            :animate="{ opacity: 1, x: 0 }"
            :transition="springSoft"
            class="flex items-center justify-between gap-2"
          >
            <h2 class="text-sm font-medium text-gray-900 [.dark_&]:text-gray-200">{{ t('agents.chat.runEventsTitle', { status: currentStatus }) }}</h2>
            <span class="font-mono text-[10px] text-gray-500">{{ t('agents.chat.recentItems', { count: ACTIVITY_MAX_VISIBLE }) }}</span>
          </motion.div>
          
          <p
            v-if="lastTurnUsageLine"
            class="mt-1 text-[10px] leading-snug"
            :class="lastTurnUsageLine.tone === 'muted' ? 'text-gray-500' : 'font-mono text-gray-500 [.dark_&]:text-gray-400'"
          >
            {{ lastTurnUsageLine.text }}
          </p>
        </div>

        <div
          class="event-viewport mt-2 overflow-x-hidden overflow-y-auto px-0.5"
          :style="{ height: `${ACTIVITY_VIEWPORT_PX}px` }"
        >
          <div class="flex min-h-full min-w-0 flex-col justify-start gap-1.5 py-1">
            <AnimatePresence initial>
              <motion.div
                v-for="(ev, i) in activity"
                :key="ev.id"
                layout
                class="max-w-full min-w-0 overflow-hidden rounded-lg border border-gray-200 bg-gray-50 px-2.5 py-2 [.dark_&]:border-gray-800/90 [.dark_&]:bg-gray-950/70"
                :initial="{ opacity: 0, y: -10, scale: 0.98 }"
                :animate="{ opacity: 1, y: 0, scale: 1 }"
                :exit="{ opacity: 0, y: 8, scale: 0.96 }"
                :transition="{ ...springSoft, delay: Math.min(i * 0.03, 0.15) }"
              >
                <div class="flex min-w-0 items-start justify-between gap-2">
                  <span
                    class="mt-0.5 inline-flex shrink-0 rounded px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wide"
                    :class="levelClass(ev.level)"
                  >
                    {{ ev.level }}
                  </span>
                  <time class="shrink-0 font-mono text-[9px] text-gray-500">{{ ev.time }}</time>
                </div>
                <button
                  type="button"
                  class="mt-1 w-full min-w-0 cursor-pointer rounded-md px-0 text-left text-[11px] font-medium leading-snug text-gray-800 transition-colors hover:bg-gray-200/80 hover:text-gray-950 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800/80 [.dark_&]:hover:text-white"
                  @click.stop="openActivityDetail(ev)"
                >
                  {{ ev.title }}
                </button>
                <p
                  v-if="ev.detail"
                  class="mt-0.5 min-w-0 break-words text-[10px] leading-snug text-gray-500 [overflow-wrap:anywhere]"
                >
                  {{ ev.detail }}
                </p>
                <div v-if="ev.progress != null" class="mt-1.5 h-0.5 overflow-hidden rounded-full bg-gray-200 [.dark_&]:bg-gray-800">
                  <motion.div
                    class="h-full rounded-full bg-blue-500/90"
                    :initial="{ width: '0%' }"
                    :animate="{ width: `${ev.progress}%` }"
                    :transition="{ type: 'spring', stiffness: 260, damping: 28 }"
                  />
                </div>
              </motion.div>
            </AnimatePresence>
          </div>
        </div>
            </aside>
          </div>
        </div>

        <Teleport to="body">
          <div
            v-if="activityDetailOpen && activityDetailItem"
            class="fixed inset-0 z-[10050] flex items-center justify-center bg-black/60 p-4"
            role="dialog"
            aria-modal="true"
            aria-labelledby="activity-detail-title"
            @click.self="closeActivityDetail()"
          >
            <div
              class="max-h-[min(85vh,640px)] w-full max-w-lg overflow-hidden rounded-2xl border border-gray-200 bg-white p-4 shadow-xl [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900"
              @click.stop
            >
              <div class="flex items-start justify-between gap-2">
                <h3
                  id="activity-detail-title"
                  class="min-w-0 flex-1 pr-2 text-sm font-semibold leading-snug text-gray-900 [.dark_&]:text-gray-100"
                >
                  {{ activityDetailItem.title }}
                </h3>
                <button
                  type="button"
                  class="inline-flex h-8 w-8 shrink-0 cursor-pointer items-center justify-center rounded-lg text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-900 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-gray-100"
                  :aria-label="t('common.close')"
                  @click="closeActivityDetail()"
                >
                  <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <p class="mt-1 font-mono text-[10px] text-gray-500 [.dark_&]:text-gray-400">
                {{ activityDetailItem.time }} · {{ activityDetailItem.level }}
              </p>
              <div
                class="mt-3 max-h-[min(55vh,420px)] overflow-y-auto overflow-x-hidden rounded-lg bg-gray-50 p-3 [.dark_&]:bg-gray-950/80"
              >
                <pre
                  class="whitespace-pre-wrap break-words font-mono text-xs leading-relaxed text-gray-800 [overflow-wrap:anywhere] [.dark_&]:text-gray-200"
                >{{ activityDetailBody(activityDetailItem) }}</pre>
              </div>
            </div>
          </div>
        </Teleport>
      </div>

      <!-- 底栏：在滚动区之下常驻，视觉上与主区分离 -->
      <footer
        class="agent-chat-footer shrink-0  px-0 pb-3 pt-3 sm:pb-4 sm:pt-3.5"
      >
        <div class="min-w-0 px-3 pr-[calc(var(--agent-chat-rail)+0.75rem)] sm:px-4 sm:pr-[calc(var(--agent-chat-rail)+1rem)]">
          <div class="mx-auto w-full max-w-3xl xl:max-w-4xl 2xl:max-w-5xl">
            <AgentHomeComposer
              ref="chatComposerRef"
              :boundary-element="chatColumnRef"
              :submit-disabled="!canSubmit || isRecording"
              :placeholder="composerPlaceholder"
              wrapper-class="mx-auto w-full max-w-3xl shrink-0  xl:max-w-4xl 2xl:max-w-5xl"
              :is-streaming="isStreaming"
              :playback-active="isPlaybackActive"
              :mic-enabled="true"
              :mic-can-start="canStartAudioRecord"
              :mic-recording="isRecording"
              :mic-disabled="micButtonDisabled"
              @submit="onComposerSubmit"
              @mic-toggle="onMicToggle"
              @interrupt="interrupt"
              @stop-playback="onStopPlayback"
            />
          </div>
        </div>
      </footer>
    </div>
  </div>
</template>

<script setup lang="ts">
import 'katex/dist/katex.min.css'
import { AnimatePresence, motion } from 'motion-v'
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { onBeforeRouteLeave, useRoute, useRouter } from 'vue-router'
import { removeRecentChatSession } from '../../api/chat'
import type { AgentComposerSubmitPayload } from './agentComposerTypes'
import AgentHomeComposer from './components/AgentHomeComposer.vue'
import AgentAssistantMessageMenu from './components/AgentAssistantMessageMenu.vue'
import AgentAvatarPanel from './components/AgentAvatarPanel.vue'
import ChatMessageArtifacts from './components/ChatMessageArtifacts.vue'
import { showTopSnack } from '../../composables/useTopSnack'
import type { ActivityItem, ActivityLevel, ChatMessage } from '../../composables/useAgentChatSession'
import { useAgentChatSession } from '../../composables/useAgentChatSession'
import { renderAssistantMarkdown } from '../../utils/agentChatMarkdown'
import { useAudioChat, type PcmFrame, type UplinkFrame, type VadEvent } from '../../composables/useAudioChat'
import { useAudioPlayback } from '../../composables/useAudioPlayback'
import { useLiveTalkingAvatar } from '../../composables/useLiveTalkingAvatar'

const ACTIVITY_VIEWPORT_PX = 300
/** 与 useAgentChatSession 中 ACTIVITY_MAX 一致（最多保留条数） */
const ACTIVITY_MAX_VISIBLE = 100

const springSoft = { type: 'spring' as const, stiffness: 380, damping: 32 }

const assistantMarkdownClass = [
  'min-w-0 max-w-[var(--agent-chat-bubble-max-width)] overflow-x-auto text-pretty text-[0.98rem] leading-[1.9]',
  'text-gray-700 [.dark_&]:text-gray-200',
  '[&_:first-child]:!mt-0 [&_:last-child]:!mb-0',
  '[&_p]:my-[0.72em] [&_p]:leading-[1.92]',
  '[&_ul]:my-[0.85em] [&_ol]:my-[0.85em] [&_ul]:pl-6 [&_ol]:pl-6',
  '[&_li]:my-[0.48em] [&_li]:leading-[1.84] [&_li+li]:mt-[0.22em] [&_li>p]:my-[0.28em] [&_li>p]:leading-[1.84]',
  '[&_strong]:font-bold [&_b]:font-bold [&_strong]:text-gray-950 [&_b]:text-gray-950 [.dark_&_strong]:text-gray-50 [.dark_&_b]:text-gray-50',
  '[&_em]:text-gray-700 [.dark_&_em]:text-gray-100',
  '[&_h1]:mt-[0.7em] [&_h1]:mb-[0.45em] [&_h1]:text-[1.52em] [&_h1]:font-extrabold [&_h1]:leading-[1.2] [&_h1]:tracking-[-0.025em]',
  '[&_h2]:mt-[1.1em] [&_h2]:mb-[0.45em] [&_h2]:border-b [&_h2]:border-gray-200 [&_h2]:pb-[0.28em] [&_h2]:text-[1.28em] [&_h2]:font-bold [&_h2]:leading-[1.2] [&_h2]:tracking-[-0.025em] [.dark_&_h2]:border-gray-500/50',
  '[&_h1]:text-gray-950 [&_h2]:text-gray-950 [.dark_&_h1]:text-gray-50 [.dark_&_h2]:text-gray-50',
  '[&_h3]:mt-[1.15em] [&_h3]:mb-[0.4em] [&_h3]:border-l-2 [&_h3]:border-blue-600/35 [&_h3]:pl-[0.55em] [&_h3]:text-[1.2em] [&_h3]:font-bold [&_h3]:leading-[1.24] [&_h3]:text-gray-950 [.dark_&_h3]:border-sky-300/40 [.dark_&_h3]:text-gray-50',
  '[&_h4]:mt-[0.9em] [&_h4]:mb-[0.24em] [&_h4]:text-[1.06em] [&_h4]:font-semibold [&_h4]:leading-[1.32] [&_h4]:text-gray-700 [.dark_&_h4]:text-gray-200',
  '[&_h5]:mt-[0.7em] [&_h5]:mb-[0.18em] [&_h5]:font-semibold [&_h5]:leading-[1.36] [&_h5]:text-gray-600',
  '[&_h6]:mt-[0.7em] [&_h6]:mb-[0.18em] [&_h6]:font-semibold [&_h6]:leading-[1.36] [&_h6]:text-gray-600 [.dark_&_h5]:text-slate-300 [.dark_&_h6]:text-slate-300',
  '[&_blockquote]:my-[0.8em] [&_blockquote]:border-l-[3px] [&_blockquote]:border-blue-600/35 [&_blockquote]:bg-gradient-to-r [&_blockquote]:from-blue-600/[0.06] [&_blockquote]:to-blue-600/0 [&_blockquote]:py-[0.3em] [&_blockquote]:pl-[0.85em] [&_blockquote]:leading-[1.82] [&_blockquote]:text-gray-700 [.dark_&_blockquote]:border-sky-300/55 [.dark_&_blockquote]:from-sky-400/[0.08] [.dark_&_blockquote]:to-sky-400/0 [.dark_&_blockquote]:text-slate-200',
  '[&_a]:break-words [&_a]:text-blue-600 [&_a]:underline [&_a]:underline-offset-2 hover:[&_a]:text-blue-700 [.dark_&_a]:text-sky-300 [.dark_&]:hover:[&_a]:text-sky-200',
  '[&_code]:rounded [&_code]:bg-gray-900/90 [&_code]:px-[0.38em] [&_code]:py-[0.24em] [&_code]:font-mono [&_code]:text-[0.88em] [&_code]:text-gray-100',
  '[&_pre]:my-[0.8em] [&_pre]:max-w-full [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:border [&_pre]:border-gray-700/70 [&_pre]:bg-gray-950/90 [&_pre]:p-3 [&_pre]:text-gray-300',
  '[&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:text-[0.84em] [&_pre_code]:leading-[1.45] [&_pre_code]:text-inherit',
  '[&_hr]:my-4 [&_hr]:border-0 [&_hr]:border-t [&_hr]:border-gray-200 [.dark_&_hr]:border-gray-700',
  '[&_table]:my-5 [&_table]:w-max [&_table]:max-w-full [&_table]:border-collapse [&_table]:text-[0.88em]',
  '[&_th]:border [&_td]:border [&_th]:border-gray-200 [&_td]:border-gray-200 [&_th]:px-[0.45em] [&_td]:px-[0.45em] [&_th]:py-[0.28em] [&_td]:py-[0.28em] [&_th]:text-left [&_th]:leading-[1.62] [&_td]:leading-[1.62] [&_th]:align-top [&_td]:align-top',
  '[&_th]:bg-gray-100 [&_tr:nth-child(even)_td]:bg-gray-50 [.dark_&_th]:border-gray-600/65 [.dark_&_td]:border-gray-600/65 [.dark_&_th]:bg-gray-800/90 [.dark_&_tr:nth-child(even)_td]:bg-gray-900/35',
  '[&_.katex]:text-[1.05em] [&_.katex]:text-gray-700 [.dark_&_.katex]:text-gray-100',
  '[&_.katex-display]:my-[0.45em] [&_.katex-display]:max-w-full [&_.katex-display]:overflow-x-auto [&_.katex-display]:overflow-y-hidden [&_.katex-display]:py-[0.15em]',
  '[&_.katex-display>.katex]:text-gray-700 [.dark_&_.katex-display>.katex]:text-gray-100',
]

const { t, locale } = useI18n()

const route = useRoute()
const router = useRouter()

const {
  phase,
  messages,
  activity,
  sessionState,
  currentStatus,
  errorMessage,
  recoverableToast,
  isStreaming,
  canSubmit,
  sessionId,
  connect,
  disconnect,
  sendUserMessage,
  interrupt,
  sendAvatarToggle,
  retryConnect,
  loadMoreHistory,
  historyHasMore,
  historyLoadingMore,
  isSessionNotFoundError,
  isSessionForbiddenError,
  onAudioDelta,
  onAudioCancel,
  sendAudioChunk,
  commitAudio,
  canStartAudioRecord,
  supportsAudioInput,
} = useAgentChatSession()

// 数字人副链路（plan: livetalking_装饰接入 + docs/数字人口型音视频对齐接力文档.md 方案 A）。
// failed/disabled 一律 swallow，不污染主聊天链路；状态机仅由 useLiveTalkingAvatar 持有。
//
// 音频权威源切换逻辑（计算属性形式，shouldUseRemoteAudio）：
// - sidecar 推了 audio track（avatar.hasRemoteAudio）+ 数字人 live → 切到 remote audio
// - 其它情况（D 方案 / sidecar 未启用 aligned / 数字人未连）→ 本地 useAudioPlayback 出声
// - 切换瞬间会有 ~200ms 静音空隙（远端 jitter buffer 起播延迟），属预期
const shouldUseRemoteAudio = ref(false)
const avatar = useLiveTalkingAvatar({
  sessionId,
  sendToggle: sendAvatarToggle,
  unmuteRemoteAudio: shouldUseRemoteAudio,
})

/** 与后端 `usage_metrics` 字段对齐（含 master_runtime 的 `tok_s_total`） */
function readUsageNumber(raw: Record<string, unknown> | undefined, keys: string[]): number | null {
  if (!raw) return null
  for (const k of keys) {
    const v = raw[k]
    if (typeof v === 'number' && Number.isFinite(v)) return v
    if (typeof v === 'string' && v.trim() !== '') {
      const n = Number(v)
      if (Number.isFinite(n)) return n
    }
  }
  return null
}

function usageHasTokenMetrics(raw: Record<string, unknown> | undefined): boolean {
  return (
    readUsageNumber(raw, ['total_tokens', 'totalTokens']) != null ||
    readUsageNumber(raw, ['prompt_tokens', 'promptTokens']) != null ||
    readUsageNumber(raw, ['output_tokens', 'completion_tokens', 'completionTokens', 'outputTokens']) != null ||
    readUsageNumber(raw, ['tok_s_total', 'tokens_per_second', 'tokensPerSecond']) != null
  )
}

const lastAssistantForUsage = computed(() => {
  for (let i = messages.value.length - 1; i >= 0; i--) {
    const m = messages.value[i]
    if (m && m.role === 'assistant' && !m.isError) return m
  }
  return null
})

/** 最近一条助手消息上的 usage（message_completed.usage_metrics），表示该轮回复的用量，非整段会话累计 */
const lastTurnUsageLine = computed((): { text: string; tone: 'muted' | 'value' } | null => {
  const msg = lastAssistantForUsage.value
  if (!msg) return null
  const raw = msg.usage as Record<string, unknown> | undefined
  if (msg.streaming && !usageHasTokenMetrics(raw)) {
    return { text: t('agents.chat.usagePending'), tone: 'muted' }
  }
  if (!usageHasTokenMetrics(raw)) return null
  const total = readUsageNumber(raw, ['total_tokens', 'totalTokens'])
  const prompt = readUsageNumber(raw, ['prompt_tokens', 'promptTokens'])
  const output = readUsageNumber(raw, ['output_tokens', 'completion_tokens', 'completionTokens', 'outputTokens'])
  const tokens = total ?? (prompt != null || output != null ? (prompt ?? 0) + (output ?? 0) : null)
  const tps = readUsageNumber(raw, ['tok_s_total', 'tokens_per_second', 'tokensPerSecond'])
  const parts: string[] = []
  if (tokens != null) parts.push(t('agents.chat.usageTokens', { tokens: tokens.toLocaleString(locale.value) }))
  if (tps != null) parts.push(t('agents.chat.usageTps', { tps: tps.toLocaleString(locale.value) }))
  if (parts.length === 0) return null
  return { text: parts.join(' · '), tone: 'value' }
})

const activityDetailOpen = ref(false)
const activityDetailItem = ref<ActivityItem | null>(null)

function activityDetailBody(ev: ActivityItem): string {
  const body = (ev.detailFull ?? ev.detail ?? '').trim()
  return body || t('agents.chat.noMoreDetail')
}

function openActivityDetail(ev: ActivityItem) {
  activityDetailItem.value = ev
  activityDetailOpen.value = true
}

function closeActivityDetail() {
  activityDetailOpen.value = false
  activityDetailItem.value = null
}

// P1 PR-3：TTS 流式 PCM 播放接入
const audioPlayback = useAudioPlayback()
const isPlaybackActive = computed(() => {
  const state = audioPlayback.state.value
  return state === 'buffering' || state === 'playing'
})
const stopAudioDeltaSub = onAudioDelta((ev) => {
  if (!ev.bytes || ev.bytes.byteLength === 0) return
  // remoteAudioActive 模式下 enqueue 内部会 no-op；仍然调用以保持 stats 一致 +
  // 如果切回本地模式时新到的分片能正常入队。
  audioPlayback.enqueue({
    bytes: ev.bytes,
    mime: ev.mime,
    isFinal: ev.isFinal,
    runId: ev.runId,
    sampleRate: ev.sampleRate ?? null,
  })
})
const stopAudioCancelSub = onAudioCancel((ev) => {
  // 仅取消本地 TTS 播放队列；不动 mic 录音 / 上行 forwarder。
  // 语音聊天期间 mic 保持长开，每轮 ASR auto-commit 后用户能直接接着说下一句，
  // 不必每轮重点麦克风按钮。控制词命中（"停"等）/ empty_transcript 等场景下
  // 后端 commit 后状态会回 ready 但不启 LLM 也无 TTS，原先把 mic 一并停掉
  // 之后没有任何路径自动重启，体验上等于"对话被吞了一轮"。
  audioPlayback.cancel(ev.reason)
  // 旁路通知数字人 panel：暂停视频显示帮助"嘴归位"的视觉感。后端 sidecar
  // 的 flush_talk 由 InterruptCoordinator._cancel_avatar 推过去，这里只动
  // 浏览器侧的 <video> pause，不动 WebRTC 连接。
  avatar.cancel()
})

// 方案 A 音频权威源切换：
// - sidecar 启用 aligned 模式（推了 audio track）+ 数字人 live → remote 出声
// - 其它情况 → 本地 useAudioPlayback 出声（D 方案兼容）
//
// shouldUseRemoteAudio 同时驱动两件事：
// (a) avatar.attachVideo 内部把 <video> 的 muted 标志设为 false → 浏览器直接
//     播 sidecar 推过来的 audio track
// (b) audioPlayback.setRemoteAudioActive(true) → 本地解码出声 short-circuit,
//     避免与 remote 重复播放
//
// 监听放在 onAudio* 订阅之后，确保 audioPlayback / avatar 都已实例化。
watch(
  [() => avatar.hasRemoteAudio.value, () => avatar.state.value],
  ([hasAudio, state]) => {
    const next = hasAudio && state === 'live'
    if (shouldUseRemoteAudio.value !== next) {
      shouldUseRemoteAudio.value = next
    }
    // setRemoteAudioActive 内部幂等
    audioPlayback.setRemoteAudioActive(next)
  },
  { immediate: true },
)

const scrollRef = ref<HTMLElement | null>(null)
const chatColumnRef = ref<HTMLElement | null>(null)
const chatComposerRef = ref<InstanceType<typeof AgentHomeComposer> | null>(null)
/** 加载更早消息时避免自动滚到底部 */
const skipAutoScroll = ref(false)

type SubmitPayloadLike = { text?: string; attachmentUrl?: string | null; attachmentName?: string | null }
type InitialSubmit = SubmitPayloadLike
const pendingInitial = ref<InitialSubmit | null>(null)
/** 从首页仅点麦克风进入：WS 就绪后自动开始录音 */
const pendingStartMicOnReady = ref(false)

const buildSubmitText = (payload: SubmitPayloadLike | null): string => {
  if (!payload) return ''
  const text = String(payload.text || '').trim()
  const attachmentUrl = String(payload.attachmentUrl || '').trim()
  if (!attachmentUrl) return text
  const attachmentLine = `${t('agents.chat.attachmentPrefix')}${payload.attachmentName ? `${payload.attachmentName}: ` : ''}${attachmentUrl}`
  return text ? `${text}\n${attachmentLine}` : attachmentLine
}

const composerPlaceholder = computed(() => {
  switch (phase.value) {
    case 'ready':
      return t('agents.chat.askPlaceholder')
    case 'connecting':
    case 'opening':
      return t('agents.chat.connecting')
    case 'error':
      return t('agents.chat.connectErrorPlaceholder')
    case 'closed':
      return t('agents.session.closed')
    default:
      return t('agents.chat.cannotSend')
  }
})

const errorBanner = computed(() => {
  if (phase.value !== 'error') return ''
  return errorMessage.value || t('agents.chat.connectError')
})

const isAssistantThinkingMessage = (msg: ChatMessage) =>
  msg.role === 'assistant' &&
  msg.streaming &&
  !msg.isError &&
  !String(msg.text || '').trim() &&
  !(msg.artifacts?.length) &&
  !(msg.audioChunks?.length)

/** 语音 ASR 的 transcript 记录的是用户本人说的话（后端会以 role=user 落库），
 *  直播期间也应贴右侧呈现，避免与页面刷新后的历史回放产生左右错位。 */
const isUserSideMessage = (msg: ChatMessage) => msg.role === 'user' || msg.role === 'transcript'

/** 同一条 assistant 消息在流式期间会反复变长，缓存避免整段 MD 重复 parse。 */
const assistantMdHtmlCache = new Map<string, { text: string; html: string }>()

const clearAssistantMarkdownCache = () => {
  assistantMdHtmlCache.clear()
}

const assistantBubbleHtml = (msg: ChatMessage): string => {
  if (msg.role !== 'assistant') return ''
  const hit = assistantMdHtmlCache.get(msg.id)
  if (hit && hit.text === msg.text) return hit.html
  const html = renderAssistantMarkdown(msg.text)
  assistantMdHtmlCache.set(msg.id, { text: msg.text, html })
  return html
}

const assistantCopyTimers = new WeakMap<HTMLButtonElement, number>()

const onAssistantMdClick = (ev: MouseEvent) => {
  const btn = (ev.target as HTMLElement | null)?.closest?.('.agent-md-copy-btn') as HTMLButtonElement | null
  if (!btn) return
  ev.preventDefault()
  ev.stopPropagation()
  const wrap = btn.closest('.agent-md-copy-wrap')
  const slot = wrap?.querySelector('.agent-md-copy-slot')
  const block = slot?.querySelector('pre, blockquote') as HTMLElement | null
  if (!block) return
  const text = block.innerText
  void navigator.clipboard.writeText(text).then(() => {
    btn.classList.add('is-copied')
    const prev = assistantCopyTimers.get(btn)
    if (prev !== undefined) window.clearTimeout(prev)
    assistantCopyTimers.set(
      btn,
      window.setTimeout(() => {
        btn.classList.remove('is-copied')
        assistantCopyTimers.delete(btn)
      }, 2000),
    )
  }).catch(() => {
    showTopSnack(t('agents.chat.copyFailed'))
  })
}

const bubbleClass = (msg: ChatMessage) => {
  if (msg.role === 'user') return 'bg-blue-600/95 text-white'
  if (msg.isError) {
    return 'border border-red-200 bg-red-50 text-red-700 [.dark_&]:border-red-500/60 [.dark_&]:bg-red-950/40 [.dark_&]:text-red-100'
  }
  if (msg.role === 'transcript') {
    return msg.streaming
      ? 'bg-blue-600/70 text-white/90 ring-1 ring-blue-300/40'
      : 'bg-blue-600/95 text-white'
  }
  if (msg.role === 'tool') {
    return 'border border-amber-200 bg-amber-50 font-mono text-xs text-amber-900 [.dark_&]:border-amber-800/70 [.dark_&]:bg-gray-900/95 [.dark_&]:text-amber-100/95'
  }
  return 'border border-gray-200 bg-white text-gray-800 shadow-sm [.dark_&]:border-gray-700/80 [.dark_&]:bg-gray-800/90 [.dark_&]:text-gray-200'
}

const levelClass = (level: ActivityLevel) => {
  switch (level) {
    case 'success':
      return 'bg-emerald-50 text-emerald-700 [.dark_&]:bg-emerald-500/15 [.dark_&]:text-emerald-200/95'
    case 'warn':
      return 'bg-amber-50 text-amber-700 [.dark_&]:bg-amber-500/15 [.dark_&]:text-amber-200/95'
    case 'progress':
      return 'bg-sky-50 text-sky-700 [.dark_&]:bg-sky-500/15 [.dark_&]:text-sky-200/95'
    case 'error':
      return 'bg-red-50 text-red-700 [.dark_&]:bg-red-500/15 [.dark_&]:text-red-200/95'
    default:
      return 'bg-gray-100 text-gray-600 [.dark_&]:bg-gray-600/25 [.dark_&]:text-gray-300'
  }
}

const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
  const el = scrollRef.value
  if (!el) return
  el.scrollTo({ top: el.scrollHeight, behavior })
}

/** 距底部小于此像素时视为在看最新内容；否则用户可能在读历史，流式不抢滚动。 */
const SCROLL_STICK_THRESHOLD_PX = 80

const isListNearBottom = (): boolean => {
  const el = scrollRef.value
  if (!el) return true
  const slack = el.scrollHeight - el.scrollTop - el.clientHeight
  return slack <= SCROLL_STICK_THRESHOLD_PX
}

/** 流式 delta 每 ~50ms 一次，smooth 会互相排队导致滚动卡顿；用 rAF 节流 + 瞬时滚动。 */
let streamingScrollPending = false
const scheduleStreamingScroll = () => {
  if (skipAutoScroll.value) return
  if (streamingScrollPending) return
  streamingScrollPending = true
  requestAnimationFrame(() => {
    streamingScrollPending = false
    if (skipAutoScroll.value) return
    if (!isListNearBottom()) return
    scrollToBottom('auto')
  })
}

watch(
  () => messages.value.length,
  () => {
    if (skipAutoScroll.value) return
    nextTick(() => {
      if (!isListNearBottom()) return
      scrollToBottom('smooth')
    })
  },
)

// 只盯"最后一条消息"的长度变化来触发流式滚动，避免 map+join 整表字符串带来的 O(N) 开销。
watch(
  () => {
    const last = messages.value[messages.value.length - 1]
    return last ? last.text.length : 0
  },
  () => scheduleStreamingScroll(),
)

watch(recoverableToast, (t) => {
  if (t) showTopSnack(t)
})

watch(phase, (p) => {
  const init = pendingInitial.value
  const payloadText = buildSubmitText(init)
  if (p === 'ready' && payloadText) {
    sendUserMessage(payloadText)
    pendingInitial.value = null
    nextTick(() => scrollToBottom('smooth'))
  }
  if (p === 'closed' || p === 'error') {
    if (audioChat.micState.value === 'recording') {
      audioChat.stopRecord()
    }
  }
})

const onComposerSubmit = (p: AgentComposerSubmitPayload) => {
  const text = p.text.trim()
  if (!text && !p.attachmentFile) return
  if (p.attachmentFile && !p.attachmentRemoteUrl) return
  if (!canSubmit.value) return

  const payloadText = buildSubmitText({
    text,
    attachmentUrl: p.attachmentRemoteUrl,
    attachmentName: p.attachmentFile?.name ?? null,
  })
  if (!payloadText) return
  sendUserMessage(payloadText)
  chatComposerRef.value?.reset()
  nextTick(() => scrollToBottom('smooth'))
}

const onRetryWs = () => {
  void retryConnect()
}

const onLoadMoreHistory = async () => {
  const el = scrollRef.value
  const prevH = el?.scrollHeight ?? 0
  const prevTop = el?.scrollTop ?? 0
  skipAutoScroll.value = true
  try {
    await loadMoreHistory()
    await nextTick()
    if (el) {
      const delta = el.scrollHeight - prevH
      el.scrollTop = prevTop + delta
    }
    await nextTick()
  } finally {
    skipAutoScroll.value = false
  }
}

const bootstrap = async () => {
  const raw = route.params.id
  const id = Array.isArray(raw) ? raw[0] : raw
  if (!id) {
    pendingStartMicOnReady.value = false
    await router.replace({ name: 'Agents' })
    return
  }

  const st = window.history.state as { initialSubmit?: InitialSubmit | null; startMicOnReady?: boolean } | null
  const initial = (st?.initialSubmit ?? null) as InitialSubmit | null
  const startMicOnReady = Boolean(st?.startMicOnReady)
  try {
    window.history.replaceState(
      { ...window.history.state, initialSubmit: null, startMicOnReady: false },
      '',
    )
  } catch {
    /* ignore */
  }
  if (buildSubmitText(initial)) {
    pendingInitial.value = initial
  }
  if (startMicOnReady) {
    pendingStartMicOnReady.value = true
  }

  try {
    await connect(String(id))
  } catch (e) {
    pendingStartMicOnReady.value = false
    if (isSessionNotFoundError(e) || isSessionForbiddenError(e)) {
      removeRecentChatSession(String(id))
      showTopSnack(t('agents.session.notFoundOrForbidden'))
      await router.replace({ name: 'Agents' })
      return
    }
    showTopSnack(errorMessage.value || t('agents.session.connectFailed'))
  }
}

onMounted(() => {
  void bootstrap()
  nextTick(() => scrollToBottom())
})

watch(
  () => route.params.id,
  (nextId, prevId) => {
    const next = Array.isArray(nextId) ? nextId[0] : nextId
    const prev = Array.isArray(prevId) ? prevId[0] : prevId
    if (!next || next === prev) return
    clearAssistantMarkdownCache()
    pendingInitial.value = null
    pendingStartMicOnReady.value = false
    if (audioChat.micState.value === 'recording') {
      audioChat.stopRecord()
    }
    disposeUplinkForward()
    audioPlayback.cancel('session-switch')
    disconnect()
    void bootstrap()
  },
)

onBeforeRouteLeave(() => {
  if (audioChat.micState.value === 'recording') {
    audioChat.stopRecord()
  }
  disposeUplinkForward()
  audioPlayback.cancel('route-leave')
  stopAudioDeltaSub()
  stopAudioCancelSub()
  disconnect()
})

// ---------------------------------------------------------------------------
// P1 PR-4：麦克风录音 → audio_chunk / session_commit 上行
// ---------------------------------------------------------------------------
const audioChat = useAudioChat({ vad: { enabled: false } })

/** 正在录音（ready/live stream/hangover 期间持续为 true）。 */
const isRecording = computed(() => audioChat.micState.value === 'recording')
let autoBargeInCapture = false

watch(recoverableToast, (toast) => {
  if (!toast) return
  const normalized = toast.toLowerCase()
  const audioUnavailable = normalized.startsWith('model_not_available:')
  const audioBusy = normalized.startsWith('busy:') && normalized.includes('refuses audio_chunk')
  if ((!audioUnavailable && !audioBusy) || audioChat.micState.value !== 'recording') return
  audioChat.stopRecord()
  disposeUplinkForward()
})

/** 按钮在权限请求 / 正在停止 / 会话终态时禁用。 */
const micButtonDisabled = computed(() => {
  if (audioChat.micState.value === 'requesting') return true
  if (phase.value === 'closed' || phase.value === 'error') return true
  return false
})

let uplinkSeq = 0
let stopUplinkForward: (() => void) | null = null

// —— barge-in 上行诊断（每秒一行）：复测"TTS 生成阶段说 2-3 秒打不断"用 ——
// 帮助回答"是不是 echoGate 把人声大量丢了 / 还是其实人声都送上去了"。
const bargeInUplinkStats = {
  windowStartMs: 0,
  sent: 0,
  suppQuiet: 0,
  suppEcho: 0,
  reset(now: number) {
    this.windowStartMs = now
    this.sent = 0
    this.suppQuiet = 0
    this.suppEcho = 0
  },
  flushIfDue(now: number) {
    if (this.windowStartMs === 0) {
      this.reset(now)
      return
    }
    if (now - this.windowStartMs < 1000) return
    if (this.sent || this.suppQuiet || this.suppEcho) {
      console.info(
        '[barge-in uplink]',
        JSON.stringify({
          state: sessionState.value,
          isStreaming: isStreaming.value,
          playback: audioPlayback.state.value,
          sent: this.sent,
          suppQuiet: this.suppQuiet,
          suppEcho: this.suppEcho,
        }),
      )
    }
    this.reset(now)
  },
}

// 后端处于"输出态"（推理/工具/TTS 流/等任务）时本批 PCM 是 barge-in 探测，
// 必须经前端 echoGate 抑制 TTS 物理回声；否则就是用户 turn 输入，原样上行。
// 决策完全由 sessionState 驱动，不再用本地"是不是自动打开 mic"的近似。
const isSessionInOutputState = (state: string | null) =>
  state === 'reasoning' ||
  state === 'tool_running' ||
  state === 'streaming_output' ||
  state === 'waiting_task'

const ensureUplinkForwarder = () => {
  if (stopUplinkForward) return
  stopUplinkForward = audioChat.onUplinkFrame((frame) => {
    const inOutput = isSessionInOutputState(sessionState.value)
    const purpose = inOutput ? 'barge_in_probe' : 'user_turn'
    const now = performance.now()
    if (inOutput) {
      const gate = audioPlayback.echoGate(frame.pcm)
      if (gate.suppress) {
        if (gate.reason === 'echo') bargeInUplinkStats.suppEcho += 1
        else bargeInUplinkStats.suppQuiet += 1
        bargeInUplinkStats.flushIfDue(now)
        return
      }
    }
    const ok = sendAudioChunk(frame.pcm, uplinkSeq, 'audio/pcm;rate=16000', purpose)
    if (!ok) {
      if (audioChat.micState.value === 'recording') {
        audioChat.stopRecord()
      }
      showTopSnack(t('agents.chat.connectionLostStopRecording'))
      return
    }
    uplinkSeq += 1
    if (purpose === 'barge_in_probe') {
      bargeInUplinkStats.sent += 1
      bargeInUplinkStats.flushIfDue(now)
    }
  })
}

const disposeUplinkForward = () => {
  stopUplinkForward?.()
  stopUplinkForward = null
}

const beginMicCapture = async (options: { auto?: boolean; quiet?: boolean } = {}) => {
  if (micButtonDisabled.value) return
  if (!canStartAudioRecord.value) {
    if (!options.quiet) {
      showTopSnack(t('agents.chat.sessionNotReadyForAudio'))
    }
    return
  }
  uplinkSeq = 0
  ensureUplinkForwarder()
  try {
    await audioChat.startRecord()
    autoBargeInCapture = Boolean(options.auto)
  } catch (e) {
    disposeUplinkForward()
    autoBargeInCapture = false
    if (options.quiet) {
      console.warn('[audio] 自动监听启动失败:', e)
    } else {
      showTopSnack(e instanceof Error ? e.message : t('agents.chat.recordingStartFailed'))
    }
  }
}

const onMicToggle = async () => {
  if (isRecording.value) {
    const shouldCommit = !autoBargeInCapture
    autoBargeInCapture = false
    audioChat.stopRecord()
    if (shouldCommit) {
      commitAudio()
    }
    return
  }
  await beginMicCapture()
}

const onStopPlayback = () => {
  // 历史上"停止播放"只清本地 audioPlayback 队列，假设后端已经把 audio_delta
  // 全推完、剩下的只是浏览器侧 jitter buffer。但实际场景里：
  // 1) 后端 TTS 经常还在 streaming（audio_delta 仍在持续往下推）
  // 2) 数字人 aligned 模式下声音权威源在 WebRTC remote track，本地
  //    audioPlayback 已经被 setRemoteAudioActive(true) short-circuit 成 idle，
  //    cancel() 完全是 no-op（没有可清的本地排队）
  // 不向后端发 interrupt，TTS 继续推 → 数字人继续动嘴 → 远端 audio 继续响。
  // 统一改走 interrupt：本地清音频 + 后端中断 LLM/TTS/avatar 全链路。
  // 代价：一旦有"只剩本地 buffer"的 D 方案纯尾段场景也会走中断，但后端
  // 看到 interrupt 时已经没有可中断的 turn，是无副作用的幂等操作。
  interrupt()
}

watch(supportsAudioInput, (next, prev) => {
  if (!next || prev !== false) return
  if (phase.value !== 'ready') return
  if (sessionState.value !== 'streaming_output') return
  if (isStreaming.value) return
  interrupt()
})

// barge-in 自动监听：仅当 TTS 正在缓冲/播放时才打开麦克风听打断。
// 文字 turn 的 LLM 回复不走 TTS（master_runtime / voice_reply.should_stream_voice_reply
// 只在 turn.input_mode 为 audio_* 时启用 TTS），所以 isPlaybackActive 永远为 false，
// 麦克风不会被自动激活。语音 turn 在 TTS 第一帧到达后才有"可打断"的对象，于是开麦。
watch(
  () => [phase.value, isPlaybackActive.value, supportsAudioInput.value, audioChat.micState.value] as const,
  async ([ph, playing, audioInput, micState]) => {
    if (ph !== 'ready' || !audioInput || !playing) return
    if (micState !== 'idle') return
    await beginMicCapture({ auto: true, quiet: true })
  },
)

watch(
  () => [phase.value, pendingStartMicOnReady.value, canStartAudioRecord.value, supportsAudioInput.value] as const,
  async () => {
    if (!pendingStartMicOnReady.value) return
    const ph = phase.value
    if (ph === 'error' || ph === 'closed') {
      pendingStartMicOnReady.value = false
      return
    }
    if (ph !== 'ready') return
    if (!supportsAudioInput.value) {
      pendingStartMicOnReady.value = false
      showTopSnack(t('agents.chat.audioInputNotSupported'))
      return
    }
    if (!canStartAudioRecord.value) return
    pendingStartMicOnReady.value = false
    await beginMicCapture()
  },
)

// ---------------------------------------------------------------------------
// P1 PR-1/PR-2 dev-only：麦克风采集 + 能量 VAD 调试
// ---------------------------------------------------------------------------
const isDev = import.meta.env.DEV

const audioDebug = reactive({
  micState: audioChat.micState.value,
  frames: 0,
  uplinkFrames: 0,
  uplinkBytes: 0,
  droppedFrames: 0,
  segments: 0,
  speaking: false,
  vadEnabled: audioChat.vadOptions.value.enabled,
  rmsDisplay: '0.000',
  durationMs: 0,
  lastFrameMs: null as number | null,
  sampleRate: audioChat.targetSampleRate,
  lastVadEvent: '' as string,
  error: '' as string,
  busy: false,
})

watch(audioChat.micState, (s) => {
  audioDebug.micState = s
})
watch(audioChat.errorMessage, (m) => {
  audioDebug.error = m ?? ''
})
watch(audioChat.vadSpeaking, (b) => {
  audioDebug.speaking = b
})

let audioDebugUnsubPcm: (() => void) | null = null
let audioDebugUnsubUplink: (() => void) | null = null
let audioDebugUnsubVad: (() => void) | null = null
let audioDebugTimer: ReturnType<typeof setInterval> | null = null

const onAudioDebugPcmFrame = (frame: PcmFrame) => {
  audioDebug.frames += 1
  audioDebug.lastFrameMs = frame.frameMs
  audioDebug.sampleRate = frame.sampleRate
  if (frame.seq === 0) {
    console.log('[audio-debug] first frame:', {
      samples: frame.samples,
      bytes: frame.pcm.byteLength,
      sampleRate: frame.sampleRate,
      frameMs: frame.frameMs,
    })
  }
}

const onAudioDebugUplinkFrame = (frame: UplinkFrame) => {
  audioDebug.uplinkFrames += 1
  audioDebug.uplinkBytes += frame.pcm.byteLength
}

const onAudioDebugVadEvent = (ev: VadEvent) => {
  audioDebug.segments = audioChat.stats.segments
  audioDebug.droppedFrames = audioChat.stats.droppedFrames
  const tag =
    ev.type === 'speech-start'
      ? 'START'
      : ev.type === 'speech-end'
        ? `END ${ev.durationMs ?? 0}ms`
        : `DROP ${ev.framesBuffered ?? 0}f`
  audioDebug.lastVadEvent = `${tag} · ${ev.reason ?? ''}`.trim()
  console.log('[audio-debug] vad:', ev)
}

const onToggleVad = () => {
  const next = !audioDebug.vadEnabled
  audioChat.configureVad({ enabled: next })
  audioDebug.vadEnabled = next
}

const onAudioDebugToggle = async () => {
  if (audioDebug.busy) return
  audioDebug.busy = true
  try {
    if (audioChat.micState.value === 'recording') {
      audioChat.stopRecord()
      // 与正式麦克风按钮对齐：停止录音后必须 commit，触发后端 session.asr.commit
      // → ASR 产出 final transcript → 启动 LLM/TTS 全链路。
      commitAudio()
      if (audioDebugTimer) {
        clearInterval(audioDebugTimer)
        audioDebugTimer = null
      }
      audioDebugUnsubPcm?.()
      audioDebugUnsubUplink?.()
      audioDebugUnsubVad?.()
      audioDebugUnsubPcm = null
      audioDebugUnsubUplink = null
      audioDebugUnsubVad = null
      audioDebug.segments = audioChat.stats.segments
      audioDebug.droppedFrames = audioChat.stats.droppedFrames
      console.log('[audio-debug] stopped.', {
        total: audioDebug.frames,
        uplink: audioDebug.uplinkFrames,
        uplinkBytes: audioDebug.uplinkBytes,
        dropped: audioDebug.droppedFrames,
        segments: audioDebug.segments,
      })
    } else {
      if (!canStartAudioRecord.value) {
        showTopSnack(t('agents.chat.sessionNotReadyForAudio'))
        return
      }
      audioDebug.frames = 0
      audioDebug.uplinkFrames = 0
      audioDebug.uplinkBytes = 0
      audioDebug.droppedFrames = 0
      audioDebug.segments = 0
      audioDebug.durationMs = 0
      audioDebug.lastFrameMs = null
      audioDebug.lastVadEvent = ''
      audioDebug.error = ''
      audioDebugUnsubPcm = audioChat.onPcmFrame(onAudioDebugPcmFrame)
      audioDebugUnsubUplink = audioChat.onUplinkFrame(onAudioDebugUplinkFrame)
      audioDebugUnsubVad = audioChat.onVadEvent(onAudioDebugVadEvent)
      // 与正式麦克风按钮对齐：注册 uplink→sendAudioChunk 转发，否则音频只在本地统计，
      // 不会真正上行到后端。
      uplinkSeq = 0
      ensureUplinkForwarder()
      try {
        await audioChat.startRecord()
      } catch (e) {
        disposeUplinkForward()
        audioDebugUnsubPcm?.()
        audioDebugUnsubUplink?.()
        audioDebugUnsubVad?.()
        audioDebugUnsubPcm = null
        audioDebugUnsubUplink = null
        audioDebugUnsubVad = null
        throw e
      }
      const startedAt = audioChat.recordStartedAt.value ?? performance.now()
      audioDebugTimer = setInterval(() => {
        audioDebug.durationMs = Math.round(performance.now() - startedAt)
        audioDebug.rmsDisplay = audioChat.stats.lastRmsNorm.toFixed(4)
      }, 150)
    }
  } catch (e) {
    console.error('[audio-debug] toggle error:', e)
    showTopSnack(e instanceof Error ? e.message : t('agents.chat.recordingOpFailed'))
  } finally {
    audioDebug.busy = false
  }
}

// ---------------------------------------------------------------------------
// P1 PR-3 dev-only：TTS 播放调试
// ---------------------------------------------------------------------------
const audioPlaybackDebug = reactive({
  state: audioPlayback.state.value as 'idle' | 'buffering' | 'playing' | 'error',
  enqueued: 0,
  played: 0,
  cancelled: 0,
  queueLength: 0,
  lastSampleRate: 0,
  lastChunkMs: 0,
  error: '' as string,
})

watch(audioPlayback.state, (s) => {
  audioPlaybackDebug.state = s
})
watch(audioPlayback.errorMessage, (m) => {
  audioPlaybackDebug.error = m ?? ''
})

const playbackStatsTimer: ReturnType<typeof setInterval> = setInterval(() => {
  audioPlaybackDebug.enqueued = audioPlayback.stats.enqueued
  audioPlaybackDebug.played = audioPlayback.stats.played
  audioPlaybackDebug.cancelled = audioPlayback.stats.cancelled
  audioPlaybackDebug.queueLength = audioPlayback.stats.queueLength
  audioPlaybackDebug.lastSampleRate = audioPlayback.stats.lastSampleRate
  audioPlaybackDebug.lastChunkMs = audioPlayback.stats.lastChunkMs
}, 200)

const onPlaybackCancel = () => {
  audioPlayback.cancel('dev-manual')
}

// Dev 调试面板当前在模板中注释保留；显式引用避免生产类型检查误判为未使用。
void isDev
void onToggleVad
void onAudioDebugToggle
void onPlaybackCancel

onBeforeUnmount(() => {
  clearAssistantMarkdownCache()
  clearInterval(playbackStatsTimer)
  disposeUplinkForward()
  audioChat.teardown()
  audioPlayback.teardown()
})
</script>

<style scoped>
.agent-chat {
  --agent-chat-rail: min(20rem, calc(100vw - 18rem));
  --agent-chat-bubble-max-width: 76ch;
}

.agent-chat :where(button:not(:disabled)) {
  cursor: pointer;
}

.agent-chat-scroll,
.event-viewport {
  scrollbar-gutter: stable;
}

.event-viewport {
  scrollbar-width: thin;
}

.agent-chat-scroll::-webkit-scrollbar {
  width: 8px;
}
.event-viewport::-webkit-scrollbar {
  width: 6px;
}
.agent-chat-scroll::-webkit-scrollbar-track,
.event-viewport::-webkit-scrollbar-track {
  background: rgba(229, 231, 235, 0.75);
  border-radius: 999px;
}
.agent-chat-scroll::-webkit-scrollbar-thumb,
.event-viewport::-webkit-scrollbar-thumb {
  background: rgba(156, 163, 175, 0.8);
  border-radius: 999px;
}
</style>

<style>
html.dark .agent-chat-scroll::-webkit-scrollbar-track,
html.dark .event-viewport::-webkit-scrollbar-track {
  background: rgba(17, 24, 39, 0.5);
}

html.dark .agent-chat-scroll::-webkit-scrollbar-thumb,
html.dark .event-viewport::-webkit-scrollbar-thumb {
  background: rgba(75, 85, 99, 0.85);
}

.agent-md pre code .hljs-keyword,
.agent-md pre code .hljs-selector-tag,
.agent-md pre code .hljs-deletion {
  color: #ff7b72;
}
.agent-md pre code .hljs-string,
.agent-md pre code .hljs-attr,
.agent-md pre code .hljs-addition,
.agent-md pre code .hljs-regexp,
.agent-md pre code .hljs-link {
  color: #a5d6ff;
}
.agent-md pre code .hljs-number,
.agent-md pre code .hljs-literal {
  color: #79c0ff;
}
.agent-md pre code .hljs-comment,
.agent-md pre code .hljs-meta,
.agent-md pre code .hljs-doctag {
  color: #8b949e;
}
.agent-md pre code .hljs-title,
.agent-md pre code .hljs-section,
.agent-md pre code .hljs-function,
.agent-md pre code .hljs-name {
  color: #d2a8ff;
}
.agent-md pre code .hljs-built_in,
.agent-md pre code .hljs-type,
.agent-md pre code .hljs-variable,
.agent-md pre code .hljs-template-variable {
  color: #ffa657;
}
</style>
