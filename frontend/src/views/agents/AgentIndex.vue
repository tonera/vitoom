<template>
  <div
    ref="agentRootRef"
    class="agent-home vt-surface vt-text-smooth relative flex h-full min-h-0 w-full min-w-0 flex-col overflow-hidden antialiased text-gray-950 [.dark_&]:text-gray-100"
  >
    <AgentSessionControls @open-history="historyOpen = !historyOpen" @new-chat="startNewChat" />

    <!-- 历史会话：从按钮右缘向右展开宽度，避免整片 translate 扫过左侧按钮 -->
    <Transition name="agent-slide">
      <aside
        v-if="historyOpen"
        class="agent-history-panel pointer-events-auto absolute bottom-0 left-[4.25rem] top-4 z-[15] overflow-hidden sm:left-[4.5rem] sm:top-5"
        role="region"
        :aria-label="t('agents.session.history')"
      >
        <div
          class="flex h-full min-h-0 w-[min(24vw,20rem)] min-w-[12rem] flex-col overflow-hidden rounded-r-2xl border border-gray-200 bg-white/95 shadow-lg ring-1 ring-gray-100 backdrop-blur-md [.dark_&]:border-gray-700/80 [.dark_&]:bg-gray-900/95 [.dark_&]:ring-black/20"
        >
          <div
            class="flex items-center gap-2 border-b border-gray-200 px-2 py-2 [.dark_&]:border-gray-800/80"
          >
            <input
              v-model="historySearchQuery"
              type="search"
              enterkeyhint="search"
              autocomplete="off"
              :aria-label="t('agents.session.searchTitleAria')"
              :placeholder="t('agents.session.searchPlaceholder')"
              class="min-w-0 flex-1 rounded-lg border border-gray-200 bg-white px-2.5 py-1.5 text-xs text-gray-900 outline-none placeholder:text-gray-400 focus:border-sky-500/55 focus:ring-2 focus:ring-sky-500/20 [.dark_&]:border-gray-600 [.dark_&]:bg-gray-950/60 [.dark_&]:text-gray-100 [.dark_&]:placeholder:text-gray-500 [.dark_&]:focus:border-sky-400/55 [.dark_&]:focus:ring-sky-400/15"
            />
            <button
              type="button"
              class="shrink-0 rounded-full p-2 text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-950 [.dark_&]:text-gray-400 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-gray-100"
              :aria-label="t('agents.session.collapse')"
              @click="historyOpen = false"
            >
              <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 18l-6-6 6-6" />
              </svg>
            </button>
          </div>
          <div
            ref="historyScrollRef"
            class="agent-history-scroll flex-1 overflow-y-auto overflow-x-hidden p-2"
            @scroll.passive="onHistoryScroll"
          >
            <p
              v-if="historyLoading && !historySessionsAll.length"
              class="px-2 py-6 text-center text-xs text-gray-500 sm:px-3 sm:text-sm [.dark_&]:text-gray-400"
            >
              {{ t('common.loading') }}
            </p>
            <p
              v-else-if="!historySessionsAll.length && historySearchQuery.trim()"
              class="px-2 py-6 text-center text-xs text-gray-500 sm:px-3 sm:text-sm [.dark_&]:text-gray-400"
            >
              {{ t('agents.session.noSearchResults') }}
            </p>
            <p
              v-else-if="!historySessionsAll.length"
              class="px-2 py-6 text-center text-xs text-gray-500 sm:px-3 sm:text-sm [.dark_&]:text-gray-400"
            >
              {{ t('agents.session.noHistory') }}
            </p>
            <ul v-else class="space-y-0.5">
              <li v-for="row in historySessionsAll" :key="row.id">
                <button
                  type="button"
                  class="flex w-full min-w-0 flex-col rounded-lg px-2.5 py-2 text-left text-xs transition-colors hover:bg-gray-100 sm:px-3 sm:py-2.5 sm:text-sm [.dark_&]:hover:bg-gray-800/80"
                  @click="openSession(row.id)"
                >
                  <span
                    class="line-clamp-2 break-words text-gray-800 [overflow-wrap:anywhere] [.dark_&]:text-gray-200"
                  >
                    {{ row.title || t('agents.session.newSession') }}
                  </span>
                  <span class="mt-0.5 text-[10px] text-gray-500 sm:text-xs [.dark_&]:text-gray-400">
                    {{ formatSessionTime(row.updatedAt) }}
                  </span>
                </button>
              </li>
            </ul>
            <p
              v-if="historyLoading && historySessionsAll.length && historyHasMore"
              class="py-2 text-center text-[10px] text-gray-400 [.dark_&]:text-gray-500"
            >
              {{ t('common.loading') }}
            </p>
          </div>
        </div>
      </aside>
    </Transition>

    <div class="flex min-h-0 flex-1 flex-col items-center justify-center overflow-hidden px-4 pb-24 pt-16 sm:px-8">
      <h1
        class="mb-10 max-w-3xl text-center text-[1.65rem] font-normal leading-snug tracking-tight text-gray-950 sm:text-4xl [.dark_&]:text-gray-100"
      >
        {{ t('agents.greeting', { name: greetingName }) }}
      </h1>

      <AgentHomeComposer
        ref="homeComposerRef"
        :mic-enabled="true"
        :boundary-element="agentRootRef"
        :submit-disabled="sessionCreating"
        @submit="onComposerSubmit"
        @mic-toggle="onHomeMicToggle"
      />
    </div>

  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { watchDebounced } from '@vueuse/core'
import {
  createChatSession,
  listChatSessions,
  pushRecentChatSession,
  type ChatSession,
  type ChatSessionAudioOutput,
  type RecentChatSessionRow,
} from '../../api/chat'
import { showTopSnack } from '../../composables/useTopSnack'
import { get, handleApiError } from '../../utils/api'
import type { AgentComposerSubmitPayload } from './agentComposerTypes'
import AgentHomeComposer from './components/AgentHomeComposer.vue'
import AgentSessionControls from './components/AgentSessionControls.vue'

const { t, locale } = useI18n()
const router = useRouter()

const defaultSessionTitle = () => t('agents.session.newSession')

/**
 * P1 默认 TTS 偏好（硬编码；后续可改成用户设置）。
 * - `tts_mode=custom_voice` 仅表达启用语音回复；默认 speaker 由共享
 *   `config/tts_speakers.json` 统一决定。
 * - `output_mode=multimodal_result` 让后端 `_is_audio_output_mode` 为 true，从而启用 TTS 通道；
 *   `input_mode=mixed` 用来预分派 ASR（P1 PR-4 已确认）。
 * - 真正的 TTS 模型选择由后端 audio_output.load_name 为空时的 `fixed_model` 兜底决定。
 */
const DEFAULT_VOICE_REPLY: ChatSessionAudioOutput = {
  tts_mode: 'custom_voice',
}

/** 与聊天页侧栏一致；列表数据来自 GET /v1/chat/sessions */
const HISTORY_PAGE_SIZE = 15

const historyOpen = ref(false)
const historySessionsAll = ref<RecentChatSessionRow[]>([])
const historyHasMore = ref(true)
const historyLoading = ref(false)
const historyScrollRef = ref<HTMLElement | null>(null)
let historyScrollLoadTid: number | undefined

function chatSessionToRow(s: ChatSession): RecentChatSessionRow {
  const raw = s.updated_at || s.created_at || ''
  const ts = raw ? Date.parse(raw) : NaN
  return {
    id: s.id,
    title: String(s.title || '').trim() || defaultSessionTitle(),
    updatedAt: Number.isFinite(ts) ? ts : Date.now(),
  }
}

const historySearchQuery = ref('')

function historyListQuery(): string | undefined {
  const s = historySearchQuery.value.trim()
  return s ? s : undefined
}

const fetchHistoryPage = async (append: boolean) => {
  if (historyLoading.value) return
  if (append && !historyHasMore.value) return
  historyLoading.value = true
  try {
    const offset = append ? historySessionsAll.value.length : 0
    const { items, count } = await listChatSessions({
      limit: HISTORY_PAGE_SIZE,
      offset,
      q: historyListQuery(),
    })
    const rows = items.map(chatSessionToRow)
    if (append) {
      const existing = new Set(historySessionsAll.value.map((r) => r.id))
      const merged = [...historySessionsAll.value]
      for (const r of rows) {
        if (!existing.has(r.id)) {
          existing.add(r.id)
          merged.push(r)
        }
      }
      historySessionsAll.value = merged
    } else {
      historySessionsAll.value = rows
    }
    historyHasMore.value = count >= HISTORY_PAGE_SIZE
  } catch (e) {
    showTopSnack(handleApiError(e).message || t('agents.session.loadHistoryFailed'))
    if (!append) {
      historySessionsAll.value = []
      historyHasMore.value = false
    }
  } finally {
    historyLoading.value = false
  }
}

/** 触底拉取服务端下一页 */
const onHistoryScroll = (ev: Event) => {
  const el = ev.target as HTMLElement | null
  if (!el) return
  const nearBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 56
  if (!nearBottom || !historyHasMore.value || historyLoading.value) return
  if (historyScrollLoadTid) clearTimeout(historyScrollLoadTid)
  historyScrollLoadTid = window.setTimeout(() => {
    historyScrollLoadTid = undefined
    void fetchHistoryPage(true)
  }, 80)
}

watch(historyOpen, (open) => {
  if (open) {
    historyHasMore.value = true
    void nextTick(() => {
      const el = historyScrollRef.value
      if (el) el.scrollTop = 0
      void fetchHistoryPage(false)
    })
  }
})

watchDebounced(
  historySearchQuery,
  () => {
    if (!historyOpen.value) return
    historyHasMore.value = true
    void nextTick(() => {
      const el = historyScrollRef.value
      if (el) el.scrollTop = 0
      void fetchHistoryPage(false)
    })
  },
  { debounce: 320 },
)
const homeComposerRef = ref<InstanceType<typeof AgentHomeComposer> | null>(null)
const sessionCreating = ref(false)

const userInfo = ref<{ nickname?: string; email?: string } | null>(null)

const greetingName = computed(() => {
  const n = userInfo.value?.nickname?.trim()
  if (n) return n.split(/\s+/)[0] || n
  return t('agents.session.friend')
})

const fetchUser = async () => {
  try {
    userInfo.value = await get<{ nickname?: string; email?: string }>('/auth/me')
  } catch {
    userInfo.value = null
  }
}

const agentRootRef = ref<HTMLElement | null>(null)

const formatSessionTime = (ts: number) => {
  const d = new Date(ts)
  if (Number.isNaN(d.getTime())) return ''
  return d.toLocaleString(locale.value, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

const openSession = (sessionId: string) => {
  historyOpen.value = false
  router.push({ name: 'AgentChat', params: { id: sessionId } })
}

/** 首页仅点麦克风：先建会话再进聊天页，由 AgentChat 在 WS ready 后自动开录，避免首页无上行通道却「假录音」。 */
const onHomeMicToggle = async () => {
  if (sessionCreating.value) return
  sessionCreating.value = true
  try {
    const session = await createChatSession({
      input_mode: 'mixed',
      output_mode: 'multimodal_result',
      audio_output: DEFAULT_VOICE_REPLY,
    })
    pushRecentChatSession({
      id: session.id,
      title: String(session.title || '').trim() || defaultSessionTitle(),
      updatedAt: Date.now(),
    })
    await router.push({
      name: 'AgentChat',
      params: { id: session.id },
      state: { startMicOnReady: true },
    })
  } catch (e) {
    showTopSnack(handleApiError(e).message || t('agents.session.createFailed'))
  } finally {
    sessionCreating.value = false
  }
}

const startNewChat = async () => {
  if (sessionCreating.value) return
  sessionCreating.value = true
  try {
    const session = await createChatSession({
      input_mode: 'mixed',
      output_mode: 'multimodal_result',
      audio_output: DEFAULT_VOICE_REPLY,
    })
    pushRecentChatSession({
      id: session.id,
      title: String(session.title || '').trim() || defaultSessionTitle(),
      updatedAt: Date.now(),
    })
    await router.push({ name: 'AgentChat', params: { id: session.id } })
  } catch (e) {
    showTopSnack(handleApiError(e).message || t('agents.session.createFailed'))
  } finally {
    sessionCreating.value = false
  }
}

const onComposerSubmit = async (p: AgentComposerSubmitPayload) => {
  const text = p.text.trim()
  if (!text && !p.attachmentFile) return
  if (p.attachmentFile && !p.attachmentRemoteUrl) return
  if (sessionCreating.value) return
  const draftTitle = text
    ? text.slice(0, 80)
    : String(p.attachmentFile?.name || '').trim().slice(0, 80) || undefined

  sessionCreating.value = true
  try {
    const session = await createChatSession({
      input_mode: 'mixed',
      output_mode: 'multimodal_result',
      title: draftTitle,
      audio_output: DEFAULT_VOICE_REPLY,
    })
    pushRecentChatSession({
      id: session.id,
      title: draftTitle || String(session.title || '').trim() || defaultSessionTitle(),
      updatedAt: Date.now(),
    })
    await router.push({
      name: 'AgentChat',
      params: { id: session.id },
      state: {
        initialSubmit: {
          text,
          attachmentUrl: p.attachmentRemoteUrl,
          attachmentName: p.attachmentFile?.name ?? null,
        },
      },
    })
    homeComposerRef.value?.reset()
  } catch (e) {
    showTopSnack(handleApiError(e).message || t('agents.session.createFailed'))
  } finally {
    sessionCreating.value = false
  }
}

void fetchUser()
</script>

<style scoped>
.agent-home {
  font-family: inherit;
}

.agent-home :where(button) {
  cursor: pointer;
}

.agent-history-panel {
  width: min(24vw, 20rem);
  min-width: 12rem;
}

.agent-slide-enter-active.agent-history-panel,
.agent-slide-leave-active.agent-history-panel {
  transition:
    width 0.3s cubic-bezier(0.22, 1, 0.36, 1),
    min-width 0.3s cubic-bezier(0.22, 1, 0.36, 1);
}

.agent-slide-enter-from.agent-history-panel,
.agent-slide-leave-to.agent-history-panel {
  width: 0 !important;
  min-width: 0 !important;
}
</style>

<style>
.agent-history-panel :where(button) {
  cursor: pointer;
}

/* 历史会话列表滚动条（与 html.dark 主题切换一致，参考 AgentChat 卡片语感） */
.agent-history-scroll {
  scrollbar-width: thin;
  scrollbar-color: rgba(100, 116, 139, 0.42) transparent;
}
.agent-history-scroll::-webkit-scrollbar {
  width: 6px;
}
.agent-history-scroll::-webkit-scrollbar-track {
  margin: 4px 0;
  background: transparent;
}
.agent-history-scroll::-webkit-scrollbar-thumb {
  border-radius: 9999px;
  background-color: rgba(100, 116, 139, 0.32);
  border: 2px solid transparent;
  background-clip: padding-box;
}
.agent-history-scroll::-webkit-scrollbar-thumb:hover {
  background-color: rgba(71, 85, 105, 0.5);
}
html.dark .agent-history-scroll {
  scrollbar-color: rgba(148, 163, 184, 0.45) transparent;
}
html.dark .agent-history-scroll::-webkit-scrollbar-thumb {
  background-color: rgba(148, 163, 184, 0.32);
}
html.dark .agent-history-scroll::-webkit-scrollbar-thumb:hover {
  background-color: rgba(203, 213, 225, 0.42);
}
</style>
