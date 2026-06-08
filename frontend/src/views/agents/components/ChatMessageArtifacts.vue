<template>
  <div
    class="chat-artifacts mt-3 flex flex-col gap-2"
    role="region"
    :aria-label="t('agents.artifacts.sectionLabel')"
  >
    <p class="text-[11px] font-medium uppercase tracking-wide text-gray-500 [.dark_&]:text-gray-400">{{ t('agents.artifacts.attachments') }}</p>
    <div class="flex flex-col gap-2">
      <div
        v-if="imageItems.length"
        class="group relative overflow-hidden rounded-xl border border-gray-200 bg-linear-to-br from-white to-gray-100/80 shadow-sm ring-1 ring-gray-200/60 transition hover:border-gray-300 hover:ring-sky-500/20 [.dark_&]:border-gray-600/50 [.dark_&]:from-gray-900/90 [.dark_&]:to-gray-950/95 [.dark_&]:ring-white/4 [.dark_&]:hover:border-gray-500/60 [.dark_&]:hover:ring-sky-500/15"
      >
        <div class="overflow-x-auto px-3 py-3 sm:px-3.5 sm:py-3.5">
          <div class="flex min-w-max flex-nowrap items-start gap-3">
            <div
              v-for="art in imageItems"
              :key="art.fileId"
              class="w-44 shrink-0"
            >
              <a
                :href="art.url"
                target="_blank"
                rel="noopener noreferrer"
                class="block outline-none focus-visible:ring-2 focus-visible:ring-sky-500/70"
                :title="t('agents.artifacts.openInNewTab', { name: displayName(art) })"
              >
                <div
                  class="relative overflow-hidden rounded-xl bg-gray-200/30 p-2 ring-1 ring-gray-200/80 [.dark_&]:bg-black/35 [.dark_&]:ring-white/5"
                >
                  <img
                    :src="art.url"
                    :alt="displayName(art)"
                    class="block h-44 w-full rounded-lg object-cover object-left"
                    loading="lazy"
                    decoding="async"
                  />
                  <div
                    class="pointer-events-none absolute inset-x-0 bottom-0 bg-linear-to-t from-black/70 to-transparent px-3 pb-2 pt-8"
                  >
                    <span class="line-clamp-1 text-xs font-medium text-white/95">{{ displayName(art) }}</span>
                  </div>
                </div>
              </a>
              <ChatMessageArtifactActions :url="art.url" />
            </div>
          </div>
        </div>
      </div>

      <div
        v-for="art in nonImageItems"
        :key="art.fileId"
        class="group relative overflow-hidden rounded-xl border border-gray-200 bg-linear-to-br from-white to-gray-100/80 shadow-sm ring-1 ring-gray-200/60 transition hover:border-gray-300 hover:ring-sky-500/20 [.dark_&]:border-gray-600/50 [.dark_&]:from-gray-900/90 [.dark_&]:to-gray-950/95 [.dark_&]:ring-white/4 [.dark_&]:hover:border-gray-500/60 [.dark_&]:hover:ring-sky-500/15"
      >
        <!-- 音频 -->
        <template v-if="isAudio(art) && (art.url || isPending(art))">
          <div class="px-3 pb-1 pt-3 sm:px-3.5">
            <div class="flex items-start gap-3">
              <div
                class="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-violet-100 text-violet-700 ring-1 ring-violet-200/80 [.dark_&]:bg-violet-500/15 [.dark_&]:text-violet-300 [.dark_&]:ring-violet-400/25"
                aria-hidden="true"
              >
                <svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <path stroke-linecap="round" d="M12 3v18M8 7v10M16 7v10M4 10v4M20 10v4" />
                </svg>
              </div>
              <div class="min-w-0 flex-1">
                <p
                  class="truncate text-sm font-medium text-gray-900 [.dark_&]:text-gray-100"
                  :title="displayName(art)"
                >
                  {{ displayName(art) }}
                </p>
                <p class="mt-0.5 text-[11px] text-gray-500 [.dark_&]:text-gray-500">
                  {{ isPending(art) ? t('agents.artifacts.audioGenerating') : t('agents.artifacts.audio') }}
                  <span v-if="art.size != null"> · {{ formatBytes(art.size) }}</span>
                  <span v-if="isPending(art) && art.progress != null"> · {{ art.progress }}%</span>
                </p>
                <audio
                  v-if="art.url"
                  :src="art.url"
                  controls
                  preload="metadata"
                  class="mt-2.5 h-9 w-full max-w-full rounded-lg border border-gray-200 bg-gray-50 opacity-95 [&::-webkit-media-controls-panel]:bg-gray-200 [.dark_&]:border-gray-700/80 [.dark_&]:bg-gray-900/40 [.dark_&]:[&::-webkit-media-controls-panel]:bg-gray-800"
                />
                <div
                  v-else
                  class="mt-2.5 flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700 [.dark_&]:border-amber-900/60 [.dark_&]:bg-amber-950/30 [.dark_&]:text-amber-300"
                >
                  <span class="inline-block h-2 w-2 animate-pulse rounded-full bg-amber-500" />
                  {{ t('agents.artifacts.audioSynthesizing') }}
                </div>
              </div>
            </div>
          </div>
          <ChatMessageArtifactActions v-if="art.url" :url="art.url" />
        </template>

        <!-- 视频 -->
        <template v-else-if="isVideo(art) && art.url">
          <div class="px-3 pb-1 pt-3 sm:px-3.5">
            <div class="flex items-start gap-3">
              <div
                class="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-rose-100 text-rose-700 ring-1 ring-rose-200/80 [.dark_&]:bg-rose-500/15 [.dark_&]:text-rose-300 [.dark_&]:ring-rose-400/25"
                aria-hidden="true"
              >
                <svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75">
                  <rect x="2" y="4" width="20" height="16" rx="2" />
                  <path stroke-linecap="round" d="M7 8v8M17 8v8" />
                </svg>
              </div>
              <div class="min-w-0 flex-1">
                <p
                  class="truncate text-sm font-medium text-gray-900 [.dark_&]:text-gray-100"
                  :title="displayName(art)"
                >
                  {{ displayName(art) }}
                </p>
                <p class="mt-0.5 text-[11px] text-gray-500 [.dark_&]:text-gray-500">
                  {{ t('agents.artifacts.video') }}
                  <span v-if="art.size != null"> · {{ formatBytes(art.size) }}</span>
                </p>
                <video
                  :src="art.url"
                  controls
                  playsinline
                  preload="metadata"
                  class="mt-2.5 max-h-52 w-full rounded-lg border border-gray-200 bg-black [.dark_&]:border-gray-700/80"
                />
              </div>
            </div>
          </div>
          <ChatMessageArtifactActions :url="art.url" />
        </template>

        <!-- 通用文件 -->
        <template v-else>
          <div class="flex items-center gap-3 px-3 py-2.5 sm:px-3.5">
            <div
              class="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-sky-100 text-sky-700 ring-1 ring-sky-200/80 [.dark_&]:bg-sky-500/12 [.dark_&]:text-sky-300 [.dark_&]:ring-sky-400/20"
              aria-hidden="true"
            >
              <svg class="h-5 w-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75">
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6z"
                />
                <path stroke-linecap="round" stroke-linejoin="round" d="M14 2v6h6M16 13H8M16 17H8M10 9H8" />
              </svg>
            </div>
            <div class="min-w-0 flex-1">
              <p
                class="truncate text-sm font-medium text-gray-900 [.dark_&]:text-gray-100"
                :title="displayName(art)"
              >
                {{ displayName(art) }}
              </p>
              <p class="mt-0.5 text-[11px] text-gray-500 [.dark_&]:text-gray-500">
                {{ formatKindLabel(art) }}
                <span v-if="art.size != null"> · {{ formatBytes(art.size) }}</span>
              </p>
            </div>
          </div>
          <ChatMessageArtifactActions v-if="art.url" :url="art.url" />
          <div
            v-else
            class="border-t border-gray-200 px-3 py-2 text-[11px] text-gray-500 [.dark_&]:border-gray-700/50 sm:px-3.5"
          >
            {{ art.name || art.fileId || t('agents.artifacts.noLink') }}
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { ChatArtifact } from '../../../composables/useAgentChatSession'
import ChatMessageArtifactActions from './ChatMessageArtifactActions.vue'

const { t } = useI18n()

const props = defineProps<{
  items: ChatArtifact[]
}>()

const imageItems = computed(() => props.items.filter((art) => isImage(art) && !!art.url))
const nonImageItems = computed(() => props.items.filter((art) => !(isImage(art) && !!art.url)))

function displayName(art: ChatArtifact): string {
  const n = (art.name || '').trim()
  if (n) return n
  try {
    const u = new URL(art.url, typeof window !== 'undefined' ? window.location.origin : 'http://localhost')
    const seg = u.pathname.split('/').filter(Boolean).pop()
    if (seg) return decodeURIComponent(seg)
  } catch {
    /* ignore */
  }
  return art.fileId || t('agents.artifacts.file')
}

function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n < 0) return ''
  const units = ['B', 'KB', 'MB', 'GB']
  let v = n
  let i = 0
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024
    i += 1
  }
  const digits = i === 0 ? 0 : v < 10 ? 1 : v < 100 ? 1 : 0
  return `${v.toFixed(digits)} ${units[i]}`
}

function isImage(art: ChatArtifact): boolean {
  return art.category === 'image' || art.mime.startsWith('image/')
}

function isAudio(art: ChatArtifact): boolean {
  return art.category === 'audio' || art.mime.startsWith('audio/')
}

function isVideo(art: ChatArtifact): boolean {
  return art.category === 'video' || art.mime.startsWith('video/')
}

function isPending(art: ChatArtifact): boolean {
  return art.status === 'pending'
}

function formatKindLabel(art: ChatArtifact): string {
  if (isAudio(art)) return t('agents.artifacts.audio')
  if (isVideo(art)) return t('agents.artifacts.video')
  if (isImage(art)) return t('agents.artifacts.image')
  const name = displayName(art)
  const ext = name.split('.').pop()
  if (ext && ext.length <= 8 && ext !== name) return t('agents.artifacts.fileWithExt', { ext: ext.toUpperCase() })
  return t('agents.artifacts.file')
}
</script>
