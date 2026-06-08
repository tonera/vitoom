<template>
  <div class="flex h-full min-h-0 flex-col overflow-hidden bg-gray-50 text-gray-950 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100">
    <div class="shrink-0 border-b border-gray-200 bg-white px-5 py-4 [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
      <div class="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
        <div>
          <h1 class="text-xl font-semibold">{{ t('assets.title') }}</h1>
          <p class="mt-1 text-sm text-gray-500 [.dark_&]:text-gray-400">
            {{ t('assets.subtitle') }}
          </p>
        </div>

        <div class="flex flex-col gap-3 lg:flex-row lg:items-center">
          <div class="flex flex-wrap gap-2" role="tablist" :aria-label="t('assets.assetTypeAriaLabel')">
            <button
              v-for="tab in tabs"
              :key="tab.value"
              type="button"
              class="rounded-full px-4 py-2 text-sm font-medium transition-colors cursor-pointer"
              :class="activeTab === tab.value ? activeTabClass : inactiveTabClass"
              role="tab"
              :aria-selected="activeTab === tab.value"
              @click="setActiveTab(tab.value)"
            >
              {{ tab.label }}
            </button>
          </div>

          <div class="flex min-w-0 items-center gap-2">
            <input
              v-model="searchKeyword"
              type="text"
              class="min-w-0 flex-1 rounded-full border border-gray-200 bg-white px-4 py-2 text-sm outline-none transition focus:border-gray-400 lg:w-64 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:focus:border-gray-500"
              :placeholder="t('assets.searchPlaceholder')"
              @keydown.enter.prevent="submitSearch"
            >
            <button
              type="button"
              class="rounded-full bg-gray-950 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-60 [.dark_&]:bg-white [.dark_&]:text-gray-950 [.dark_&]:hover:bg-gray-200 cursor-pointer"
              :disabled="loading"
              @click="submitSearch"
            >
              {{ t('common.search') }}
            </button>
            <button
              type="button"
              class="rounded-full border border-gray-200 px-4 py-2 text-sm font-medium text-gray-700 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-60 [.dark_&]:border-gray-700 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800 cursor-pointer"
              :disabled="loading"
              @click="loadAssets"
            >
              {{ t('common.refresh') }}
            </button>
          </div>
        </div>
      </div>
    </div>

    <div class="assets-main-scroll min-h-0 flex-1 overflow-y-auto p-5">
      <div v-if="loading" class="flex min-h-[360px] items-center justify-center rounded-2xl border border-gray-200 bg-white text-sm text-gray-500 [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900 [.dark_&]:text-gray-400">
        {{ t('assets.loading') }}
      </div>

      <div v-else-if="items.length === 0" class="flex min-h-[360px] flex-col items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-white px-6 text-center [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900">
        <p class="text-base font-semibold text-gray-900 [.dark_&]:text-white">{{ t('assets.emptyTitle') }}</p>
        <p class="mt-2 max-w-md text-sm text-gray-500 [.dark_&]:text-gray-400">
          {{ t('assets.emptyDesc') }}
        </p>
      </div>

      <div v-else class="grid items-stretch gap-4" :class="gridClass">
        <article
          v-for="item in items"
          :key="item.id"
          class="group relative flex h-full min-h-0 min-w-0 flex-col overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm transition hover:-translate-y-0.5 hover:shadow-md [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900"
        >
          <button
            type="button"
            class="absolute right-1.5 top-1.5 z-20 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-black/45 text-white shadow-sm transition hover:bg-black/60 hover:text-red-200 cursor-pointer"
            :aria-label="t('assets.deleteAsset')"
            :title="t('common.delete')"
            @click.stop="openDeleteConfirm(item)"
          >
            <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="2"
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
              />
            </svg>
          </button>
          <template v-if="isImage(item)">
            <button
              type="button"
              class="relative block aspect-[3/4] w-full shrink-0 overflow-hidden bg-gray-100 text-left [.dark_&]:bg-gray-800 cursor-pointer"
              @click="openPreview(item)"
            >
              <img
                v-if="thumbUrl(item)"
                :src="thumbUrl(item)"
                :alt="assetTitle(item)"
                class="h-full w-full object-cover transition duration-300 group-hover:scale-105"
                loading="lazy"
              >
              <div v-else class="flex h-full w-full items-center justify-center text-sm text-gray-400">{{ t('assets.image') }}</div>
              <div class="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent p-3 opacity-0 transition-opacity group-hover:opacity-100">
                <p class="truncate text-sm font-medium text-white">{{ assetTitle(item) }}</p>
              </div>
            </button>
            <AssetCardFooter
              :item="item"
              :show-remix="remixSupported(item)"
              @preview="openPreview"
              @copy="copyAssetLink"
              @download="downloadAsset"
              @remix="applyRemixFromAsset"
            />
          </template>

          <template v-else-if="isVideo(item)">
            <button
              type="button"
              class="relative block aspect-[3/4] w-full shrink-0 overflow-hidden bg-gray-100 text-left [.dark_&]:bg-gray-800 cursor-pointer"
              @click="openPreview(item)"
            >
              <img
                v-if="thumbUrl(item) && thumbUrl(item) !== assetUrl(item)"
                :src="thumbUrl(item)"
                :alt="assetTitle(item)"
                class="h-full w-full object-cover transition duration-300 group-hover:scale-105"
                loading="lazy"
              >
              <video v-else-if="assetUrl(item)" :src="assetUrl(item)" class="h-full w-full object-cover" muted preload="metadata" />
              <div v-else class="flex h-full w-full items-center justify-center text-sm text-gray-400">{{ t('assets.video') }}</div>
              <div class="absolute inset-0 flex items-center justify-center bg-black/10 opacity-90">
                <span class="flex h-12 w-12 items-center justify-center rounded-full bg-black/60 text-white shadow-lg">
                  <svg class="ml-0.5 h-5 w-5" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                    <path d="M6.5 4.75v10.5L15 10 6.5 4.75z" />
                  </svg>
                </span>
              </div>
            </button>
            <AssetCardFooter
              :item="item"
              :show-remix="remixSupported(item)"
              @preview="openPreview"
              @copy="copyAssetLink"
              @download="downloadAsset"
              @remix="applyRemixFromAsset"
            />
          </template>

          <template v-else-if="isAudio(item)">
            <div class="flex h-full min-w-0 flex-col p-4">
              <div class="flex min-w-0 items-start gap-3">
                <div class="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-indigo-50 text-sm font-semibold text-indigo-700 [.dark_&]:bg-indigo-950/40 [.dark_&]:text-indigo-300">
                  AUD
                </div>
                <div class="min-w-0 flex-1">
                  <h3 class="truncate text-sm font-semibold" :title="assetTitle(item)">{{ assetTitle(item) }}</h3>
                  <p class="mt-1 text-xs text-gray-500 [.dark_&]:text-gray-400">{{ formatDate(item.created_at) }} · {{ formatFileSize(item.file_size) }}</p>
                </div>
              </div>
              <audio v-if="audioPlaybackUrl(item)" class="mt-4 w-full" :src="audioPlaybackUrl(item)" controls preload="metadata" />
              <PromptCopyBlock
                :prompt="item.prompt || ''"
                wrapper-class="mt-4"
                paragraph-class="line-clamp-2 min-h-[2.75rem] text-sm leading-relaxed text-gray-500 [.dark_&]:text-gray-400"
              />
              <div class="mt-auto flex flex-wrap gap-2 pt-4">
                <ActionButton :label="t('common.play')" @click="openPreview(item)" />
                <ActionButton :label="t('common.download')" @click="downloadAsset(item)" />
                <ActionButton :label="t('common.copyLink')" @click="copyAssetLink(item)" />
                <ActionButton v-if="remixSupported(item)" label="Remix" @click="applyRemixFromAsset(item)" />
              </div>
            </div>
          </template>

          <template v-else>
            <div class="flex h-full min-w-0 flex-col p-4">
              <div class="flex min-w-0 items-start gap-3">
                <div class="flex h-12 w-12 shrink-0 items-center justify-center rounded-2xl bg-amber-50 text-xs font-semibold text-amber-700 [.dark_&]:bg-amber-950/40 [.dark_&]:text-amber-300">
                  {{ fileExtension(item) }}
                </div>
                <div class="min-w-0 flex-1">
                  <h3 class="truncate text-sm font-semibold" :title="assetTitle(item)">{{ assetTitle(item) }}</h3>
                  <p class="mt-1 text-xs text-gray-500 [.dark_&]:text-gray-400">{{ item.mime_type || t('assets.file') }}</p>
                </div>
              </div>
              <p class="mt-4 text-xs text-gray-500 [.dark_&]:text-gray-400">
                {{ formatDate(item.created_at) }} · {{ formatFileSize(item.file_size) }} · {{ item.task_type || 'text' }}
              </p>
              <PromptCopyBlock
                :prompt="item.prompt || ''"
                wrapper-class="mt-3"
                paragraph-class="line-clamp-2 min-h-[2.75rem] text-sm leading-relaxed text-gray-500 [.dark_&]:text-gray-400"
              />
              <div class="mt-auto flex flex-wrap gap-2 pt-4">
                <ActionButton :label="t('common.open')" @click="openAsset(item)" />
                <ActionButton :label="t('common.download')" @click="downloadAsset(item)" />
                <ActionButton :label="t('common.copyLink')" @click="copyAssetLink(item)" />
                <ActionButton v-if="remixSupported(item)" label="Remix" @click="applyRemixFromAsset(item)" />
              </div>
            </div>
          </template>
        </article>
      </div>
    </div>

    <div class="shrink-0 border-t border-gray-200 bg-white/80 px-5 py-3 backdrop-blur [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900/80">
      <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <p class="text-xs text-gray-500 [.dark_&]:text-gray-400">
          {{ t('assets.pagination', { total, page, lastPage }) }}
        </p>
        <PaginationBar :page="page" :last-page="lastPage" :per-page="perPage" :show-when-single="true" @change="onPageChange" />
      </div>
    </div>

    <Teleport to="body">
      <div
        v-if="previewAsset"
        class="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
        role="presentation"
        @click.self="closePreview"
      >
        <button
          type="button"
          class="absolute left-3 top-1/2 z-10 flex h-11 w-11 -translate-y-1/2 items-center justify-center rounded-full bg-white/90 text-gray-900 shadow-lg transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-40 [.dark_&]:bg-gray-900/90 [.dark_&]:text-white [.dark_&]:hover:bg-gray-900 sm:left-6 cursor-pointer"
          :disabled="!canPreviewPrev"
          :aria-label="t('assets.previewPrev')"
          :title="t('assets.prev')"
          @click.stop="previewPrev"
        >
          <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
        </button>

        <button
          type="button"
          class="absolute right-3 top-1/2 z-10 flex h-11 w-11 -translate-y-1/2 items-center justify-center rounded-full bg-white/90 text-gray-900 shadow-lg transition hover:bg-white disabled:cursor-not-allowed disabled:opacity-40 [.dark_&]:bg-gray-900/90 [.dark_&]:text-white [.dark_&]:hover:bg-gray-900 sm:right-6 cursor-pointer"
          :disabled="!canPreviewNext"
          :aria-label="t('assets.previewNext')"
          :title="t('assets.next')"
          @click.stop="previewNext"
        >
          <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
          </svg>
        </button>

        <div class="max-h-full w-full max-w-5xl overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-2xl [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900">
          <div class="flex items-center justify-between border-b border-gray-100 px-4 py-3 [.dark_&]:border-gray-800">
            <div class="min-w-0">
              <h2 class="truncate text-sm font-semibold text-gray-950 [.dark_&]:text-white">{{ assetTitle(previewAsset) }}</h2>
              <p class="mt-0.5 text-xs text-gray-500 [.dark_&]:text-gray-400">{{ previewPositionText }} · {{ formatDate(previewAsset.created_at) }} · {{ formatFileSize(previewAsset.file_size) }}</p>
            </div>
            <button type="button" class="rounded-full p-2 text-gray-500 transition hover:bg-gray-100 hover:text-gray-900 [.dark_&]:text-gray-400 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-white cursor-pointer" @click="closePreview">
              <span class="sr-only">{{ t('common.close') }}</span>
              <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div class="relative max-h-[75vh] overflow-auto bg-gray-50 p-4 [.dark_&]:bg-gray-950">
            <img v-if="isImage(previewAsset) && assetUrl(previewAsset)" :src="assetUrl(previewAsset)" :alt="assetTitle(previewAsset)" class="mx-auto max-h-[68vh] rounded-xl object-contain">
            <video v-else-if="isVideo(previewAsset) && assetUrl(previewAsset)" :src="assetUrl(previewAsset)" class="mx-auto max-h-[68vh] max-w-full rounded-xl bg-black" controls autoplay />
            <div v-else-if="isAudio(previewAsset)" class="mx-auto max-w-2xl rounded-2xl border border-gray-200 bg-white p-5 [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
              <p class="mb-4 text-sm text-gray-500 [.dark_&]:text-gray-400">{{ previewAsset.prompt || t('assets.noPromptInfo') }}</p>
              <audio v-if="audioPlaybackUrl(previewAsset)" :src="audioPlaybackUrl(previewAsset)" class="w-full" controls autoplay />
            </div>
            <div v-else class="mx-auto max-w-2xl rounded-2xl border border-gray-200 bg-white p-5 text-center [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
              <p class="text-sm text-gray-500 [.dark_&]:text-gray-400">{{ t('assets.previewUnsupported') }}</p>
              <div class="mt-4 flex justify-center gap-2">
                <ActionButton :label="t('common.open')" @click="openAsset(previewAsset)" />
                <ActionButton :label="t('common.download')" @click="downloadAsset(previewAsset)" />
              </div>
            </div>

            <button
              type="button"
              class="absolute bottom-6 left-1/2 z-20 flex h-11 w-11 -translate-x-1/2 items-center justify-center rounded-full bg-white/90 text-gray-900 shadow-lg ring-1 ring-black/5 backdrop-blur transition hover:bg-white [.dark_&]:bg-gray-900/90 [.dark_&]:text-white [.dark_&]:ring-white/10 [.dark_&]:hover:bg-gray-900 cursor-pointer"
              :aria-label="t('assets.downloadCurrent')"
              :title="t('common.download')"
              @click.stop="downloadAsset(previewAsset)"
            >
              <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1M12 4v12m0 0l-4-4m4 4l4-4" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </Teleport>

    <Teleport to="body">
      <div
        v-if="deleteConfirmItem"
        class="fixed inset-0 z-[60] flex items-center justify-center bg-black/45 p-4 backdrop-blur-[2px]"
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-asset-dialog-title"
        @click.self="closeDeleteConfirm"
      >
        <div
          class="w-full max-w-[420px] overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-2xl ring-1 ring-black/5 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900 [.dark_&]:ring-white/10"
          @click.stop
        >
          <div class="flex gap-4 px-6 pt-6">
            <div
              class="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-red-50 text-red-600 [.dark_&]:bg-red-950/50 [.dark_&]:text-red-400"
              aria-hidden="true"
            >
              <svg class="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                />
              </svg>
            </div>
            <div class="min-w-0 flex-1 pb-2">
              <h2 id="delete-asset-dialog-title" class="text-lg font-semibold text-gray-950 [.dark_&]:text-white">
                {{ t('assets.deleteTitle') }}
              </h2>
              <p class="mt-2 text-sm leading-relaxed text-gray-600 [.dark_&]:text-gray-400">
                {{ t('assets.deleteConfirm', { name: assetTitle(deleteConfirmItem) }) }}
              </p>
            </div>
          </div>
          <div class="mt-2 flex justify-end gap-2 border-t border-gray-100 px-6 py-4 [.dark_&]:border-gray-800">
            <button
              type="button"
              class="rounded-full border border-gray-200 bg-white px-4 py-2 text-sm font-medium text-gray-700 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50 [.dark_&]:border-gray-600 [.dark_&]:bg-gray-900 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800 cursor-pointer"
              :disabled="deleteInProgress"
              @click="closeDeleteConfirm"
            >
              {{ t('common.cancel') }}
            </button>
            <button
              type="button"
              class="rounded-full bg-red-600 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-60 cursor-pointer"
              :disabled="deleteInProgress"
              @click="executeDeleteAsset"
            >
              {{ deleteInProgress ? t('assets.deleting') : t('common.delete') }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { computed, defineComponent, h, onBeforeUnmount, onMounted, onUnmounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import PaginationBar from '../../components/PaginationBar.vue'
import { useRouter } from 'vue-router'
import { listUserAssets, deleteUserAsset, type AssetCategory, type UserAsset } from '../../api/assets'
import { showTopSnack } from '../../composables/useTopSnack'
import { handleApiError } from '../../utils/api'
import { setLocalCache } from '../../utils/localCache'
import { stripUndefinedDeep } from '../../utils/stripUndefined'
import {
  TASK_CREATE_CACHE_KEY,
  taskCreateCacheKeyByType,
  type TaskCreateRequest,
} from '../../utils/taskRunner'

type AssetTab = 'all' | 'image' | 'video' | 'audio' | 'file'

const { t } = useI18n()

/** Prompt 区域：与 AgentChat 助手消息「复制」相同的图标与剪贴板逻辑（见 AgentAssistantMessageMenu.vue） */
const PromptCopyBlock = defineComponent({
  name: 'PromptCopyBlock',
  props: {
    prompt: { type: String, default: '' },
    wrapperClass: { type: String, default: 'mt-2' },
    paragraphClass: {
      type: String,
      default: 'line-clamp-2 min-h-[2.25rem] text-xs leading-5 text-gray-500 [.dark_&]:text-gray-400',
    },
  },
  setup(props) {
    const { t: tCopy } = useI18n()
    const copied = ref(false)
    let resetTimer: ReturnType<typeof setTimeout> | null = null

    onBeforeUnmount(() => {
      if (resetTimer != null) {
        clearTimeout(resetTimer)
        resetTimer = null
      }
    })

    function onCopy(e: MouseEvent) {
      e.stopPropagation()
      const text = String(props.prompt ?? '').trim()
      if (!text) {
        showTopSnack(tCopy('assets.noPromptToCopy'))
        return
      }
      void navigator.clipboard.writeText(text).then(
        () => {
          if (resetTimer != null) {
            clearTimeout(resetTimer)
            resetTimer = null
          }
          copied.value = true
          resetTimer = setTimeout(() => {
            copied.value = false
            resetTimer = null
          }, 3000)
        },
        () => showTopSnack(tCopy('assets.copyFailed')),
      )
    }

    function copyIcon() {
      return h('svg', {
        class: 'h-4 w-4 shrink-0 text-gray-500 [.dark_&]:text-gray-400',
        fill: 'none',
        stroke: 'currentColor',
        viewBox: '0 0 24 24',
        'aria-hidden': 'true',
      }, [
        h('path', {
          'stroke-linecap': 'round',
          'stroke-linejoin': 'round',
          'stroke-width': '2',
          d: 'M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z',
        }),
      ])
    }

    function checkIcon() {
      return h('svg', {
        class: 'h-4 w-4 shrink-0 text-green-600 [.dark_&]:text-green-400',
        fill: 'none',
        stroke: 'currentColor',
        viewBox: '0 0 24 24',
        'aria-hidden': 'true',
      }, [
        h('path', {
          'stroke-linecap': 'round',
          'stroke-linejoin': 'round',
          'stroke-width': '2',
          d: 'M5 13l4 4L19 7',
        }),
      ])
    }

    return () => {
      const display = String(props.prompt ?? '').trim() ? props.prompt : tCopy('assets.noPromptInfo')
      const done = copied.value
      return h('div', { class: `${props.wrapperClass} flex min-w-0 items-start gap-1` }, [
        h(
          'button',
          {
            type: 'button',
            class: done
              ? 'inline-flex h-6 min-w-5 shrink-0 cursor-pointer items-center justify-center rounded px-0.5 text-green-600 [.dark_&]:text-green-400'
              : 'inline-flex h-6 min-w-5 shrink-0 cursor-pointer items-center justify-center rounded px-0.5 text-gray-500 hover:bg-gray-200/80 hover:text-gray-900 [.dark_&]:hover:bg-gray-700/40 [.dark_&]:hover:text-gray-300',
            title: done ? tCopy('common.copied') : tCopy('common.copy'),
            'aria-label': done ? tCopy('common.copied') : tCopy('assets.copyPrompt'),
            onClick: onCopy,
          },
          [done ? checkIcon() : copyIcon()],
        ),
        h('p', { class: `min-w-0 flex-1 ${props.paragraphClass}` }, display),
      ])
    }
  },
})

const ActionButton = defineComponent({
  props: {
    label: {
      type: String,
      required: true,
    },
  },
  emits: ['click'],
  setup(props, { emit }) {
    return () =>
      h(
        'button',
        {
          type: 'button',
          class:
            'rounded-xl border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-700 transition hover:bg-gray-50 [.dark_&]:border-gray-700 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800 cursor-pointer',
          onClick: () => emit('click'),
        },
        props.label
      )
  },
})

const tabDefs: Array<{ labelKey: string; value: AssetTab; category?: AssetCategory }> = [
  { labelKey: 'assets.tabs.all', value: 'all' },
  { labelKey: 'assets.tabs.image', value: 'image', category: 'image' },
  { labelKey: 'assets.tabs.video', value: 'video', category: 'video' },
  { labelKey: 'assets.tabs.audio', value: 'audio', category: 'audio' },
  { labelKey: 'assets.tabs.file', value: 'file', category: 'text' },
]

const tabs = computed(() =>
  tabDefs.map((tab) => ({
    ...tab,
    label: t(tab.labelKey),
  }))
)

const activeTab = ref<AssetTab>('all')
const searchKeyword = ref('')
const submittedKeyword = ref('')
const items = ref<UserAsset[]>([])
const loading = ref(false)
const total = ref(0)
const page = ref(1)
const perPage = ref(24)
const previewAsset = ref<UserAsset | null>(null)
const deleteConfirmItem = ref<UserAsset | null>(null)
const deleteInProgress = ref(false)

const router = useRouter()

const activeTabClass = 'bg-gray-950 text-white shadow-sm [.dark_&]:bg-white [.dark_&]:text-gray-950'
const inactiveTabClass = 'bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-gray-950 [.dark_&]:bg-gray-800 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-700 [.dark_&]:hover:text-white'

const activeCategory = computed(() => tabs.value.find((tab) => tab.value === activeTab.value)?.category)
const lastPage = computed(() => Math.max(1, Math.ceil(total.value / perPage.value)))
const gridClass = computed(() => {
  if (activeTab.value === 'audio' || activeTab.value === 'file') return 'grid-cols-1 xl:grid-cols-2'
  return 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5'
})
const previewIndex = computed(() => {
  const currentId = previewAsset.value?.id
  if (!currentId) return -1
  return items.value.findIndex((item) => item.id === currentId)
})
const canPreviewPrev = computed(() => previewIndex.value > 0)
const canPreviewNext = computed(() => previewIndex.value >= 0 && previewIndex.value < items.value.length - 1)
const previewPositionText = computed(() => {
  if (previewIndex.value < 0) return t('assets.pageItems', { count: items.value.length })
  return `${previewIndex.value + 1} / ${items.value.length}`
})

function isImage(item: UserAsset | null | undefined) {
  return String(item?.category || '').toLowerCase() === 'image'
}

function isVideo(item: UserAsset | null | undefined) {
  return String(item?.category || '').toLowerCase() === 'video'
}

function isAudio(item: UserAsset | null | undefined) {
  return String(item?.category || '').toLowerCase() === 'audio'
}

function assetUrl(item: UserAsset | null | undefined) {
  return String(item?.url || item?.http_url || '').trim()
}

/** ASR 音频任务：列表/预览中的播放器应使用输入音频；产物 url 多为识别文本等，不适合作为播放源。 */
function audioPlaybackUrl(item: UserAsset | null | undefined) {
  if (!item) return ''
  const taskType = String(item.task_type ?? '').trim().toLowerCase()
  const jobType = String(item.task_params?.job_type ?? '').trim().toUpperCase()
  if (taskType === 'audio' && jobType === 'ASR') {
    const input = String(item.task_params?.input_audio_url ?? '').trim()
    if (input) return input
  }
  return assetUrl(item)
}

function downloadUrl(item: UserAsset | null | undefined) {
  const url = assetUrl(item)
  if (!url) return ''
  try {
    const parsed = new URL(url, window.location.href)
    if (parsed.pathname.startsWith('/outputs/')) {
      return `${parsed.pathname}${parsed.search}${parsed.hash}`
    }
  } catch {
    // Keep the original value for relative URLs that URL() cannot normalize.
  }
  return url
}

function thumbUrl(item: UserAsset | null | undefined) {
  return String(item?.thumb_url || item?.thumb_http_url || item?.url || item?.http_url || '').trim()
}

function assetTitle(item: UserAsset | null | undefined) {
  const name = String(item?.file_name || '').trim()
  if (name) return name
  const path = String(item?.storage_path || '').trim()
  if (!path) return t('assets.unnamed')
  return path.split(/[\\/]/).filter(Boolean).pop() || t('assets.unnamed')
}

function fileExtension(item: UserAsset) {
  const name = assetTitle(item)
  const ext = name.includes('.') ? name.split('.').pop() || '' : ''
  return (ext || 'FILE').slice(0, 5).toUpperCase()
}

function formatDate(value?: string | null) {
  if (!value) return t('assets.unknownTime')
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return t('assets.unknownTime')
  return date.toLocaleString('zh-CN', { hour12: false })
}

function formatFileSize(value?: number | null) {
  const size = Number(value || 0)
  if (!Number.isFinite(size) || size <= 0) return t('assets.unknownSize')
  const units = ['B', 'KB', 'MB', 'GB']
  let n = size
  let unit = 0
  while (n >= 1024 && unit < units.length - 1) {
    n /= 1024
    unit += 1
  }
  return `${n >= 10 || unit === 0 ? n.toFixed(0) : n.toFixed(1)} ${units[unit]}`
}

type RemixRouteName = 'ImageGenerate' | 'ImageEdit' | 'VideoGenerate' | 'Audio'

const TASK_REMIX_CACHE_TTL_MS = 30 * 24 * 60 * 60 * 1000

/** 作品集接口若未带 task_type，用 category 兜底（与后端约定一致时再收紧） */
function effectiveTaskType(item: UserAsset): string {
  const explicit = String(item.task_type ?? '').trim().toLowerCase()
  if (explicit) return explicit
  return String(item.category ?? '').trim().toLowerCase()
}

function resolveRemixRoute(item: UserAsset): RemixRouteName | null {
  const tt = effectiveTaskType(item)
  const job = String(item.task_params?.job_type ?? '').trim().toUpperCase()
  if (tt === 'image' && job === 'MK') return 'ImageGenerate'
  if (tt === 'image' && job === 'ED') return 'ImageEdit'
  if (tt === 'video') return 'VideoGenerate'
  if (tt === 'audio' && (job === 'ASR' || job === 'TTS')) return 'Audio'
  return null
}

function remixSupported(item: UserAsset): boolean {
  return resolveRemixRoute(item) != null
}

function buildRemixTaskCachePayload(item: UserAsset): TaskCreateRequest | null {
  const route = resolveRemixRoute(item)
  if (!route) return null
  const tp = item.task_params
  const raw: Record<string, unknown> =
    tp && typeof tp === 'object' && !Array.isArray(tp) ? { ...(tp as Record<string, unknown>) } : {}

  const itemPrompt = String(item.prompt ?? '').trim()
  if (itemPrompt && !String(raw.prompt ?? '').trim()) {
    raw.prompt = itemPrompt
  }

  if (route === 'ImageGenerate') {
    return stripUndefinedDeep({
      ...raw,
      task_type: 'image',
      job_type: 'MK',
    }) as TaskCreateRequest
  }

  if (route === 'ImageEdit') {
    let tpl: string[] = []
    const tl = raw.tpl_list
    if (Array.isArray(tl)) {
      tpl = tl.map((x) => String(x).trim()).filter(Boolean)
    }
    if (!tpl.length) {
      const u = assetUrl(item).trim() || thumbUrl(item).trim()
      if (u) tpl = [u]
    }
    if (!tpl.length) return null
    return stripUndefinedDeep({
      ...raw,
      task_type: 'image',
      job_type: 'ED',
      tpl_list: tpl,
    }) as TaskCreateRequest
  }

  if (route === 'VideoGenerate') {
    return stripUndefinedDeep({
      ...raw,
      task_type: 'video',
    }) as TaskCreateRequest
  }

  return stripUndefinedDeep({
    ...raw,
    task_type: 'audio',
  }) as TaskCreateRequest
}

function applyRemixFromAsset(item: UserAsset) {
  const routeName = resolveRemixRoute(item)
  const payload = buildRemixTaskCachePayload(item)
  if (!routeName || !payload) {
    showTopSnack(t('assets.remixUnsupported'))
    return
  }
  const subtype = payload.task_type
  if (subtype !== 'image' && subtype !== 'video' && subtype !== 'audio') {
    showTopSnack(t('assets.remixFailed'))
    return
  }
  setLocalCache(TASK_CREATE_CACHE_KEY, payload, { ttlMs: TASK_REMIX_CACHE_TTL_MS })
  setLocalCache(taskCreateCacheKeyByType(subtype), payload, { ttlMs: TASK_REMIX_CACHE_TTL_MS })
  router.push({ name: routeName })
}

const AssetCardFooter = defineComponent({
  props: {
    item: {
      type: Object,
      required: true,
    },
    showRemix: {
      type: Boolean,
      default: false,
    },
  },
  emits: ['preview', 'copy', 'download', 'remix'],
  setup(props, { emit }) {
    const item = props.item as UserAsset
    return () =>
      h('div', { class: 'flex min-h-0 min-w-0 flex-1 flex-col p-3' }, [
        h('h3', { class: 'min-w-0 truncate text-sm font-semibold text-gray-950 [.dark_&]:text-white', title: assetTitle(item) }, assetTitle(item)),
        h('p', { class: 'mt-1 min-w-0 truncate text-xs text-gray-500 [.dark_&]:text-gray-400' }, `${formatDate(item.created_at)} · ${formatFileSize(item.file_size)}`),
        h(PromptCopyBlock, { prompt: item.prompt || '' }),
        h('div', { class: 'mt-auto flex flex-wrap gap-2 pt-3' }, [
          h(ActionButton, { label: t('assets.preview'), onClick: () => emit('preview', item) }),
          h(ActionButton, { label: t('common.download'), onClick: () => emit('download', item) }),
          h(ActionButton, { label: t('common.copyLink'), onClick: () => emit('copy', item) }),
          ...(props.showRemix ? [h(ActionButton, { label: 'Remix', onClick: () => emit('remix', item) })] : []),
        ]),
      ])
  },
})

async function loadAssets() {
  loading.value = true
  try {
    const result = await listUserAssets({
      category: activeCategory.value,
      keyword: submittedKeyword.value.trim() || undefined,
      limit: perPage.value,
      offset: (page.value - 1) * perPage.value,
    })
    items.value = result.items || []
    total.value = Number(result.total || 0)
  } catch (error) {
    console.error('Failed to load assets:', error)
    showTopSnack(t('assets.loadFailed'))
    items.value = []
    total.value = 0
  } finally {
    loading.value = false
  }
}

function setActiveTab(tab: AssetTab) {
  if (activeTab.value === tab) return
  activeTab.value = tab
  page.value = 1
}

function submitSearch() {
  submittedKeyword.value = searchKeyword.value.trim()
  page.value = 1
  loadAssets()
}

function onPageChange(payload: { page: number; perPage: number }) {
  page.value = payload.page
  perPage.value = payload.perPage
  loadAssets()
}

function openPreview(item: UserAsset) {
  previewAsset.value = item
}

function closePreview() {
  previewAsset.value = null
}

function previewPrev() {
  if (!canPreviewPrev.value) return
  previewAsset.value = items.value[previewIndex.value - 1] || null
}

function previewNext() {
  if (!canPreviewNext.value) return
  previewAsset.value = items.value[previewIndex.value + 1] || null
}

function handlePreviewKeydown(event: KeyboardEvent) {
  if (deleteConfirmItem.value) {
    if (event.key === 'Escape') {
      event.preventDefault()
      closeDeleteConfirm()
    }
    return
  }
  if (!previewAsset.value) return
  if (event.key === 'ArrowLeft') {
    event.preventDefault()
    previewPrev()
  } else if (event.key === 'ArrowRight') {
    event.preventDefault()
    previewNext()
  } else if (event.key === 'Escape') {
    event.preventDefault()
    closePreview()
  }
}

function openAsset(item: UserAsset) {
  const url = assetUrl(item)
  if (!url) {
    showTopSnack(t('assets.noOpenUrl'))
    return
  }
  window.open(url, '_blank', 'noopener,noreferrer')
}

function downloadAsset(item: UserAsset) {
  const url = downloadUrl(item)
  if (!url) {
    showTopSnack(t('assets.noDownloadUrl'))
    return
  }
  const a = document.createElement('a')
  a.href = url
  a.download = assetTitle(item)
  a.rel = 'noopener noreferrer'
  document.body.appendChild(a)
  a.click()
  a.remove()
}

async function copyAssetLink(item: UserAsset) {
  const url = assetUrl(item)
  if (!url) {
    showTopSnack(t('assets.noCopyUrl'))
    return
  }
  try {
    await navigator.clipboard.writeText(url)
    showTopSnack(t('assets.linkCopied'))
  } catch (error) {
    console.error('Failed to copy asset link:', error)
    showTopSnack(t('assets.copyFailed'))
  }
}

function openDeleteConfirm(item: UserAsset) {
  deleteConfirmItem.value = item
}

function closeDeleteConfirm() {
  if (deleteInProgress.value) return
  deleteConfirmItem.value = null
}

async function executeDeleteAsset() {
  const item = deleteConfirmItem.value
  if (!item || deleteInProgress.value) return
  deleteInProgress.value = true
  try {
    await deleteUserAsset(item.id)
    showTopSnack(t('assets.deleted'))
    deleteConfirmItem.value = null
    if (previewAsset.value?.id === item.id) {
      closePreview()
    }
    items.value = items.value.filter((row) => row.id !== item.id)
    total.value = Math.max(0, total.value - 1)
    if (items.value.length === 0 && page.value > 1) {
      page.value -= 1
      await loadAssets()
    }
  } catch (error) {
    console.error('Failed to delete asset:', error)
    showTopSnack(handleApiError(error).message || t('assets.deleteFailed'))
  } finally {
    deleteInProgress.value = false
  }
}

watch(activeTab, loadAssets)
watch(searchKeyword, (value) => {
  if (String(value || '').trim() !== '' || !submittedKeyword.value.trim()) return
  submittedKeyword.value = ''
  page.value = 1
  loadAssets()
})
onMounted(() => {
  loadAssets()
  window.addEventListener('keydown', handlePreviewKeydown)
})

onUnmounted(() => {
  window.removeEventListener('keydown', handlePreviewKeydown)
})
</script>

<style scoped>
/* 主列表区域：细滚动条、圆角 thumb，与页面灰阶风格一致 */
.assets-main-scroll {
  scrollbar-width: thin;
  scrollbar-color: rgb(209 213 219) transparent;
}

html.dark .assets-main-scroll {
  scrollbar-color: rgb(75 85 99) transparent;
}

.assets-main-scroll::-webkit-scrollbar {
  width: 10px;
}

.assets-main-scroll::-webkit-scrollbar-track {
  margin: 6px 0;
  background: transparent;
}

.assets-main-scroll::-webkit-scrollbar-thumb {
  min-height: 40px;
  border-radius: 9999px;
  border: 3px solid transparent;
  background-clip: padding-box;
  background-color: rgb(209 213 219);
}

.assets-main-scroll::-webkit-scrollbar-thumb:hover {
  background-color: rgb(156 163 175);
}

html.dark .assets-main-scroll::-webkit-scrollbar-thumb {
  background-color: rgb(75 85 99);
}

html.dark .assets-main-scroll::-webkit-scrollbar-thumb:hover {
  background-color: rgb(107 114 128);
}
</style>

