<template>
  <div
    :class="[
      variant === 'plain'
        ? 'space-y-3'
        : 'rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-3 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35',
      fill ? 'h-full flex flex-col' : '',
    ]"
  >
    <!-- <div class="flex items-center justify-between gap-3">
      <div class="text-sm font-medium text-gray-300">
        {{ label }}（最多 {{ max }} 张）
      </div>
    </div> -->

    <div
      class="grid gap-2 items-stretch"
      :class="[
        max === 1 ? 'grid-cols-1' : 'grid-cols-3 sm:grid-cols-4',
        fill ? 'flex-1 h-full' : '',
        fill && max === 1 ? 'grid-rows-1 min-h-0' : '',
        dragActive ? 'ring-2 ring-indigo-500 rounded-xl p-2' : '',
      ]"
      :style="fill && max === 1 && fillMinHeightPx > 0 ? { minHeight: fillMinHeightPx + 'px' } : undefined"
      @dragenter.prevent="dragActive = true"
      @dragover.prevent
      @dragleave.prevent="dragActive = false"
      @drop.prevent="onDrop"
    >
      <div
        v-for="(url, idx) in internal"
        :key="url + ':' + idx"
        class="relative rounded-xl overflow-hidden border border-gray-200 bg-gray-100/50 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-900/30"
        :class="fill && max === 1 ? 'h-full min-h-0' : 'aspect-square'"
      >
        <video
          v-if="isVideoUrl(url)"
          :src="url"
          class="w-full h-full object-cover"
          muted
          playsinline
          preload="metadata"
        />
        <img v-else-if="isImageUrl(url)" :src="url" class="w-full h-full object-cover" />
        <div
          v-else
          class="w-full h-full flex flex-col items-center justify-center gap-2 p-3 text-center text-gray-500 [.dark_&]:text-gray-300"
        >
          <div class="text-3xl">📄</div>
          <div class="max-w-full truncate text-xs">{{ fileNameFromUrl(url) }}</div>
        </div>
        <!-- 缩略图关闭 -->
        <button
          type="button"
          class="absolute top-1 right-1 w-5 h-5 rounded-full bg-black/60 hover:bg-black/80 text-white text-sm flex items-center justify-center  cursor-pointer"
          @click="removeAt(idx)"
          aria-label="remove"
        >
          ✕
        </button>
      </div>

      <!-- 新增入口：下拉菜单（集成多种方式） -->
      <div v-if="internal.length < max" class="relative" :class="fill && max === 1 ? 'h-full min-h-0' : ''">
        <button
          ref="anchorRef"
          type="button"
          class="w-full rounded-xl border border-dashed border-gray-300 bg-gray-50/90 hover:bg-gray-100/95 flex flex-col items-center justify-center gap-2 text-xs leading-none text-gray-500 cursor-pointer [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/20 [.dark_&]:hover:bg-gray-900/30 [.dark_&]:text-gray-400"
          :class="fill && max === 1 ? 'h-full min-h-0' : 'aspect-square'"
          @click="toggleMenu()"
        >
          <div
            class="w-8 h-8 rounded-lg border border-gray-200 bg-white flex items-center justify-center text-gray-500 [.dark_&]:bg-gray-800/70 [.dark_&]:border-gray-700 [.dark_&]:text-gray-200"
          >
            +
          </div>
          <div class="text-gray-600 leading-none [.dark_&]:text-gray-300">{{ label }}</div>
        </button>

        <input
          ref="fileInputRef"
          type="file"
          hidden
          class="hidden"
          :accept="accept"
          multiple
          @change="onPickFiles"
        />

        <Teleport to="body">
          <div
            v-if="menuOpen"
            ref="menuRef"
            class="fixed z-9999 w-64 rounded-xl border border-gray-200 bg-white/95 text-gray-800 shadow-2xl backdrop-blur p-2 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900/95 [.dark_&]:text-gray-200"
            :style="{ left: menuLeft + 'px', top: menuTop + 'px' }"
          >
            <button
              type="button"
              class="w-full flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 text-sm cursor-pointer [.dark_&]:hover:bg-gray-800/80"
              @click="handlePickFromCreations"
            >
              <span class="w-5 text-center">🧰</span>
              {{ t('components.upload.fromAssets') }}
            </button>

            <button
              type="button"
              class="w-full flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 text-sm cursor-pointer [.dark_&]:hover:bg-gray-800/80"
              @click="handlePickFromUploads"
            >
              <span class="w-5 text-center">📤</span>
              {{ t('components.upload.fromUploads') }}
            </button>
       
            <button
              type="button"
              class="w-full flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 text-sm cursor-pointer [.dark_&]:hover:bg-gray-800/80"
              @click="handleUpload"
            >
              <span class="w-5 text-center">📎</span>
              {{ t('components.upload.uploadFile') }}
            </button>

            <button
              v-if="qrEnabled"
              type="button"
              class="w-full flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-100 text-sm cursor-pointer [.dark_&]:hover:bg-gray-800/80"
              @click="handleQrUpload"
            >
              <span class="w-5 text-center">▦</span>
              {{ t('components.upload.scanUpload') }}
            </button>
       
          </div>
        </Teleport>
      </div>
    </div>

    

    <Teleport to="body">
      <div
        v-if="snack.open"
        class="fixed top-4 left-1/2 -translate-x-1/2 z-9999 max-w-[92vw] px-4 py-3 rounded-xl border border-red-200 bg-white/95 text-red-800 shadow-xl backdrop-blur [.dark_&]:border-red-500/30 [.dark_&]:bg-gray-900/90 [.dark_&]:text-red-100"
        role="status"
        aria-live="polite"
      >
        {{ snack.message }}
      </div>
    </Teleport>

    <Teleport to="body">
      <div
        v-if="history.open"
        class="fixed inset-0 z-9999 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
        @mousedown.self="history.open = false"
      >
        <div class="vt-card p-4 flex flex-col overflow-hidden w-[1120px] h-[720px] max-w-[92vw] max-h-[86vh]">
          <!-- header -->
          <div class="space-y-3">
            <!-- 第一行：标题 + 关闭按钮 -->
            <div class="flex items-center justify-between gap-4">
              <div class="flex items-center gap-3 min-w-0">
                <div class="text-base font-semibold text-gray-900 truncate [.dark_&]:text-gray-100">
                  {{ historyTitle }}
                </div>
                <div class="text-xs text-gray-500 shrink-0 [.dark_&]:text-gray-400">
                  {{ t('components.upload.selectedCount', { current: internal.length, max }) }}
                </div>
              </div>
              <button
                type="button"
                class="shrink-0 w-8 h-8 rounded-full border border-gray-200 hover:bg-gray-100 text-gray-600 flex items-center justify-center text-lg cursor-pointer [.dark_&]:border-transparent [.dark_&]:hover:bg-gray-700 [.dark_&]:text-gray-200"
                @click="history.open = false"
                :aria-label="t('common.close')"
              >
                ✕
              </button>
            </div>
            
            <!-- 第二行：标签切换 + 搜索 -->
            <div class="flex items-center justify-between gap-3">
              <div
                class="inline-flex rounded-xl border border-gray-200 bg-gray-100/90 p-1 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900/40"
              >
                <button
                  type="button"
                  class="px-3 py-1.5 text-xs rounded-lg transition-colors cursor-pointer"
                  :class="
                    history.source === 'creations'
                      ? 'bg-indigo-600 text-white'
                      : 'text-gray-600 hover:bg-white/90 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800/60'
                  "
                  @click="switchSource('creations')"
                >
                  {{ t('components.upload.assets') }}
                </button>
                <button
                  type="button"
                  class="px-3 py-1.5 text-xs rounded-lg transition-colors cursor-pointer"
                  :class="
                    history.source === 'uploads'
                      ? 'bg-indigo-600 text-white'
                      : 'text-gray-600 hover:bg-white/90 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800/60'
                  "
                  @click="switchSource('uploads')"
                >
                  {{ t('components.upload.uploads') }}
                </button>
              </div>

              <div class="flex items-center gap-2 min-w-0 w-[520px]">
                <div class="relative flex-1 min-w-0">
                  <input
                    v-model="history.keyword"
                    type="text"
                    :placeholder="history.source === 'creations' ? t('components.upload.searchPrompt') : t('components.upload.searchFilename')"
                    class="w-full pl-10 pr-3 py-2 rounded-xl border border-gray-200 bg-white text-sm text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-200 [.dark_&]:placeholder:text-gray-500"
                    @keydown.enter="loadHistory(0)"
                  />
                  <div class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 text-sm [.dark_&]:text-gray-500">
                    ⌕
                  </div>
                </div>
                <button
                  type="button"
                  class="px-4 py-2 text-sm rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white cursor-pointer"
                  @click="loadHistory(0)"
                >
                  {{ t('common.search') }}
                </button>
              </div>
            </div>
          </div>

          <!-- body -->
          <div class="mt-4 flex-1 overflow-y-auto vt-scroll pr-1">
            <div v-if="history.loading" class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
              <div
                v-for="i in 12"
                :key="'sk-' + i"
                class="rounded-2xl border border-gray-200 bg-gray-100/60 aspect-square overflow-hidden [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-900/30"
              >
                <div
                  class="w-full h-full animate-pulse bg-linear-to-br from-gray-200/80 to-gray-100/20 [.dark_&]:from-gray-800/30 [.dark_&]:to-gray-900/10"
                ></div>
              </div>
            </div>

            <div v-else-if="history.error" class="py-14 text-center">
              <div class="text-sm text-red-600 [.dark_&]:text-red-300">{{ history.error }}</div>
              <button
                type="button"
                class="mt-4 px-4 py-2 text-sm rounded-xl border border-gray-200 bg-white hover:bg-gray-50 text-gray-800 cursor-pointer [.dark_&]:border-gray-700 [.dark_&]:bg-gray-800 [.dark_&]:hover:bg-gray-700 [.dark_&]:text-gray-200"
                @click="loadHistory(history.offset)"
              >
                {{ t('components.upload.retry') }}
              </button>
            </div>

            <div v-else-if="history.items.length === 0" class="py-16 text-center">
              <div class="text-sm text-gray-500 [.dark_&]:text-gray-400">{{ t('components.upload.noData') }}</div>
              <div class="mt-2 text-xs text-gray-400 [.dark_&]:text-gray-500">{{ t('components.upload.tryOtherKeyword') }}</div>
            </div>

            <div v-else class="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6 p-2">
              <button
                v-for="item in history.items"
                :key="item.id"
                type="button"
                class="group relative rounded-2xl overflow-hidden border border-gray-200 bg-gray-100/50 aspect-square transition-all hover:shadow-lg hover:shadow-gray-400/25 hover:-translate-y-0.5 cursor-pointer [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-900/30 [.dark_&]:hover:shadow-black/30"
                :class="internal.includes(item.url) ? 'ring-2 ring-indigo-500 border-indigo-500/40' : 'hover:ring-2 hover:ring-indigo-500/70'"
                :title="item.file_name || ''"
                @click="selectHistory(item)"
              >
                <video
                  v-if="isVideoUrl(item.url)"
                  :src="item.thumb_url || item.url"
                  class="w-full h-full object-cover group-hover:brightness-110 transition"
                  muted
                  playsinline
                  preload="metadata"
                />
                <img
                  v-else-if="isImageUrl(item.url)"
                  :src="item.thumb_url || item.url"
                  class="w-full h-full object-cover group-hover:brightness-110 transition"
                />
                <div
                  v-else
                  class="w-full h-full flex flex-col items-center justify-center gap-2 p-3 text-gray-500 [.dark_&]:text-gray-300"
                >
                  <div class="text-3xl">📄</div>
                  <div class="max-w-full truncate text-[11px]">{{ item.file_name || fileNameFromUrl(item.url) }}</div>
                </div>
                <div class="absolute inset-x-0 bottom-0 p-2 bg-linear-to-t from-black/70 to-transparent">
                  <div class="text-[11px] text-gray-200 truncate">
                    {{ item.file_name || t('components.upload.unnamed') }}
                  </div>
                </div>
                <div
                  v-if="internal.includes(item.url)"
                  class="absolute top-2 left-2 w-6 h-6 rounded-full bg-indigo-600 text-white text-xs flex items-center justify-center shadow"
                  :title="t('components.upload.selected')"
                >
                  ✓
                </div>
              </button>
            </div>
          </div>

          <!-- footer -->
          <div class="mt-4 flex items-center justify-between gap-3 text-xs text-gray-500 [.dark_&]:text-gray-400">
            <div class="flex items-center gap-2">
              <span>{{
                t('components.upload.pagination', {
                  total: history.total,
                  page: Math.floor(history.offset / history.limit) + 1,
                  lastPage: Math.max(1, Math.ceil(history.total / history.limit)),
                })
              }}</span>
            </div>
            <PaginationBar
              :page="Math.floor(history.offset / history.limit) + 1"
              :last-page="Math.max(1, Math.ceil(history.total / history.limit))"
              :per-page="history.limit"
              @change="({ page }) => loadHistory(Math.max(0, (page - 1) * history.limit))"
            />
          </div>
        </div>
      </div>
    </Teleport>

    <QrUploadModal ref="qrModalRef" :accept="accept" @uploaded="onQrUploaded" />
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import axios from 'axios'
import PaginationBar from './PaginationBar.vue'
import QrUploadModal from './QrUploadModal.vue'
import type { QrUploadResult } from '../composables/useQrCodeUpload'
import { getAccessToken } from '../utils/auth'
import { getBackendRequestBaseURL } from '../utils/runtimeConfig'
import { uploadFile } from '../utils/upload'

type UserFileItem = {
  id: string
  url: string
  thumb_url?: string
  file_name?: string
  created_at?: string
  mime_type?: string
  file_size?: number
}

const props = withDefaults(defineProps<{
  modelValue: string[]
  max?: number
  label?: string
  variant?: 'card' | 'plain'
  /**
   * File input accept string (e.g. "image/*", "video/*", "image/*,video/*").
   * Default: "image/*"
   */
  accept?: string
  /**
   * When picking from creations (/v1/user/files), which category to load.
   * Default: "image"
   */
  category?: 'image' | 'video' | string
  /**
   * 当父容器设置了明确高度时，开启该选项可让内部选择区域按高度满铺（尤其适合 max=1 的场景）。
   */
  fill?: boolean
  /**
   * fill + max=1 时的最小高度（px）。
   * - 默认 120：避免父容器未设置高度时组件塌陷
   * - 设为 0：完全自适应父容器高度（不会用 min-height 撑开父容器）
   */
  fillMinHeight?: number
  enableFavorites?: boolean
  enableQr?: boolean
}>(), {
  enableQr: true,
})
const emit = defineEmits<{
  (e: 'update:modelValue', v: string[]): void
}>()

const { t } = useI18n()

const max = computed(() => Math.max(1, Math.min(9, Number(props.max ?? 9))))
const label = computed(() => String(props.label || t('components.upload.referenceImage')))
const variant = computed(() => (props.variant === 'plain' ? 'plain' : 'card'))
const accept = computed(() => String(props.accept || 'image/*'))
const category = computed(() => String(props.category || 'image'))
const fill = computed(() => Boolean(props.fill))
const fillMinHeightPx = computed(() => {
  const v = Number(props.fillMinHeight ?? 120)
  if (!Number.isFinite(v)) return 120
  return Math.max(0, Math.floor(v))
})
const qrEnabled = computed(() => props.enableQr !== false)

// 关键：用 computed(get/set) 实现标准 v-model，避免 watcher 双向同步导致递归更新
const internal = computed<string[]>({
  get: () => (Array.isArray(props.modelValue) ? props.modelValue : []),
  set: (v) => emit('update:modelValue', Array.isArray(v) ? v : []),
})

const dragActive = ref(false)
const fileInputRef = ref<HTMLInputElement | null>(null)

function triggerPick() {
  fileInputRef.value?.click()
}

function removeAt(idx: number) {
  const next = [...internal.value]
  next.splice(idx, 1)
  internal.value = next
}

async function uploadOne(file: File) {
  if (internal.value.length >= max.value) return
  if (!isAllowedFile(file)) {
    showSnack(t('components.upload.unsupportedType'))
    return
  }
  const url = await uploadFile(file)
  internal.value = [...internal.value, url].slice(0, max.value)
}

// ---- dropdown menu (集成入口) ----
const anchorRef = ref<HTMLElement | null>(null)
const menuRef = ref<HTMLElement | null>(null)
const menuOpen = ref(false)
const menuLeft = ref(0)
const menuTop = ref(0)

const snack = reactive({ open: false, message: '' })
let snackTimer: number | undefined

function showSnack(message: string) {
  snack.message = message
  snack.open = true
  if (snackTimer) window.clearTimeout(snackTimer)
  snackTimer = window.setTimeout(() => {
    snack.open = false
    snackTimer = undefined
  }, 2600)
}

function positionMenu() {
  const a = anchorRef.value?.getBoundingClientRect?.()
  const m = menuRef.value?.getBoundingClientRect?.()
  if (!a) return
  const vw = window.innerWidth
  const vh = window.innerHeight
  const mw = m?.width || 256
  const mh = m?.height || 220

  let left = a.left
  let top = a.bottom + 8
  if (left + mw > vw - 8) left = Math.max(8, vw - mw - 8)
  if (top + mh > vh - 8) top = Math.max(8, a.top - mh - 8)
  menuLeft.value = Math.floor(left + window.scrollX)
  menuTop.value = Math.floor(top + window.scrollY)
}

function addMenuListeners(add: boolean) {
  const onResizeOrScroll = () => positionMenu()
  const onKeydown = (e: KeyboardEvent) => {
    if (e.key === 'Escape') menuOpen.value = false
  }
  const onMousedown = (e: MouseEvent) => {
    const o = menuRef.value
    const a = anchorRef.value
    if (!o || !a) return
    if (!o.contains(e.target as any) && !a.contains(e.target as any)) {
      menuOpen.value = false
    }
  }
  if (add) {
    window.addEventListener('resize', onResizeOrScroll)
    window.addEventListener('scroll', onResizeOrScroll, true)
    document.addEventListener('keydown', onKeydown)
    document.addEventListener('mousedown', onMousedown)
    ;(menuRef as any)._rh = onResizeOrScroll
    ;(menuRef as any)._kh = onKeydown
    ;(menuRef as any)._mh = onMousedown
  } else {
    const refAny = menuRef as any
    if (refAny?._rh) window.removeEventListener('resize', refAny._rh)
    if (refAny?._rh) window.removeEventListener('scroll', refAny._rh, true)
    if (refAny?._kh) document.removeEventListener('keydown', refAny._kh)
    if (refAny?._mh) document.removeEventListener('mousedown', refAny._mh)
    if (refAny) refAny._rh = refAny._kh = refAny._mh = null
  }
}

async function toggleMenu() {
  menuOpen.value = !menuOpen.value
  if (menuOpen.value) {
    await nextTick()
    positionMenu()
    addMenuListeners(true)
  } else {
    addMenuListeners(false)
  }
}

async function closeMenu() {
  menuOpen.value = false
  addMenuListeners(false)
}

async function handlePickFromCreations() {
  await closeMenu()
  await openHistory()
}

async function handlePickFromUploads() {
  await closeMenu()
  await openUploads()
}
async function handleUpload() {
  await closeMenu()
  triggerPick()
}

const qrModalRef = ref<InstanceType<typeof QrUploadModal> | null>(null)

async function handleQrUpload() {
  await closeMenu()
  qrModalRef.value?.open()
}

function onQrUploaded(result: QrUploadResult) {
  const url = String(result.url || '').trim()
  if (!url || internal.value.length >= max.value || internal.value.includes(url)) return
  internal.value = [...internal.value, url].slice(0, max.value)
}

async function onPickFiles(e: Event) {
  const input = e.target as HTMLInputElement
  const files = Array.from(input.files || [])
  input.value = ''
  if (!files.length) return
  for (const f of files) {
    if (internal.value.length >= max.value) break
    await uploadOne(f)
  }
}

async function onDrop(e: DragEvent) {
  dragActive.value = false
  const files = Array.from(e.dataTransfer?.files || []).filter((f) => isAllowedFile(f))
  if (!files.length) return
  for (const f of files) {
    if (internal.value.length >= max.value) break
    await uploadOne(f)
  }
}

const history = reactive({
  open: false,
  loading: false,
  error: '',
  items: [] as UserFileItem[],
  total: 0,
  limit: 60,
  offset: 0,
  keyword: '',
  source: 'creations' as 'creations' | 'uploads',
})

const historyTitle = computed(() => {
  if (history.source === 'uploads') return t('components.upload.selectUploadFile')
  return accept.value.toLowerCase().includes('video')
    ? t('components.upload.selectHistoryFile')
    : t('components.upload.selectHistoryImage')
})

function isVideoUrl(url: string) {
  const u = String(url || '').toLowerCase()
  return /\.(mp4|mov|mkv|webm|avi)(?:\?|#|$)/.test(u)
}

function isImageUrl(url: string) {
  const u = String(url || '').toLowerCase()
  return /\.(png|jpe?g|webp|gif|bmp|svg)(?:\?|#|$)/.test(u) || /^data:image\//.test(u)
}

function fileNameFromUrl(url: string) {
  try {
    const pathname = new URL(url, window.location.href).pathname
    return decodeURIComponent(pathname.split('/').filter(Boolean).pop() || t('components.upload.file'))
  } catch {
    return String(url || '').split('/').filter(Boolean).pop() || t('components.upload.file')
  }
}

const imageExts = new Set(['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.svg'])
const videoExts = new Set(['.mp4', '.mov', '.mkv', '.webm', '.avi'])
const audioExts = new Set(['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg', '.opus', '.weba'])
const docExts = new Set(['.pdf', '.doc', '.docx', '.txt', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods', '.odp', '.csv', '.rtf', '.md', '.markdown', '.html', '.htm'])
const archiveExts = new Set(['.zip', '.rar', '.7z', '.tar', '.gz', '.tgz', '.bz2', '.tbz2', '.xz', '.zst', '.lz4', '.cab', '.jar', '.war'])

function parseAcceptRules(rawAccept: string) {
  return String(rawAccept || '')
    .toLowerCase()
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
}

function extFromName(name: string) {
  const match = String(name || '').toLowerCase().match(/\.([a-z0-9]+)(?:\?|#|$)/)
  return match ? `.${match[1]}` : ''
}

function mimeOrExtMatchesAccept(mime: string, nameOrUrl: string, rawAccept = accept.value) {
  const rules = parseAcceptRules(rawAccept)
  if (!rules.length) return true
  const ct = String(mime || '').toLowerCase()
  const ext = extFromName(nameOrUrl)
  return rules.some((rule) => {
    if (rule === '*/*') return true
    if (rule.startsWith('.')) return ext === rule
    if (rule.endsWith('/*')) return ct.startsWith(rule.slice(0, -1))
    return ct === rule
  })
}

function isAllowedFile(file: File) {
  return mimeOrExtMatchesAccept(String(file.type || ''), String(file.name || ''))
}

function looksAllowedUploadItem(item: UserFileItem) {
  const mime = String(item?.mime_type || '').toLowerCase()
  const nameOrUrl = String(item?.file_name || item?.url || '').toLowerCase()
  if (mimeOrExtMatchesAccept(mime, nameOrUrl)) return true

  // 当调用方传的是 image/* / video/* 等通配符，而历史数据缺少 mime_type 时，按常见后缀兜底。
  const ext = extFromName(nameOrUrl)
  const rules = parseAcceptRules(accept.value)
  return rules.some((rule) => {
    if (rule === 'image/*') return imageExts.has(ext)
    if (rule === 'video/*') return videoExts.has(ext)
    if (rule === 'audio/*') return audioExts.has(ext)
    if (rule === 'application/*') return docExts.has(ext) || archiveExts.has(ext)
    if (rule === 'text/*') return docExts.has(ext)
    return false
  })
}

async function switchSource(next: 'creations' | 'uploads') {
  if (history.source === next) return
  history.source = next
  history.offset = 0
  await loadHistory(0)
}

async function loadHistory(offset = 0) {
  history.loading = true
  history.error = ''
  try {
    const token = getAccessToken()
    const endpoint = history.source === 'uploads' ? '/v1/uploads' : '/v1/user/files'
    const params: any =
      history.source === 'uploads'
        ? { limit: history.limit, offset, keyword: history.keyword || undefined }
        : { category: category.value, limit: history.limit, offset, keyword: history.keyword || undefined }

    const resp = await axios.get(endpoint, {
      baseURL: getBackendRequestBaseURL(),
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      params,
    })
    // 后端现在直接返回绝对 URL，前端无需再拼接
    const rawItems = (resp?.data?.data?.items || []) as UserFileItem[]
    // client-side filter by accept (uploads list may contain mixed types)
    history.items = rawItems.filter((it) => looksAllowedUploadItem(it))
    history.total = Number(resp?.data?.data?.total || 0)
    history.offset = offset
  } catch (e: any) {
    history.error = e?.response?.data?.detail || e?.message || t('components.upload.loadFailed')
  } finally {
    history.loading = false
  }
}

async function openHistory() {
  history.open = true
  history.source = 'creations'
  await loadHistory(0)
}

async function openUploads() {
  history.open = true
  history.source = 'uploads'
  await loadHistory(0)
}

function selectHistory(item: UserFileItem) {
  if (!item?.url) return
  if (internal.value.length >= max.value) return
  if (internal.value.includes(item.url)) return
  internal.value = [...internal.value, item.url].slice(0, max.value)
  // 选择一张后立即关闭弹窗（用户可再次打开继续添加下一张）
  history.open = false
}

onBeforeUnmount(() => {
  addMenuListeners(false)
  if (snackTimer) {
    window.clearTimeout(snackTimer)
    snackTimer = undefined
  }
})
</script>


