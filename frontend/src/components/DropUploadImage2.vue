<template>
  <div :class="[variant === 'plain' ? 'space-y-3' : 'vt-card-muted p-4 space-y-3', fill ? 'h-full flex flex-col' : '']">
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
        class="relative rounded-xl overflow-hidden border border-gray-700/60 bg-gray-900/30"
        :class="fill && max === 1 ? 'h-full min-h-0' : 'aspect-square'"
      >
        <img :src="url" class="w-full h-full object-cover" />
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
          class="w-full rounded-xl border border-dashed border-gray-700/70 bg-gray-900/20 hover:bg-gray-900/30 flex flex-col items-center justify-center gap-2 text-xs leading-none text-gray-400 cursor-pointer"
          :class="fill && max === 1 ? 'h-full min-h-0' : 'aspect-square'"
          @click="toggleMenu()"
        >
          <div class="w-8 h-8 rounded-lg bg-gray-800/70 border border-gray-700 flex items-center justify-center text-gray-200">
            +
          </div>
          <div class="text-gray-300 leading-none">{{ label }}</div>
        </button>

        <Teleport to="body">
          <div
            v-if="menuOpen"
            ref="menuRef"
            class="fixed z-9999 w-64 rounded-xl border border-gray-700 bg-gray-900/95 text-gray-200 shadow-2xl backdrop-blur p-2"
            :style="{ left: menuLeft + 'px', top: menuTop + 'px' }"
          >
            <button
              type="button"
              class="w-full flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-800/80 text-sm cursor-pointer"
              @click="handlePickFromCreations"
            >
              <span class="w-5 text-center">🧰</span>
              {{ t('components.upload.fromAssets') }}
            </button>

            <button
              type="button"
              class="w-full flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-800/80 text-sm cursor-pointer"
              @click="handlePickFromUploads"
            >
              <span class="w-5 text-center">📤</span>
              {{ t('components.upload.fromUploads') }}
            </button>
       
            <button
              type="button"
              class="w-full flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-gray-800/80 text-sm cursor-pointer"
              @click="handleUpload"
            >
              <span class="w-5 text-center">🖼️</span>
              {{ t('components.upload.uploadImage') }}
            </button>
       
          </div>
        </Teleport>
      </div>
    </div>

    <input
      ref="fileInputRef"
      type="file"
      class="hidden"
      accept="image/*"
      multiple
      @change="onPickFiles"
    />



  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import axios from 'axios'
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

const props = defineProps<{
  modelValue: string[]
  max?: number
  label?: string
  variant?: 'card' | 'plain'
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
}>()
const emit = defineEmits<{
  (e: 'update:modelValue', v: string[]): void
}>()

const { t } = useI18n()

const max = computed(() => Math.max(1, Math.min(9, Number(props.max ?? 9))))
const label = computed(() => String(props.label || t('components.upload.referenceImage')))
const variant = computed(() => (props.variant === 'plain' ? 'plain' : 'card'))
const fill = computed(() => Boolean(props.fill))
const fillMinHeightPx = computed(() => {
  const v = Number(props.fillMinHeight ?? 120)
  if (!Number.isFinite(v)) return 120
  return Math.max(0, Math.floor(v))
})

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
  const url = await uploadFile(file)
  internal.value = [...internal.value, url].slice(0, max.value)
}

// ---- dropdown menu (集成入口) ----
const anchorRef = ref<HTMLElement | null>(null)
const menuRef = ref<HTMLElement | null>(null)
const menuOpen = ref(false)
const menuLeft = ref(0)
const menuTop = ref(0)

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
  const files = Array.from(e.dataTransfer?.files || []).filter((f) => (f.type || '').startsWith('image/'))
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

async function loadHistory(offset = 0) {
  history.loading = true
  history.error = ''
  try {
    const token = getAccessToken()
    const endpoint = history.source === 'uploads' ? '/v1/uploads' : '/v1/user/files'
    const params: any =
      history.source === 'uploads'
        ? { limit: history.limit, offset, keyword: history.keyword || undefined }
        : { category: 'image', limit: history.limit, offset, keyword: history.keyword || undefined }

    const resp = await axios.get(endpoint, {
      baseURL: getBackendRequestBaseURL(),
      headers: {
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      params,
    })
    // 后端现在直接返回绝对 URL，前端无需再拼接
    history.items = (resp?.data?.data?.items || []) as UserFileItem[]
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

onBeforeUnmount(() => {
  addMenuListeners(false)
})
</script>


