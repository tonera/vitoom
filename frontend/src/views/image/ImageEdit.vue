<template>
  <div
    class="vt-surface vt-text-smooth h-full flex flex-col overflow-hidden antialiased text-gray-950 [.dark_&]:text-gray-100"
  >
    <div class="flex-1 flex gap-6 overflow-hidden min-h-0">
      <!-- 左侧：参数配置区 -->
      <div class="w-96 shrink-0 flex flex-col overflow-hidden min-h-0">
        <div class="rounded-lg flex-1 flex flex-col overflow-hidden min-h-0 pr-2">
          <form @submit.prevent="handleEdit" class="flex-1 flex flex-col min-h-0">
            <div class="flex-1 min-h-0 overflow-y-auto vt-scroll space-y-4 p-1">
              <!-- 图片选择/上传 -->
              <DropUploadImage v-model="tplList" :max="9"/>

              <!-- 提示词 -->
              <div
                class="rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-2 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35"
              >
                <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">
                  {{ t('image.edit.promptRequired') }} <span class="text-red-400">*</span>
                </label>
                <PromptTextarea
                  v-model="form.prompt"
                  :rows="4"
                  :placeholder="t('image.edit.promptPlaceholder')"
                  :show-char-count="true"
                />
              </div>

              <!-- 模型选择卡片 + CFG/Steps/Sampler（三列同一行） -->
              <div
                class="rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-3 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35"
              >
                <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">
                  {{ t('image.model') }} <span class="text-red-400">*</span>
                </label>
                <ModelSelectorV2
                  v-model="form.modelKey"
                  :models="modelOptions"
                  :total-count="modelTotal"
                  :meta="modelMeta"
                  @page-change="handleModelPageChange"
                  @search-change="handleModelSearchChange"
                  @filter-change="handleModelFilterChange"
                  ck_point="checkpoint"
                  mode="single"
                  variant="compact"
                />
                <p v-if="modelLoading" class="vt-help">{{ t('image.loadingModels') }}</p>

                <ModelSelectorV2
                  v-model="form.lora"
                  :family="selectedCheckpointFamily"
                  :models="loraModelOptions"
                  :total-count="loraTotal"
                  :meta="loraMeta"
                  @page-change="handleLoraPageChange"
                  @search-change="handleLoraSearchChange"
                  @filter-change="handleLoraFilterChange"
                  ck_point="lora"
                  mode="multiple"
                  variant="compact"
                />
                <p v-if="loraLoading" class="vt-help">{{ t('image.loadingLoras') }}</p>

                <div class="pt-1">
                  <div class="grid grid-cols-3 gap-3">
                    <div>
                      <RangeInput
                        v-model="form.guidanceScale"
                        label="CFG"
                        :min="0"
                        :max="20"
                        :step="0.5"
                        :clamp="true"
                        :round-to-step="true"
                      />
                    </div>
                    <div>
                      <RangeInput
                        v-model="form.numInferenceSteps"
                        label="Steps"
                        :min="1"
                        :max="100"
                        :step="1"
                        :clamp="true"
                        :round-to-step="true"
                      />
                    </div>
                    <div>
                      <UpPanel
                        v-model="form.schedulerName"
                        :label="t('image.sampler')"
                        :options="samplerOptions"
                      />
                    </div>
                  </div>
                </div>

                

              </div>

            <!-- 参数（宽高比 / 分辨率 / 输出数量） -->
            <div
              class="rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-3 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35"
            >
              <div class="grid grid-cols-3 gap-4">
                <div class="space-y-2">
                  <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">
                    {{ t('image.aspectRatio') }} <span class="text-red-400">*</span>
                  </label>
                  <AspectRatioSelector
                    name="aspectRatio"
                    v-model="form.aspectRatio"
                    :init="aspectRatioOptions"
                    :label="t('image.aspectRatio')"
                    :mode="'aspect'"
                    :def="form.aspectRatio || '1:1'"
                  />
                </div>
                <div class="space-y-2">
                  <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">
                    {{ t('image.resolution') }} <span class="text-red-400">*</span>
                  </label>
                  <AspectRatioSelector
                    name="resolution"
                    v-model="form.resolution"
                    :init="resolutionOptions"
                    :label="t('image.resolution')"
                    :mode="'panel'"
                    :def="form.resolution || resolutionOptions[0]?.value"
                  />
                </div>
                <div class="space-y-2">
                  <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">
                    {{ t('image.outputCount') }} <span class="text-red-400">*</span>
                  </label>
                  <AspectRatioSelector
                    name="numImages"
                    v-model="form.numImages"
                    :init="numImagesOptions"
                    :label="t('image.outputCount')"
                    :mode="'panel'"
                    :def="form.numImages || 1"
                  />
                </div>
              </div>
            </div>

            </div>

            <!-- 底部固定栏：高级按钮 + 开始编辑按钮并排 -->
            <div class="shrink-0 pt-3 p-1">
              <div
                class="rounded-2xl border border-gray-200 bg-white/85 backdrop-blur p-3 shadow-lg [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-900/70 [.dark_&]:shadow-xl"
              >
                <div class="flex items-center gap-3">
                  <button
                    ref="advancedBtnRef"
                    type="button"
                    class="w-11 h-11 shrink-0 rounded-xl border border-gray-200 bg-white text-gray-500 hover:text-gray-950 hover:bg-gray-50 transition-colors cursor-pointer flex items-center justify-center [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-900/40 [.dark_&]:text-white/60 [.dark_&]:hover:text-white [.dark_&]:hover:bg-gray-900/60"
                    :title="t('image.advancedOptions')"
                    @click.stop="toggleAdvanced"
                  >
                    <svg class="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 16v-2m8-6h-2M6 12H4m13.657-6.343l-1.414 1.414M7.757 16.243l-1.414 1.414m0-11.314l1.414 1.414m10.486 10.486l1.414 1.414" />
                    </svg>
                  </button>

                  <button
                    type="submit"
                    :disabled="submitDisabled"
                    :title="submitDisabled ? submitDisabledReason : ''"
                    :aria-busy="(isSubmitting || isGenerating) ? 'true' : 'false'"
                    class="w-full rounded-2xl bg-gray-950 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-50 [.dark_&]:bg-white [.dark_&]:text-gray-950 [.dark_&]:hover:bg-gray-200 cursor-pointer"
                  >
                    <svg
                      v-if="isSubmitting"
                      class="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                      <path
                        class="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647A7.962 7.962 0 0112 20a7.962 7.962 0 01-2-.205V20z"
                      ></path>
                    </svg>
                    <span class="inline-flex items-center gap-2">
                      <span v-if="isGenerating && !isSubmitting" class="relative flex h-2 w-2">
                        <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-white/70 opacity-75"></span>
                        <span class="relative inline-flex rounded-full h-2 w-2 bg-white"></span>
                      </span>
                      <span>{{ submitButtonText }}</span>
                    </span>
                  </button>
                </div>
              </div>
            </div>

          </form>
        </div>
      </div>

      <!-- 右侧：预览区 -->
      <div class="flex-1 flex flex-col overflow-hidden min-w-0 p-1">
        <MediaGalleryStrip
          v-model:activeKey="previewActiveKey"
          :items="thumbQueue"
          :active-item="previewActiveItem"
          :loading="userFilesLoading"
          :error-text="userFilesError"
          :show-upload="true"
          :show-download="true"
          :can-load-more="true"
          @open="openMediaByKey"
          @upload="handlePreviewUpload"
          @download="handlePreviewDownload"
          @load-more="loadNextUserFiles"
        >
          <template #side>
            <div v-if="quickButtonsLoading" class="text-xs text-gray-500 py-2 [.dark_&]:text-gray-400">
              {{ t('image.edit.loadingQuickButtons') }}
            </div>
            <div v-else-if="quickButtons.length === 0" class="text-xs text-gray-500 py-2 [.dark_&]:text-gray-400">
              {{ t('image.edit.noQuickButtons') }}
            </div>
            <div v-else class="grid grid-cols-2 gap-3">
              <button
                v-for="btn in quickButtons"
                :key="btn.key"
                type="button"
                :disabled="Boolean(getQuickButtonDisabledReason(btn))"
                :title="getQuickButtonDisabledReason(btn) || ''"
                class="h-12 rounded-lg bg-emerald-600/90 hover:bg-emerald-600 text-gray-950 font-semibold border border-emerald-500/40 shadow-sm cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed [.dark_&]:text-white"
                @click="onQuickButton(btn)"
              >
                <span class="inline-flex items-center gap-2">
                  <span
                    v-if="quickButtonRunning[btn.key]"
                    class="w-3 h-3 rounded-full border-2 border-emerald-950/40 border-t-emerald-950 animate-spin [.dark_&]:border-white/40 [.dark_&]:border-t-white"
                    aria-hidden="true"
                  ></span>
                  <span>{{ quickButtonLabel(btn) }}</span>
                </span>
              </button>
            </div>
          </template>
        </MediaGalleryStrip>
      </div>
    </div>

    <MediaLightbox v-model:open="mediaOpen" v-model:activeKey="mediaActiveKey" :items="mediaItems" />

    <!-- 高级选项浮层：跟随底部按钮（与 ImageDesign 一致） -->
    <Teleport to="body">
      <div v-if="advancedOpen" class="fixed inset-0 z-9998 bg-black/20" @click="advancedOpen = false">
        <div
          ref="advancedPopoverRef"
          class="fixed w-[420px] max-w-[calc(100vw-24px)] max-h-[min(80vh,640px)] overflow-y-auto vt-scroll p-4 rounded-2xl border border-gray-200 bg-white/95 shadow-2xl backdrop-blur ring-2 ring-gray-200/60 [.dark_&]:border-indigo-400/45 [.dark_&]:bg-gray-900/95 [.dark_&]:ring-indigo-500/20"
          :style="advancedPopoverStyle"
          role="dialog"
          aria-modal="false"
          @click.stop
        >
          <div class="flex items-center justify-between gap-3">
            <div class="text-sm font-semibold text-gray-950 [.dark_&]:text-gray-100">{{ t('image.edit.advancedOptionsTitle') }}</div>
            <button
              type="button"
              class="w-8 h-8 rounded-lg border border-gray-200 bg-gray-50 hover:bg-gray-100 text-gray-600 hover:text-gray-950 flex items-center justify-center cursor-pointer [.dark_&]:border-white/10 [.dark_&]:bg-black/20 [.dark_&]:hover:bg-black/30 [.dark_&]:text-white"
              :title="t('common.close')"
              @click="advancedOpen = false"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div class="mt-3 border-t border-gray-200 [.dark_&]:border-white/10"></div>

          <div class="mt-4">
            <ImageAdvancedOptions
              v-model:negativePrompt="form.negativePrompt"
              v-model:seed="form.seed"
              v-model:seedMode="form.seedMode"
              v-model:removeBg="form.removeBg"
              v-model:faceEnhance="form.faceEnhance"
              v-model:lowVram="form.lowVram"
            />
          </div>
        </div>
      </div>
    </Teleport>
    
    <!-- 隐藏的文件输入，用于预览区域上传 -->
    <input
      ref="previewUploadInputRef"
      type="file"
      class="hidden"
      accept="image/*"
      @change="onPreviewUploadFile"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch, toRef } from 'vue'
import { useI18n } from 'vue-i18n'
import PromptTextarea from '../../components/PromptTextarea.vue'
import ModelSelectorV2 from '../../components/ModelSelectorV2.vue'
import AspectRatioSelector from '../../components/AspectRatioSelector.vue'
import { type GeneratedImageDetails } from '../../components/GeneratedImageCard.vue'
import MediaLightbox, { type MediaLightboxItem } from '../../components/MediaLightbox.vue'
import { useMediaLightbox } from '../../composables/useMediaLightbox'
import { useCheckpointModels, type UiModelOption } from '../../composables/useCheckpointModels'
import { useLoraModels } from '../../composables/useLoraModels'
import { useTaskFeed } from '../../composables/useTaskFeed'
import MediaGalleryStrip from '../../components/MediaGalleryStrip.vue'
import DropUploadImage from '../../components/DropUploadImage.vue'
import RangeInput from '../../components/RangeInput.vue'
import UpPanel from '../../components/UpPanel.vue'
import ImageAdvancedOptions from './components/ImageAdvancedOptions.vue'
import { buildEdImageRequest } from '../../modules/image/domain/imageTaskRequestFactory'
import { sdAspList } from '../../config/aspectRatios'
import { resolutionList } from '../../config/resolutions'
import { taskCreateCacheKeyByType, type TaskCreateRequest } from '../../utils/taskRunner'
import { getLocalCache } from '../../utils/localCache'
import { getRaw } from '../../utils/api'
import { useUserFilesGallery, type UserFileItem } from '../../composables/useUserFilesGallery'
import { uploadFile, isImageFile, downloadFile } from '../../utils/upload'
import { useTaskRunManager } from '../../composables/useTaskRunManager'
import { showTopSnack, showTopSnackError } from '../../composables/useTopSnack'
import { applyModelConfigDefaults } from '../../utils/modelConfig'
import { useAnchoredPopover } from '../../composables/useAnchoredPopover'
import { samplerOptions } from '../../config/samplers'

const { t, te } = useI18n()

const tplList = ref<string[]>([])

const form = reactive({
  prompt: '',
  modelKey: '' as string,
  aspectRatio: '1:1',
  resolution: 1 as number,
  numImages: 1 as number,
  guidanceScale: 3.5 as number,
  schedulerName: '' as string,
  numInferenceSteps: 25 as number,
  seedMode: 'random' as 'random' | 'custom',
  seed: -1 as number,
  negativePrompt: '',
  removeBg: false as boolean,
  faceEnhance: false as boolean,
  lowVram: false as boolean,
  lora: [] as Array<{ value: string; weight?: number; locked?: boolean }>,
  // UI 内部结构：loras 为数组 [{ name, weight, trigger_word }]；提交时会 JSON.stringify
  loras: [] as Array<{ name: string; weight: number; trigger_word?: string }>,
})

const {
  modelOptions,
  modelLoading,
  modelTotal,
  modelMeta,
  selectedModelExists,
  selectedCheckpointFamily,
  selectedModelConfig,
  fetchModels,
  handleModelPageChange,
  handleModelSearchChange,
  handleModelFilterChange,
  restoreCachedModel,
  persistSelectedModel,
} = useCheckpointModels({
  modelKey: toRef(form, 'modelKey'),
  onlyEditable: true,
  taskType: 'image',
  cacheKey: 'vitoom:model:last:checkpoint:image',
})
const {
  loraModelOptions,
  loraLoading,
  loraTotal,
  loraMeta,
  fetchLoras,
  handleLoraPageChange,
  handleLoraSearchChange,
  handleLoraFilterChange,
} = useLoraModels({
  baseFamily: selectedCheckpointFamily,
  selected: toRef(form, 'lora'),
  payload: toRef(form, 'loras'),
  defaultWeight: 0.5,
})

const aspectRatioOptions = sdAspList
const resolutionOptions = resolutionList
const numImagesOptions = Array.from({ length: 9 }, (_, i) => ({ value: i + 1, label: `${i + 1}` }))

const { feed, createRunHooks } = useTaskFeed<GeneratedImageDetails>({
  onInsertedKey: (key) => {
    previewActiveKey.value = key
  },
})
const loading = ref(false)

type PreviewThumbItem = {
  key: string
  kind: 'image' | 'video'
  thumbSrc: string
  originalSrc: string
  posterSrc?: string
  title?: string
  source: 'generated' | 'userfile'
}

const {
  items: userFiles,
  loading: userFilesLoading,
  error: userFilesError,
  load: loadUserFiles,
  loadNext: loadNextUserFiles,
  prepend: prependUserFile,
} = useUserFilesGallery({
  category: 'image',
  limit: 60,
  onLoaded: () => {
    // 初次加载：若当前没有选中项，默认选中第一张
    if (!previewActiveKey.value && thumbQueue.value.length > 0) {
      previewActiveKey.value = thumbQueue.value[0]!.key
    }
  },
})

const previewActiveKey = ref<string | null>(null)
const previewUploadInputRef = ref<HTMLInputElement | null>(null)

const thumbQueue = computed<PreviewThumbItem[]>(() => {
  const out: PreviewThumbItem[] = []
  const seen = new Set<string>()
  const push = (it: PreviewThumbItem) => {
    if (!it?.key) return
    if (seen.has(it.key)) return
    seen.add(it.key)
    out.push(it)
  }

  // 生成结果：始终放在最前（feed 本身是 newest-first）
  for (const x of feed.value) {
    if (x.kind !== 'image' && x.kind !== 'video') continue
    push({
      key: x.key,
      kind: x.kind,
      thumbSrc: x.thumbSrc,
      originalSrc: x.originalSrc || x.thumbSrc,
      posterSrc: x.kind === 'video' ? x.posterSrc || x.thumbSrc : undefined,
      title: x.title,
      source: 'generated',
    })
  }

  // 作品集（/v1/user/files）
  for (const f of userFiles.value) {
    const url = String(f?.url || '')
    if (!url) continue
    const mime = String(f?.mime_type || '').toLowerCase()
    const isVideo = mime.startsWith('video/')
    push({
      key: url,
      kind: isVideo ? 'video' : 'image',
      thumbSrc: String(f?.thumb_url || url),
      originalSrc: url,
      posterSrc: isVideo ? String(f?.thumb_url || url) : undefined,
      title: f?.file_name ? String(f.file_name) : undefined,
      source: 'userfile',
    })
  }

  return out
})

const previewActiveItem = computed(() => {
  if (!thumbQueue.value.length) return null
  const byKey = previewActiveKey.value ? thumbQueue.value.find((x) => x.key === previewActiveKey.value) : undefined
  return byKey || thumbQueue.value[0]!
})

const mediaItems = computed<MediaLightboxItem[]>(() =>
  thumbQueue.value.map((x) => ({
    key: x.key,
    type: x.kind,
    src: x.originalSrc,
    title: x.title,
    poster: x.kind === 'video' ? x.posterSrc || x.thumbSrc : undefined,
  }))
)

const { mediaOpen, mediaActiveKey, openMediaByKey } = useMediaLightbox(mediaItems)


type QuickButtonConfig = {
  key: string
  url?: string
  job_type: string
  prompt?: string
  load_name?: string
  upscale?: number
  label: string
  keep_size?: string
}

/** 旧版 quickButton.json 仅含中文 label 时的 key 映射 */
const LEGACY_QUICK_BUTTON_LABEL_KEYS: Record<string, string> = {
  转线稿: 'lineArt',
  线稿上色: 'colorizeLineArt',
  人脸修复: 'faceRestore',
  吉卜力: 'ghibli',
  新海诚: 'shinkai',
  移除背景: 'removeBg',
  背景虚化: 'bgBlur',
  纯色背景: 'solidBg',
  移除杂物: 'removeClutter',
  人像美化: 'portraitRetouch',
  补光提亮: 'brighten',
  色彩增强: 'colorEnhance',
  水彩风: 'watercolor',
  赛博朋克: 'cyberpunk',
  油画风: 'oilPainting',
  铅笔素描: 'pencilSketch',
  '3D手办': 'figurine3d',
  胶片电影感: 'filmLook',
  夜景增强: 'nightEnhance',
  重绘: 'redraw',
  高清: 'upscaleHd',
  移除文字: 'removeText',
  移除水印: 'removeWatermark',
  一键移除: 'quickRemove',
}

function resolveQuickButtonKey(rawKey?: string, label?: string): string {
  const k = String(rawKey || '').trim()
  if (k) return k
  const lab = String(label || '').trim()
  return LEGACY_QUICK_BUTTON_LABEL_KEYS[lab] || lab
}

function quickButtonLabel(btn: QuickButtonConfig): string {
  const i18nKey = `image.edit.quickButtons.${btn.key}`
  return te(i18nKey) ? t(i18nKey) : btn.label
}

const quickButtons = ref<QuickButtonConfig[]>([])
const quickButtonsLoading = ref(false)
const quickButtonRunning = reactive<Record<string, boolean>>({})

const defaultQuickButtons: QuickButtonConfig[] = [
]

async function loadQuickButtons() {
  quickButtonsLoading.value = true
  try {
    const resp = await fetch('/quickButton.json', { cache: 'no-store' })
    const json = await resp.json().catch(() => null)
    const arr = Array.isArray(json) ? (json as any[]) : null
    if (!arr) {
      quickButtons.value = [...defaultQuickButtons]
      return
    }

    const normalized: QuickButtonConfig[] = arr
      .map((x) => {
        const label = String(x?.label || '').trim()
        const key = resolveQuickButtonKey(x?.key, label)
        return {
          key,
          url: x?.url ? String(x.url) : undefined,
          job_type: String(x?.job_type || '').trim(),
          prompt: x?.prompt !== undefined ? String(x.prompt) : undefined,
          load_name: x?.load_name !== undefined ? String(x.load_name) : undefined,
          upscale: x?.upscale !== undefined ? Number(x.upscale) : undefined,
          label,
          keep_size: x?.keep_size !== undefined ? String(x.keep_size) : undefined,
        }
      })
      .filter((x) => x.key && x.job_type)

    quickButtons.value = normalized.length ? normalized : [...defaultQuickButtons]
  } catch {
    quickButtons.value = [...defaultQuickButtons]
  } finally {
    quickButtonsLoading.value = false
  }
}

function getQuickButtonDisabledReason(btn: QuickButtonConfig): string {
  const cur = previewActiveItem.value
  if (!cur) return t('image.edit.selectImageFirst')
  if (cur.kind !== 'image') return t('image.edit.videoNotSupported')
  if (quickButtonRunning[btn.key]) return t('common.processing')
  return ''
}

function resolveModelByLoadNameOrSelected(loadName?: string) {
  const name = String(loadName || '').trim()
  if (!name) return modelOptions.value.find((m) => m.value === form.modelKey)
  return (
    modelOptions.value.find((m) => String(m.load_name || '').trim() === name) ||
    modelOptions.value.find((m) => String(m.name || '').trim() === name) ||
    modelOptions.value.find((m) => String(m.label || '').trim() === name)
  )
}

/** 分页里没有该 model_key 时按 model_key 拉取单模型并插入 options（与 useCheckpointModels 可编辑页合并逻辑一致） */
async function hydrateEditableModelFromServerByKey(modelKey: string): Promise<UiModelOption | null> {
  const mid = String(modelKey || '').trim()
  if (!mid) return null
  try {
    const resp = await getRaw<any>(`/models/${encodeURIComponent(mid)}`)
    const md = resp?.data
    if (!md || !Boolean(md?.capabilities?.editable)) return null
    const opt: UiModelOption = {
      value: String(md?.model_key || mid),
      label: String(md?.name || md?.load_name || md?.model_key || mid),
      name: String(md?.name || md?.load_name || md?.model_key || mid),
      load_name: md?.load_name ? String(md.load_name) : undefined,
      thumb: (md?.thumb || '') as string,
      storage_mode: md?.storage_mode ? String(md.storage_mode) : '',
      family: md?.family ? String(md.family) : '',
      asset_type: String(md?.asset_type || 'checkpoint'),
      capabilities: { ...(md?.capabilities || {}), editable: true },
      runtime_config: (md as any)?.runtime_config ?? undefined,
      video_profile:
        md?.video_profile ??
        md?.runtime_config?.video_profile ??
        undefined,
    }
    if (!modelOptions.value.some((m) => m.value === opt.value)) {
      modelOptions.value = [opt, ...modelOptions.value]
    }
    return opt
  } catch {
    return null
  }
}

async function applyTaskCacheCheckpointSelection(cached: TaskCreateRequest) {
  const id = typeof cached.model_key === 'string' ? cached.model_key.trim() : ''
  const rawLoadName = String((cached as any).load_name ?? '').trim()

  if (id && !modelOptions.value.some((m) => m.value === id)) {
    const hydrated = await hydrateEditableModelFromServerByKey(id)
    if (hydrated) persistSelectedModel(hydrated)
  }

  const picked =
    (id ? modelOptions.value.find((m) => m.value === id) : undefined) ||
    (rawLoadName ? resolveModelByLoadNameOrSelected(rawLoadName) : undefined) ||
    undefined

  if (picked?.value) {
    form.modelKey = picked.value
    persistSelectedModel(picked)
  }
}

async function onQuickButton(btn: QuickButtonConfig) {
  const reason = getQuickButtonDisabledReason(btn)
  if (reason) {
    showTopSnack(reason)
    return
  }
  const cur = previewActiveItem.value!
  const curUrl = normalizeImageUrl(cur.originalSrc || cur.thumbSrc)

  const btnLabel = quickButtonLabel(btn)

  // mark running
  quickButtonRunning[btn.key] = true

  try {
    const jobType = String(btn.job_type || '').trim() || 'ED'
    const base = getAspectBase(form.aspectRatio)
    const width = Math.min(4096, Math.max(64, base.width))
    const height = Math.min(4096, Math.max(64, base.height))

    const upscaleRaw = Number(btn.upscale ?? 1) || 1
    const upscale = upscaleRaw === 1 || upscaleRaw === 2 || upscaleRaw === 4 ? upscaleRaw : 1
    // 如果有keep_size，则将keep_size传入
    const keepSize = btn.keep_size ? btn.keep_size : undefined

    // model is required for ED-like tasks; RBG/SR do not require diffusion models.
    const needsModel = !(jobType === 'RBG' || jobType === 'SR')
    const model = needsModel ? resolveModelByLoadNameOrSelected(btn.load_name) : undefined
    if (needsModel && !model?.value) {
      showTopSnack(btn.load_name ? t('image.edit.modelNotFound', { name: btn.load_name }) : t('image.edit.selectModelFirst'))
      quickButtonRunning[btn.key] = false
      return
    }

    const req: TaskCreateRequest = {
      task_type: 'image',
      job_type: jobType,
      prompt: (btn.prompt ?? '') as any,
      width,
      height,
      generate_num: 1,
      model_key: needsModel ? model?.value : undefined,
      load_name: needsModel ? model?.load_name : undefined,
      family: needsModel ? model?.family : undefined,
      fast_mode: true,
      upscale,
      keep_size: keepSize,
      // Backend expects tpl_list for ED/RBG/SR; also keep url for potential future job types (e.g. FS).
      tpl_list: [curUrl],
      url: curUrl as any,
    }

    const hooks = createRunHooks({
      title: { image: btnLabel, video: btnLabel },
      fallbackDownloadName: { image: 'quick-image.png', video: 'quick-video.mp4' },
      buildDetails: ({ file, taskId, req: runReq }) => {
        return {
          prompt: (runReq as any).prompt ?? '',
          taskId,
          fileName: (file as any)?.file_name,
        } satisfies GeneratedImageDetails
      },
      onError: (err) => {
        if (err.phase === 'create') {
          showTopSnackError(t('image.edit.quickButtonFailed', { label: btnLabel, msg: err.message }))
          quickButtonRunning[btn.key] = false
          return
        }
        showTopSnackError(err.message)
      },
    })
    await runTask(req, 1, {
      ...hooks,
      onAddPlaceholders: () => {
        // no placeholder UI for quick buttons (button itself shows loading)
      },
      onRemoveKeys: () => {
        // no placeholder UI
      },
      onTerminal: () => {
        quickButtonRunning[btn.key] = false
      },
    })
  } catch (e: any) {
    const msg = e?.response?.data?.detail || e?.message || String(e)
    showTopSnackError(t('image.edit.quickButtonFailed', { label: btnLabel, msg }))
    quickButtonRunning[btn.key] = false
  }
}

const { generatingCount, runTask, disconnectAll } = useTaskRunManager()

// 高级选项浮层（Popover）
const {
  open: advancedOpen,
  anchorRef: advancedBtnRef,
  popoverRef: advancedPopoverRef,
  style: advancedPopoverStyle,
  toggle: toggleAdvanced,
} = useAnchoredPopover()
// NOTE: template refs (ref="advancedBtnRef") are not counted by TS noUnusedLocals
void advancedBtnRef
void advancedPopoverRef

const trimmedPrompt = computed(() => String(form.prompt || '').trim())

const isSubmitting = computed(() => loading.value)
const isGenerating = computed(() => generatingCount.value > 0)
const submitDisabledReason = computed(() => {
  if (isSubmitting.value) return t('common.submittingTask')
  if (isGenerating.value) return t('common.processingWait')
  if (!tplList.value.length) return t('image.edit.selectImageFirst')
  if (!trimmedPrompt.value) return t('common.enterPrompt')
  if (!form.modelKey) return modelLoading.value ? t('common.modelsLoading') : t('common.selectModel')
  if (!modelLoading.value && modelOptions.value.length === 0) return t('common.noModelsAvailable')
  if (modelOptions.value.length > 0 && !selectedModelExists.value) return t('common.modelUnavailable')
  if (!form.aspectRatio) return t('common.selectAspectRatio')
  if (!form.resolution) return t('common.selectResolution')
  return ''
})
const submitDisabled = computed(() => Boolean(submitDisabledReason.value))
const submitButtonText = computed(() => {
  if (isSubmitting.value) return t('common.submitting')
  if (isGenerating.value) return t('common.processing')
  return t('image.edit.startEdit')
})

// feed operations are handled by useTaskFeed()

const getAspectBase = (val: string) => {
  const found = sdAspList.find((x) => x.val === val)
  return found || sdAspList[0]!
}

/**
 * 将图片 URL 转换为绝对 URL
 * 如果已经是绝对 URL（以 http:// 或 https:// 开头），则保持不变
 * 如果是相对路径（以 /outputs/ 开头），则拼接为 http://host:port/outputs/...
 */
const normalizeImageUrl = (url: string): string => {
  if (!url) return url
  // 如果已经是绝对 URL，直接返回
  if (url.startsWith('http://') || url.startsWith('https://')) {
    return url
  }
  // 如果是相对路径，拼接为绝对 URL
  const origin = window.location.origin
  // 确保路径以 / 开头
  const path = url.startsWith('/') ? url : `/${url}`
  return `${origin}${path}`
}

// 切换模型：仅当 model_config 明确提供字段时覆盖（规则同 ImageDesign）
watch(
  () => form.modelKey,
  () => {
    applyModelConfigDefaults(selectedModelConfig.value, {
      setGuidanceScale: (v) => (form.guidanceScale = v),
      setNumInferenceSteps: (v) => (form.numInferenceSteps = v),
      setSchedulerName: (v) => (form.schedulerName = v),
    })
  }
)

const handleEdit = async () => {
  if (submitDisabled.value) {
    showTopSnack(submitDisabledReason.value || t('common.cannotSubmit'))
    return
  }

  loading.value = true
  try {
    const want = Math.max(1, Math.min(9, Number(form.numImages || 1)))

    const base = getAspectBase(form.aspectRatio)
    const upscaleRaw = Number(form.resolution) || 1
    const upscale = upscaleRaw === 1 || upscaleRaw === 2 || upscaleRaw === 4 ? upscaleRaw : 1
    const width = Math.min(4096, Math.max(64, base.width))
    const height = Math.min(4096, Math.max(64, base.height))

    const selectedModel = modelOptions.value.find((m) => m.value === form.modelKey)
    const guidanceScaleRaw = Number(form.guidanceScale)
    const guidanceScale = Number.isFinite(guidanceScaleRaw) ? guidanceScaleRaw : 7.5
    // 将 tplList 中的每个 URL 转换为绝对 URL
    const normalizedTplList = tplList.value.map((url) => normalizeImageUrl(url))
    const req = buildEdImageRequest({
      prompt: trimmedPrompt.value,
      negativePrompt: form.negativePrompt || '',
      width,
      height,
      generateNum: want,
      modelKey: form.modelKey || undefined,
      loadName: selectedModel?.load_name,
      family: selectedModel?.family,
      // guidanceScale 允许为 0；仅当 NaN/非有限值时才回退默认
      guidanceScale,
      numInferenceSteps: Number(form.numInferenceSteps) || 30,
      schedulerName: String(form.schedulerName || ''),
      // 约定：-1 表示随机 seed（与 build*ImageRequest 的约束保持一致）
      seed: form.seedMode === 'custom' ? Math.max(1, Math.floor(Number(form.seed) || 1)) : -1,
      removeBg: Boolean(form.removeBg),
      faceEnhance: Boolean(form.faceEnhance),
      upscale,
      tplList: normalizedTplList,
      lorasPayload: form.loras,
    }) satisfies TaskCreateRequest

    const { taskId } = await runTask(
      req,
      want,
      createRunHooks({
        title: { image: t('image.edit.title'), video: t('common.generateVideo') },
        fallbackDownloadName: { image: 'edited-image.jpeg', video: 'edited-video.mp4' },
        buildDetails: ({ file, taskId, req: runReq }) => {
          return {
            prompt: (runReq as any).prompt ?? '',
            taskId,
            fileName: (file as any)?.file_name,
          } satisfies GeneratedImageDetails
        },
        onError: (err) => {
          if (err.phase === 'create') showTopSnackError(t('common.generateFailed', { msg: err.message }))
          else showTopSnackError(err.message)
        },
      })
    )
    if (taskId && selectedModel) {
      persistSelectedModel(selectedModel)
    }
  } catch (error: any) {
    const msg = error?.response?.data?.detail || error?.message || String(error)
    showTopSnackError(t('common.generateFailed', { msg }))
  } finally {
    loading.value = false
  }
}

onBeforeUnmount(() => {
  disconnectAll()
})

onMounted(async () => {
  const cached = getLocalCache<TaskCreateRequest>(taskCreateCacheKeyByType('image'))
  if (cached) {
    if (typeof cached.prompt === 'string') form.prompt = cached.prompt
    if (typeof cached.model_key === 'string') form.modelKey = cached.model_key
    if (typeof (cached as any).negative_prompt === 'string') form.negativePrompt = String((cached as any).negative_prompt || '')
    if ((cached as any).guidance_scale !== undefined) {
      // guidanceScale 允许为 0；仅当值为 NaN 时才回退默认
      const v = Number((cached as any).guidance_scale)
      if (!Number.isNaN(v)) form.guidanceScale = v
    }
    if ((cached as any).schedulerName !== undefined) form.schedulerName = String((cached as any).schedulerName || '')
    if ((cached as any).num_inference_steps !== undefined) form.numInferenceSteps = Number((cached as any).num_inference_steps) || form.numInferenceSteps

    const seed = Number((cached as any).seed || 0)
    if (seed > 0) {
      form.seedMode = 'custom'
      form.seed = seed
    } else {
      form.seedMode = 'random'
      form.seed = seed === 0 ? -1 : seed
    }
    if ((cached as any).remove_bg !== undefined) form.removeBg = Boolean((cached as any).remove_bg)
    if ((cached as any).face_enhance !== undefined) form.faceEnhance = Boolean((cached as any).face_enhance)
    const gen = Number((cached as any).generate_num)
    if (Number.isFinite(gen) && gen >= 1) form.numImages = Math.max(1, Math.min(9, Math.floor(gen)))
    const upscale = Number((cached as any).upscale)
    if (upscale === 1 || upscale === 2 || upscale === 4) form.resolution = upscale
    const w = Number((cached as any).width)
    const h = Number((cached as any).height)
    if (Number.isFinite(w) && Number.isFinite(h)) {
      const match = sdAspList.find((x) => Number(x.width) === w && Number(x.height) === h)
      if (match) form.aspectRatio = match.val
    }
    const cachedTpl = (cached as any).tpl_list
    if (Array.isArray(cachedTpl) && cachedTpl.length) {
      tplList.value = cachedTpl.map((x: unknown) => String(x).trim()).filter(Boolean).slice(0, 9)
    }
  }

  /**
   * 任务创建缓存里若已有 model_key / load_name（如 Remix 写入的 task_params），应优先用之；
   * 否则会先 restoreCachedModel 覆盖 form.modelKey，且从未按 load_name 解析。
   */
  const hasTaskCacheModelHint =
    !!cached &&
    ((typeof cached.model_key === 'string' && cached.model_key.trim() !== '') ||
      (typeof (cached as any).load_name === 'string' && String((cached as any).load_name).trim() !== ''))

  if (!hasTaskCacheModelHint) {
    restoreCachedModel()
  }

  await fetchModels()

  if (cached) {
    await applyTaskCacheCheckpointSelection(cached)
  }

  fetchLoras()
  loadUserFiles(true)
  loadQuickButtons()
})

// 由 MediaGalleryStrip 触发 load-more，这里保留 loadNextUserFiles()

// 预览区域上传按钮点击
function handlePreviewUpload() {
  previewUploadInputRef.value?.click()
}

// 预览区域上传文件处理
async function onPreviewUploadFile(e: Event) {
  const input = e.target as HTMLInputElement
  const files = Array.from(input.files || [])
  input.value = ''
  if (!files.length) return
  
  const file = files[0]
  if (!file || !isImageFile(file)) {
    showTopSnack(t('image.edit.selectImageFirst'))
    return
  }
  
  try {
    const url = await uploadFile(file)
    
    // 将上传的图片添加到userFiles，使其进入thumbQueue
    const newFile: UserFileItem = {
      id: `upload-${Date.now()}`,
      url: url,
      thumb_url: url,
      file_name: file.name,
      mime_type: file.type,
      file_size: file.size,
      created_at: new Date().toISOString(),
    }
    
    // 添加到作品集队列开头，使其显示在最前面（并去重）
    prependUserFile(newFile)
    
    // 设置为当前选中的预览项
    previewActiveKey.value = url
    
    // 同时更新tplList，使其可以被右侧快捷编辑按钮编辑
    if (!tplList.value.includes(url)) {
      tplList.value = [url, ...tplList.value].slice(0, 9)
    }
    
    showTopSnack(t('common.uploadSuccess'))
  } catch (error: any) {
    const msg = error?.response?.data?.detail || error?.message || String(error)
    showTopSnackError(t('common.uploadFailed', { msg }))
  }
}

// 预览区域下载按钮点击
async function handlePreviewDownload() {
  const item = previewActiveItem.value
  if (!item) {
    showTopSnack(t('common.noFileToDownload'))
    return
  }
  
  const url = item.originalSrc || item.thumbSrc
  if (!url) {
    showTopSnack(t('common.invalidFileUrl'))
    return
  }
  
  showTopSnack(t('common.startDownload'))
  const success = await downloadFile({
    url,
    mediaType: item.kind,
    title: item.title,
  })
  
  if (success) {
    showTopSnack(t('common.downloadSuccess'))
  } else {
    showTopSnackError(t('common.downloadFailed'))
  }
}
</script>
