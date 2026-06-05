<template>
  <div class="h-full flex flex-col overflow-hidden text-gray-950 [.dark_&]:text-gray-100">

    <!-- 主要内容区域：左右布局 -->
    <div class="flex-1 flex gap-6 overflow-hidden min-h-0">
      <!-- 左侧：参数配置区 -->
      <div class="w-96 shrink-0 flex flex-col overflow-hidden min-h-0">
        <div class="rounded-lg flex-1 flex flex-col overflow-hidden min-h-0 pr-2">
          <form @submit.prevent="handleGenerate" class="flex-1 flex flex-col min-h-0">
            <div class="flex-1 min-h-0 overflow-y-auto vt-scroll space-y-4 p-1">
              <!-- 提示词卡片 -->
              <div class="rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-2 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35">
                <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">
                  {{ t('image.prompt') }} <span class="text-red-400">*</span>
                </label>
                <PromptTextarea
                  v-model="form.prompt"
                  :rows="4"
                  :placeholder="t('image.promptPlaceholder')"
                  :show-char-count="true"
                />
              </div>

            

              <!-- 模型选择卡片 + CFG/Steps/Sampler（三列同一行） -->
              <div class="rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-3 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35">
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

                <!-- 三列参数：去掉整体外框（按你标注的红线要求） -->
                <div class="pt-1">
                  <div class="grid grid-cols-3 gap-3">
                    <!-- CFG -->
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

                    <!-- Steps -->
                    <div>
                      <RangeInput
                        v-model="form.numInferenceSteps"
                        label="Steps"
                        :min="1"
                        :max="50"
                        :step="1"
                        :clamp="true"
                        :round-to-step="true"
                      />
                    </div>

                    <!-- Sampler -->
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



            <!-- 参数设置卡片 -->
            <div class="rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-3 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35">
              <div class="grid grid-cols-3 gap-4">
                <!-- 宽高比 -->
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
              <p class="vt-help pt-1">{{ t('image.outputCountHelp') }}</p>
            </div>
            </div>

            <!-- 底部固定栏：高级按钮 + 生成按钮并排 -->
            <div class="shrink-0 pt-3 p-1">
              <div class="rounded-2xl border border-gray-200 bg-white/85 backdrop-blur p-3 shadow-lg [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-900/70 [.dark_&]:shadow-xl">
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
                    <svg v-if="isSubmitting" class="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647A7.962 7.962 0 0112 20a7.962 7.962 0 01-2-.205V20z"></path>
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

      <!-- 右侧：图片预览区 -->
      <div class="flex-1 flex flex-col overflow-hidden min-w-0 p-1">
        <MediaGalleryGrid :items="feed" @open="openMediaByKey" />
      </div>
    </div>

    <MediaLightbox v-model:open="mediaOpen" v-model:activeKey="mediaActiveKey" :items="mediaItems" />

    <!-- 高级选项浮层：跟随底部按钮 --> 
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
            <div class="text-sm font-semibold text-gray-950 [.dark_&]:text-gray-100">{{ t('image.advancedOptions') }}</div>
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
              v-model:fastMode="form.fastMode"
            />
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onBeforeUnmount, onMounted, watch, toRef } from 'vue'
import { useI18n } from 'vue-i18n'
import PromptTextarea from '../../components/PromptTextarea.vue'
import ModelSelectorV2 from '../../components/ModelSelectorV2.vue'
import AspectRatioSelector from '../../components/AspectRatioSelector.vue'
import type { GeneratedImageDetails } from '../../components/GeneratedImageCard.vue'
import MediaLightbox, { type MediaLightboxItem } from '../../components/MediaLightbox.vue'
import { useMediaLightbox } from '../../composables/useMediaLightbox'
import { useCheckpointModels } from '../../composables/useCheckpointModels'
import { useLoraModels } from '../../composables/useLoraModels'
import { useTaskFeed } from '../../composables/useTaskFeed'
import type { TaskFeedItem } from '../../composables/useTaskFeed'
import MediaGalleryGrid from '../../components/MediaGalleryGrid.vue'
import { sdAspList } from '../../config/aspectRatios'
import { resolutionList } from '../../config/resolutions'
import ImageAdvancedOptions from './components/ImageAdvancedOptions.vue'
import RangeInput from '../../components/RangeInput.vue'
import UpPanel from '../../components/UpPanel.vue'
import { buildMkImageRequest } from '../../modules/image/domain/imageTaskRequestFactory'
import { taskCreateCacheKeyByType, type TaskCreateRequest } from '../../utils/taskRunner'
import { getLocalCache } from '../../utils/localCache'
import { useTaskRunManager } from '../../composables/useTaskRunManager'
import { showTopSnack, showTopSnackError } from '../../composables/useTopSnack'
import { applyModelConfigDefaults } from '../../utils/modelConfig'
import { useAnchoredPopover } from '../../composables/useAnchoredPopover'
import { samplerOptions } from '../../config/samplers'

const { t } = useI18n()

// 表单数据
const form = reactive({
  prompt: '',
  negativePrompt: '',
  guidanceScale: 3.5 as number,
  schedulerName: '' as string,
  numInferenceSteps: 25 as number,
  seedMode: 'random' as 'random' | 'custom',
  seed: -1 as number,
  removeBg: false as boolean,
  faceEnhance: false as boolean,
  lowVram: false as boolean,
  fastMode: true as boolean,
  modelKey: '' as string,
  lora: [] as Array<{ value: string; weight?: number; locked?: boolean }>,
  // UI 内部结构：loras 为数组 [{ name, weight, trigger_word }]；提交时会 JSON.stringify
  loras: [] as Array<{ name: string; weight: number; trigger_word?: string }>,
  aspectRatio: '1:1',
  resolution: 1 as number,
  numImages: 1 as number,
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
  onlyEditable: false,
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

// 宽高比选项（使用公共配置）
const aspectRatioOptions = sdAspList

// 分辨率选项（使用公共配置）
const resolutionOptions = resolutionList

// 输出图片数量选项
const numImagesOptions = Array.from({ length: 9 }, (_, i) => ({
  value: i + 1,
  label: `${i + 1}`,
}))

const { feed, createRunHooks } = useTaskFeed<GeneratedImageDetails>()
const loading = ref(false)

const mediaItems = computed<MediaLightboxItem[]>(() => {
  return feed.value
    .filter(
      (x): x is Extract<TaskFeedItem<GeneratedImageDetails>, { kind: 'image' | 'video' }> =>
        x.kind === 'image' || x.kind === 'video'
    )
    .map((x) => ({
      key: x.key,
      type: x.kind,
      src: x.originalSrc || x.thumbSrc,
      title: x.title,
      poster: x.kind === 'video' ? (x.posterSrc || x.thumbSrc) : undefined,
    }))
})
const { mediaOpen, mediaActiveKey, openMediaByKey } = useMediaLightbox(mediaItems)

// ws/pending tracking (not reactive)
const { generatingCount, runTask, disconnectAll } = useTaskRunManager()

const trimmedPrompt = computed(() => String(form.prompt || '').trim())

const isSubmitting = computed(() => loading.value)
const isGenerating = computed(() => generatingCount.value > 0)
const submitDisabledReason = computed(() => {
  if (isSubmitting.value) return t('common.submittingTask')
  if (isGenerating.value) return t('common.processingWait')
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
  if (isGenerating.value) return t('common.generating')
  return t('image.generate')
})

// feed operations are handled by useTaskFeed()

const getAspectBase = (val: string) => {
  const found = sdAspList.find((x) => x.val === val)
  return found || sdAspList[0]!
}

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

// 切换模型：仅当 model_config 明确提供字段时覆盖（不读缓存）
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

// 生成图片
const handleGenerate = async () => {
  // Guard: keep behavior consistent with button state (e.g. user hits Enter)
  if (submitDisabled.value) {
    showTopSnack(submitDisabledReason.value || t('common.cannotGenerate'))
    return
  }

  loading.value = true
  try {
    const want = Math.max(1, Math.min(9, Number(form.numImages || 1)))

    const base = getAspectBase(form.aspectRatio)
    // 分辨率下拉对应后端 upscale（仅允许 1/2/4）
    const upscaleRaw = Number(form.resolution) || 1
    const upscale = upscaleRaw === 1 || upscaleRaw === 2 || upscaleRaw === 4 ? upscaleRaw : 1

    // width/height 仅由宽高比决定；upscale 负责最终超分
    const width = Math.min(4096, Math.max(64, base.width))
    const height = Math.min(4096, Math.max(64, base.height))

    const prompt = trimmedPrompt.value
    const selectedModelMeta = modelOptions.value.find((m) => m.value === form.modelKey)
    const guidanceScaleRaw = Number(form.guidanceScale)
    const guidanceScale = Number.isFinite(guidanceScaleRaw) ? guidanceScaleRaw : 7.5
    const req = buildMkImageRequest({
      prompt,
      negativePrompt: form.negativePrompt || '',
      width,
      height,
      generateNum: want,
      modelKey: form.modelKey || undefined,
      loadName: selectedModelMeta?.load_name,
      family: selectedModelMeta?.family,
      // guidanceScale 允许为 0；仅当 NaN/非有限值时才回退默认
      guidanceScale,
      numInferenceSteps: Number(form.numInferenceSteps) || 30,
      schedulerName: String(form.schedulerName || ''),
      // 约定：-1 表示随机 seed（与 buildMkImageRequest 的约束保持一致）
      seed: form.seedMode === 'custom' ? Math.max(1, Math.floor(Number(form.seed) || 1)) : -1,
      removeBg: Boolean(form.removeBg),
      faceEnhance: Boolean(form.faceEnhance),
      fastMode: Boolean(form.fastMode),
      upscale,
      lorasPayload: form.loras,
    }) satisfies TaskCreateRequest

    const selectedModel = modelOptions.value.find((m) => m.value === form.modelKey)
    const { taskId } = await runTask(
      req,
      want,
      createRunHooks({
        title: { image: t('common.generateImage'), video: t('common.generateVideo') },
        fallbackDownloadName: { image: 'generated-image.jpeg', video: 'generated-video.mp4' },
        buildDetails: ({ file, taskId, msg, req: runReq }) => {
          return {
            prompt: (runReq as any).prompt ?? '',
            negativePrompt: (runReq as any).negative_prompt ?? '',
            cfgScale: (runReq as any).guidance_scale ?? undefined,
            steps: (runReq as any).num_inference_steps ?? undefined,
            sampler: (runReq as any).schedulerName ?? undefined,
            seed: (file as any)?.seed ?? (runReq as any).seed,
            modelName: (msg as any)?.load_name ?? (runReq as any).load_name,
            family: (msg as any)?.family ?? (runReq as any).family,
            width: (file as any)?.width ?? (runReq as any).width,
            height: (file as any)?.height ?? (runReq as any).height,
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
    // Cache last selected model UI meta only after task created successfully.
    if (taskId && selectedModel) {
      persistSelectedModel(selectedModel)
    }
  } catch (error) {
    const anyErr = error as any
    const data = anyErr?.response?.data
    const msg =
      (data && (data.msg || data.message)) ||
      (data && data.detail ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) : '') ||
      anyErr?.message ||
      String(error)
    showTopSnackError(t('common.generateFailed', { msg }))
  } finally {
    loading.value = false
  }
}

onBeforeUnmount(() => {
  // cleanup ongoing ws connections
  disconnectAll()
})

onMounted(() => {
  // Restore last successful task create request (image) into the UI form.
  const cached = getLocalCache<TaskCreateRequest>(taskCreateCacheKeyByType('image'))
  if (cached) {
    if (typeof cached.prompt === 'string') form.prompt = cached.prompt
    if (typeof cached.negative_prompt === 'string') form.negativePrompt = cached.negative_prompt
    if (typeof cached.model_key === 'string') form.modelKey = cached.model_key
    if (cached.guidance_scale !== undefined) {
      // guidanceScale 允许为 0；仅当值为 NaN 时才回退默认
      const v = Number(cached.guidance_scale)
      if (!Number.isNaN(v)) form.guidanceScale = v
    }
    if (cached.schedulerName !== undefined) form.schedulerName = String(cached.schedulerName || '')
    if (cached.num_inference_steps !== undefined) form.numInferenceSteps = Number(cached.num_inference_steps) || form.numInferenceSteps

    const seed = Number(cached.seed || 0)
    if (seed > 0) {
      form.seedMode = 'custom'
      form.seed = seed
    } else {
      form.seedMode = 'random'
      form.seed = seed === 0 ? -1 : seed
    }

    if (cached.remove_bg !== undefined) form.removeBg = Boolean(cached.remove_bg)
    if (cached.face_enhance !== undefined) form.faceEnhance = Boolean(cached.face_enhance)

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
  }

  // Restore last selected checkpoint model so ModelSelector can display selection without fetching.
  restoreCachedModel()
  fetchModels()
  fetchLoras()
})
</script>

