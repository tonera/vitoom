<template>
  <div
    class="vt-surface vt-text-smooth h-full flex flex-col overflow-hidden antialiased text-gray-950 [.dark_&]:text-gray-100"
  >
    <div class="flex-1 flex gap-6 overflow-hidden min-h-0">
      <div class="w-96 shrink-0 flex flex-col overflow-hidden min-h-0">
        <div class="rounded-lg flex-1 flex flex-col overflow-hidden min-h-0 pr-2">
          <form @submit.prevent="handleGenerate" class="flex-1 flex flex-col min-h-0">
            <div class="flex-1 min-h-0 overflow-y-auto vt-scroll space-y-4 p-1">
              <div
                class="rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-2 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35"
              >
                <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('video.generateMode') }}</label>
                <select
                  v-model="selectedTaskMode"
                  class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
                >
                  <option v-for="mode in localizedTaskModeOptions" :key="mode.key" :value="mode.key">
                    {{ mode.label }}
                  </option>
                </select>
                <p class="vt-help">
                  {{ localizedCurrentTaskMode.description || t('video.taskModeHelp') }}
                </p>
              </div>

              <div
                v-if="showMediaCard"
                class="rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-3 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35"
              >
                <div class="text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('video.inputMedia') }}</div>
                <div class="flex gap-3">
                  <div v-if="isInputVisible('url')" class="mx-auto w-24 h-24">
                    <DropUploadImage
                      v-model="poseRefList"
                      :max="1"
                      :label="inputLabel('url', currentTaskMode.key === 'inp' ? 'video.firstFrame' : 'video.referenceImage')"
                      accept="image/*"
                      category="image"
                      variant="plain"
                      fill
                      :fill-min-height="0"
                    />
                  </div>
                  <div v-if="isInputVisible('ref_video')" class="mx-auto w-24 h-24">
                    <DropUploadImage
                      v-model="refVideoList"
                      :max="1"
                      :label="inputLabel('ref_video', currentTaskMode.key === 'ivv2v' || currentTaskMode.key === 's2v' ? 'video.poseVideo' : 'video.referenceVideo')"
                      accept="video/*"
                      category="video"
                      variant="plain"
                      fill
                      :fill-min-height="0"
                    />
                  </div>
                  <div v-if="isInputVisible('face_video')" class="mx-auto w-24 h-24">
                    <DropUploadImage
                      v-model="faceVideoList"
                      :max="1"
                      :label="inputLabel('face_video', 'video.faceVideo')"
                      accept="video/*"
                      category="video"
                      variant="plain"
                      fill
                      :fill-min-height="0"
                    />
                  </div>
                  <div v-if="isInputVisible('image_file2')" class="mx-auto w-24 h-24">
                    <DropUploadImage
                      v-model="poseCtrlList"
                      :max="1"
                      :label="inputLabel('image_file2', 'video.lastFrame')"
                      accept="image/*"
                      category="image"
                      variant="plain"
                      fill
                      :fill-min-height="0"
                    />
                  </div>
                </div>

                <div v-if="isInputVisible('prompt_wav_path')" class="space-y-2">
                  <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">
                    {{ inputLabel('prompt_wav_path', 'video.audio') }}
                    <span v-if="isInputRequired('prompt_wav_path')" class="text-red-400">*</span>
                  </label>
                  <div class="flex items-center gap-2">
                    <input ref="audioFileInputRef" type="file" accept="audio/*" hidden @change="handleAudioPicked" />
                    <button
                      type="button"
                      class="cursor-pointer rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-700 transition-colors hover:bg-gray-50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/50 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800/80"
                      :disabled="audioUploading"
                      @click="audioFileInputRef?.click()"
                    >
                      {{ audioUploading ? t('common.uploading') : t('video.uploadAudio') }}
                    </button>
                    <button
                      v-if="form.prompt_wav_path"
                      type="button"
                      class="cursor-pointer rounded-xl border border-gray-200/90 bg-gray-50 px-3 py-2 text-gray-600 transition-colors hover:bg-gray-100 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/30 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800/50"
                      @click="clearAudio"
                    >
                      {{ t('common.clear') }}
                    </button>
                  </div>
                  <p class="vt-help">
                    {{ audioFileName || (form.prompt_wav_path ? basenameFromUrl(form.prompt_wav_path) : t('video.audioUploadHint')) }}
                  </p>
                </div>
              </div>

              <div
                v-if="isInputVisible('prompt')"
                class="rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-2 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35"
              >
                <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">
                  {{ inputLabel('prompt', 'video.prompt') }}
                  <span v-if="isInputRequired('prompt')" class="text-red-400">*</span>
                </label>
                <PromptTextarea
                  v-model="form.prompt"
                  :rows="4"
                  :placeholder="inputPlaceholder('prompt', 'video.promptPlaceholder')"
                />
              </div>

              <div
                v-if="showExtraParamsCard"
                class="rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-3 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35"
              >
                <div class="grid gap-3" :class="isInputVisible('direction') && isInputVisible('speed') ? 'grid-cols-2' : 'grid-cols-1'">
                  <div v-if="isInputVisible('direction')" class="space-y-2">
                    <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">
                      {{ inputLabel('direction', 'video.cameraDirection') }}
                      <span v-if="isInputRequired('direction')" class="text-red-400">*</span>
                    </label>
                    <select
                      v-model="form.direction"
                      class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
                    >
                      <option value="">{{ t('video.selectDirection') }}</option>
                      <option
                        v-for="option in directionOptions"
                        :key="option.value"
                        :value="option.value"
                      >
                        {{ option.label }}
                      </option>
                    </select>
                  </div>

                  <div v-if="isInputVisible('speed')" class="space-y-2">
                    <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">
                      {{ inputLabel('speed', 'video.cameraSpeed') }}
                      <span v-if="isInputRequired('speed')" class="text-red-400">*</span>
                    </label>
                    <input
                      v-model.number="form.speed"
                      type="number"
                      :min="speedRange.min"
                      :max="speedRange.max"
                      :step="speedRange.step"
                      class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
                    />
                  </div>
                </div>
              </div>

              <div
                class="rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-3 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35"
              >
                <div class="space-y-1">
                  <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">
                    {{ t('image.model') }} <span class="text-red-400">*</span>
                  </label>
                  <p class="vt-help">
                    {{ t('video.compatibleModels', { task: localizedCurrentTaskMode.label, count: compatibleModelOptions.length }) }}
                  </p>
                </div>

                <VideoModelSelector
                  v-model="form.modelKey"
                  :models="compatibleModelOptions"
                  :loading="modelLoading"
                  :task-label="localizedCurrentTaskMode.label"
                />
                <p v-if="modelLoading" class="vt-help">{{ t('image.loadingModels') }}</p>

                <ModelSelectorV2
                  v-if="currentTaskMode.controls.lora"
                  v-model="form.lora"
                  :family="selectedCheckpointFamily"
                  :models="loraModelOptions"
                  :total-count="loraTotal"
                  :meta="loraMeta"
                  @open="handleOpenLoraSelector"
                  @page-change="handleLoraPageChange"
                  @search-change="handleLoraSearchChange"
                  @filter-change="handleLoraFilterChange"
                  ck_point="lora"
                  mode="multiple"
                  variant="compact"
                />
                <p v-if="currentTaskMode.controls.lora && loraLoading" class="vt-help">{{ t('image.loadingLoras') }}</p>
              </div>

              <div
                class="rounded-xl border border-gray-200 bg-gray-50/80 p-4 space-y-3 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35"
              >
                <div class="grid grid-cols-3 gap-4">
                  <div v-if="currentTaskMode.controls.aspect_ratio" class="space-y-2 tooltip" :title="t('image.aspectRatio')">
                    <AspectRatioSelector
                      name="aspectRatio"
                      v-model="form.aspectRatio"
                      :init="aspectRatioOptions"
                      :showTooltip="true"
                      :label="t('image.aspectRatio')"
                      :mode="'aspect'"
                      :def="form.aspectRatio || '1:1'"
                    />
                  </div>
                  <div v-if="currentTaskMode.controls.resolution" class="space-y-2">
                    <AspectRatioSelector
                      name="resolution"
                      v-model="form.resolution"
                      :init="resolutionOptions"
                      :showTooltip="true"
                      :label="t('image.resolution')"
                      :mode="'panel'"
                      :def="form.resolution || resolutionOptions[0]?.value"
                    />
                  </div>
                  <div v-if="currentTaskMode.controls.num_images" class="space-y-2">
                    <AspectRatioSelector
                      name="numImages"
                      v-model="form.numImages"
                      :init="numImagesOptions"
                      :showTooltip="true"
                      :label="t('image.outputCount')"
                      :mode="'panel'"
                      :def="form.numImages || 1"
                    />
                  </div>
                </div>

                <div class="grid grid-cols-2 gap-4">
                  <div class="space-y-2">
                    <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('video.duration') }}</label>
                    <RangeInput
                      v-model="form.duration"
                      label="Duration"
                      :min="5"
                      :max="10"
                      :step="1"
                      :clamp="true"
                      :show-tooltip="false"
                      :round-to-step="true"
                    />
                    <p class="vt-help">{{ t('video.durationHelp') }}</p>
                  </div>
                  <div class="space-y-2">
                    <label class="block text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">FPS</label>
                    <select
                      v-model.number="form.fps"
                      class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
                    >
                      <option :value="15">15</option>
                      <option :value="24">24</option>
                    </select>
                    <p class="vt-help">{{ t('video.fpsHelp') }}</p>
                  </div>
                </div>
              </div>
            </div>

            <div class="shrink-0 pt-3 p-1">
              <div
                class="rounded-2xl border border-gray-200 bg-white/85 p-3 shadow-lg backdrop-blur [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-900/70 [.dark_&]:shadow-xl"
              >
                <div class="flex items-center gap-3">
                  <button
                    ref="advancedBtnRef"
                    type="button"
                    class="flex h-11 w-11 shrink-0 cursor-pointer items-center justify-center rounded-xl border border-gray-200 bg-white text-gray-500 transition-colors hover:bg-gray-50 hover:text-gray-950 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-900/40 [.dark_&]:text-white/60 [.dark_&]:hover:bg-gray-900/60 [.dark_&]:hover:text-white"
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

      <div class="flex-1 flex flex-col overflow-hidden min-w-0 p-1">
        <MediaGalleryStrip
          v-model:activeKey="previewActiveKey"
          :items="thumbQueue"
          :active-item="previewActiveItem"
          :loading="userFilesLoading"
          :error-text="userFilesError"
          :show-upload="false"
          :show-download="true"
          :can-load-more="true"
          @open="openMediaByKey"
          @download="handlePreviewDownload"
          @load-more="loadNextUserFiles"
        />
      </div>
    </div>

    <MediaLightbox v-model:open="mediaOpen" v-model:activeKey="mediaActiveKey" :items="mediaItems" />

    <Teleport to="body">
      <div v-if="advancedOpen" class="fixed inset-0 z-9998 bg-black/20" @click="advancedOpen = false">
        <div
          ref="advancedPopoverRef"
          class="fixed max-h-[min(80vh,640px)] w-[420px] max-w-[calc(100vw-24px)] overflow-y-auto rounded-2xl border border-gray-200 bg-white/95 p-4 shadow-2xl ring-2 ring-gray-200/60 backdrop-blur vt-scroll [.dark_&]:border-indigo-400/45 [.dark_&]:bg-gray-900/95 [.dark_&]:ring-indigo-500/20"
          :style="advancedPopoverStyle"
          role="dialog"
          aria-modal="false"
          @click.stop
        >
          <div class="flex items-center justify-between gap-3">
            <div class="text-sm font-semibold text-gray-950 [.dark_&]:text-gray-100">{{ t('image.advancedOptions') }}</div>
            <button
              type="button"
              class="flex h-8 w-8 cursor-pointer items-center justify-center rounded-lg border border-gray-200 bg-gray-50 text-gray-600 hover:bg-gray-100 hover:text-gray-950 [.dark_&]:border-white/10 [.dark_&]:bg-black/20 [.dark_&]:text-white [.dark_&]:hover:bg-black/30"
              :title="t('common.close')"
              @click="advancedOpen = false"
            >
              <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div class="mt-3 border-t border-gray-200 [.dark_&]:border-white/10"></div>

          <div class="mt-4">
            <VideoAdvancedOptions
              v-model:negativePrompt="form.negativePrompt"
              v-model:seed="form.seed"
              v-model:seedMode="form.seedMode"
              v-model:numInferenceSteps="form.numInferenceSteps"
              v-model:guidanceScale="form.guidanceScale"
            />
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch, toRef } from 'vue'
import { useI18n } from 'vue-i18n'
import PromptTextarea from '../../components/PromptTextarea.vue'
import ModelSelectorV2 from '../../components/ModelSelectorV2.vue'
import { type GeneratedImageDetails } from '../../components/GeneratedImageCard.vue'
import MediaLightbox, { type MediaLightboxItem } from '../../components/MediaLightbox.vue'
import { useMediaLightbox } from '../../composables/useMediaLightbox'
import { useCheckpointModels, type UiModelOption } from '../../composables/useCheckpointModels'
import { useLoraModels } from '../../composables/useLoraModels'
import { useTaskFeed } from '../../composables/useTaskFeed'
import MediaGalleryStrip from '../../components/MediaGalleryStrip.vue'
import DropUploadImage from '../../components/DropUploadImage.vue'
import VideoAdvancedOptions from './components/VideoAdvancedOptions.vue'
import VideoModelSelector from './components/VideoModelSelector.vue'
import { serializeLoras } from '../../modules/image/domain/imageTaskRequestFactory'
import { sdAspList } from '../../config/aspectRatios'
import AspectRatioSelector from '../../components/AspectRatioSelector.vue'
import RangeInput from '../../components/RangeInput.vue'
import { taskCreateCacheKeyByType, type TaskCreateRequest } from '../../utils/taskRunner'
import { getLocalCache } from '../../utils/localCache'
import { downloadFile, uploadFile } from '../../utils/upload'
import { useTaskRunManager } from '../../composables/useTaskRunManager'
import { useUserFilesGallery } from '../../composables/useUserFilesGallery'
import { showTopSnack, showTopSnackError } from '../../composables/useTopSnack'
import { applyModelConfigDefaults } from '../../utils/modelConfig'
import { useAnchoredPopover } from '../../composables/useAnchoredPopover'
import { computeVideoSizeByResolutionAndAspect } from '../../utils/videoSize'
import {
  fallbackVideoTaskModes,
  getVideoTaskModeByKey,
  normalizeVideoTaskModes,
  videoGenerateVisibleTaskModeKeys,
  type VideoModelProfile,
  type VideoTaskInputConfig,
  type VideoTaskModeConfig,
  type VideoTaskModeKey,
} from './videoTaskProfiles'

const { t, te } = useI18n()

const poseRefList = ref<string[]>([])
const refVideoList = ref<string[]>([])
const faceVideoList = ref<string[]>([])
const poseCtrlList = ref<string[]>([])
const audioFileInputRef = ref<HTMLInputElement | null>(null)
const audioUploading = ref(false)
const audioFileName = ref('')

const aspectRatioOptions = sdAspList.map((x) => ({
  ...x,
  value: x.label,
  val: x.label,
}))
const numImagesOptions = Array.from({ length: 9 }, (_, i) => ({ value: i + 1, label: `${i + 1}` }))

const form = reactive({
  prompt: '',
  modelKey: '' as string,
  lora: [] as Array<{ value: string; weight?: number; locked?: boolean }>,
  loras: [] as Array<{ name: string; weight: number; trigger_word?: string }>,
  aspectRatio: '1:1',
  resolution: 720 as number,
  numImages: 1 as number,
  negativePrompt: '',
  guidanceScale: 3.5 as number,
  numInferenceSteps: 25 as number,
  seedMode: 'random' as 'random' | 'custom',
  seed: -1 as number,
  removeBg: false as boolean,
  faceEnhance: false as boolean,
  lowVram: false as boolean,
  url: '' as string,
  ref_video: '' as string,
  face_video: '' as string,
  image_file2: '' as string,
  prompt_wav_path: '' as string,
  direction: '' as string,
  speed: 1 as number,
  duration: 5 as number,
  fps: 24 as number,
  edit_act: '' as string,
  job_type: 'MKV',
})

const selectedTaskMode = ref<VideoTaskModeKey>('t2v')

const {
  modelOptions,
  modelLoading,
  modelMeta,
  selectedCheckpointFamily,
  selectedModelConfig,
  fetchModels,
  restoreCachedModel,
  persistSelectedModel,
} = useCheckpointModels({
  modelKey: toRef(form, 'modelKey'),
  onlyEditable: false,
  taskType: 'video',
  cacheKey: 'vitoom:model:last:checkpoint:video',
  perPage: 100,
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
  taskType: 'video',
  baseFamily: selectedCheckpointFamily,
  selected: toRef(form, 'lora'),
  payload: toRef(form, 'loras'),
  defaultWeight: 0.8,
  autoFetchOnBaseModelChange: false,
})

const taskModeOptions = computed<VideoTaskModeConfig[]>(() => {
  const all = normalizeVideoTaskModes(modelMeta.value?.video_task_modes || fallbackVideoTaskModes)
  const visible = new Set(videoGenerateVisibleTaskModeKeys)
  return all.filter((mode) => visible.has(mode.key))
})

function localizeTaskMode(mode: VideoTaskModeConfig): VideoTaskModeConfig {
  const labelKey = `video.taskModes.${mode.key}.label`
  const descKey = `video.taskModes.${mode.key}.description`
  return {
    ...mode,
    label: te(labelKey) ? t(labelKey) : mode.label,
    description: te(descKey) ? t(descKey) : mode.description,
  }
}

const localizedTaskModeOptions = computed(() => taskModeOptions.value.map(localizeTaskMode))

const currentTaskMode = computed<VideoTaskModeConfig>(() => {
  return getVideoTaskModeByKey(taskModeOptions.value, selectedTaskMode.value) || taskModeOptions.value[0] || fallbackVideoTaskModes[0]!
})

const localizedCurrentTaskMode = computed(() => localizeTaskMode(currentTaskMode.value))

const compatibleModelOptions = computed<UiModelOption[]>(() =>
  modelOptions.value.filter((model) => {
    const profile = model.video_profile as VideoModelProfile | undefined
    return Boolean(profile?.task_modes?.some((mode) => mode.key === selectedTaskMode.value))
  })
)

const selectedModel = computed<UiModelOption | undefined>(() =>
  modelOptions.value.find((item) => item.value === form.modelKey)
)

const selectedCompatibleModel = computed<UiModelOption | undefined>(() =>
  compatibleModelOptions.value.find((item) => item.value === form.modelKey)
)

const selectedModelTaskMode = computed<VideoTaskModeConfig | undefined>(() => {
  const profile = selectedModel.value?.video_profile as VideoModelProfile | undefined
  return profile?.task_modes?.find((mode) => mode.key === selectedTaskMode.value)
})

const currentResolutions = computed<number[]>(() => {
  const fromModel = selectedModelTaskMode.value?.supported_resolutions
  if (Array.isArray(fromModel) && fromModel.length > 0) return fromModel
  return currentTaskMode.value.supported_resolutions || [480, 720]
})

const resolutionOptions = computed(() =>
  currentResolutions.value.map((value) => ({
    value,
    label: `${value}P`,
  }))
)

// const selectedModelResolutionHint = computed(() => {
//   const resolutions = currentResolutions.value
//   if (!selectedModel.value || resolutions.length === 0) return ''
//   if (resolutions.length === 1) return `该模型仅支持 ${resolutions[0]}P`
//   return `该模型支持 ${resolutions.map((item) => `${item}P`).join(' / ')}`
// })

function getInputConfig(key: string): VideoTaskInputConfig {
  return currentTaskMode.value.inputs?.[key] || { visible: false, required: false }
}

function isInputVisible(key: string) {
  return Boolean(getInputConfig(key).visible)
}

function isInputRequired(key: string) {
  return Boolean(getInputConfig(key).required)
}

function inputLabel(key: string, fallbackKey: string) {
  if (te(fallbackKey)) return t(fallbackKey)
  const cfg = getInputConfig(key)
  if (cfg.label) return String(cfg.label)
  return t(fallbackKey)
}

function inputPlaceholder(key: string, fallbackKey: string) {
  const modeKey = currentTaskMode.value.key
  const modeSpecificKey = `video.taskModes.${modeKey}.${key}Placeholder`
  if (te(modeSpecificKey)) return t(modeSpecificKey)
  if (te(fallbackKey)) return t(fallbackKey)
  const cfg = getInputConfig(key)
  if (cfg.placeholder) return String(cfg.placeholder)
  return t(fallbackKey)
}

const directionOptions = computed(() => {
  const cfg = getInputConfig('direction')
  if (!Array.isArray(cfg.options)) return []
  return cfg.options.map((option) => {
    const dirKey = `video.directions.${option.value}`
    return {
      ...option,
      label: te(dirKey) ? t(dirKey) : option.label,
    }
  })
})

const speedRange = computed(() => {
  const cfg = getInputConfig('speed')
  return {
    min: Number(cfg.min ?? 0.1),
    max: Number(cfg.max ?? 4),
    step: Number(cfg.step ?? 0.1),
    defaultValue: Number(cfg.default ?? 1),
  }
})

const showMediaCard = computed(() =>
  ['url', 'ref_video', 'face_video', 'image_file2', 'prompt_wav_path'].some((key) => isInputVisible(key))
)
const showExtraParamsCard = computed(() => ['direction', 'speed'].some((key) => isInputVisible(key)))

watch(
  () => form.modelKey,
  () => {
    applyModelConfigDefaults(selectedModelConfig.value, {
      setGuidanceScale: (v) => (form.guidanceScale = v),
      setNumInferenceSteps: (v) => (form.numInferenceSteps = v),
    })
  }
)

watch(
  taskModeOptions,
  (options) => {
    if (!options.some((item) => item.key === selectedTaskMode.value)) {
      selectedTaskMode.value = (options[0]?.key || 't2v') as VideoTaskModeKey
    }
  },
  { immediate: true }
)

watch(
  currentTaskMode,
  (mode) => {
    form.job_type = mode.job_type
    if (isInputVisible('speed')) {
      const nextDefault = Number(getInputConfig('speed').default ?? speedRange.value.defaultValue)
      if (!Number.isFinite(Number(form.speed)) || Number(form.speed) <= 0) {
        form.speed = nextDefault
      }
    }
  },
  { immediate: true }
)

watch(
  () => compatibleModelOptions.value.map((item) => item.value).join(','),
  () => {
    if (!form.modelKey) return
    const stillCompatible = compatibleModelOptions.value.some((item) => item.value === form.modelKey)
    if (!stillCompatible) {
      form.modelKey = ''
      form.lora = []
      form.loras = []
      if (!modelLoading.value) showTopSnack(t('video.modelRemovedIncompatible'))
    }
  }
)

watch(
  [currentResolutions, selectedModelTaskMode],
  ([resolutions, mode]) => {
    if (!resolutions.length) return
    if (!resolutions.includes(Number(form.resolution))) {
      form.resolution = Number(mode?.default_resolution ?? currentTaskMode.value.default_resolution ?? resolutions[0])
    }
  },
  { immediate: true }
)

watch(
  () => poseRefList.value,
  (newVal) => {
    form.url = newVal?.[0] || ''
  },
  { immediate: true }
)

watch(
  () => refVideoList.value,
  (newVal) => {
    form.ref_video = newVal?.[0] || ''
  },
  { immediate: true }
)

watch(
  () => faceVideoList.value,
  (newVal) => {
    form.face_video = newVal?.[0] || ''
  },
  { immediate: true }
)

watch(
  () => poseCtrlList.value,
  (newVal) => {
    form.image_file2 = newVal?.[0] || ''
  },
  { immediate: true }
)

const { feed, createRunHooks } = useTaskFeed<GeneratedImageDetails>({
  onInsertedKey: (key) => {
    previewActiveKey.value = key
  },
})
const loading = ref(false)

const {
  open: advancedOpen,
  anchorRef: advancedBtnRef,
  popoverRef: advancedPopoverRef,
  style: advancedPopoverStyle,
  toggle: toggleAdvanced,
} = useAnchoredPopover()
void advancedBtnRef
void advancedPopoverRef

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
} = useUserFilesGallery({
  category: 'video',
  limit: 60,
  onLoaded: () => {
    if (!previewActiveKey.value && thumbQueue.value.length > 0) {
      previewActiveKey.value = thumbQueue.value[0]!.key
    }
  },
})

const previewActiveKey = ref<string | null>(null)

const thumbQueue = computed<PreviewThumbItem[]>(() => {
  const out: PreviewThumbItem[] = []
  const seen = new Set<string>()
  const push = (it: PreviewThumbItem) => {
    if (!it?.key || seen.has(it.key)) return
    seen.add(it.key)
    out.push(it)
  }

  for (const x of feed.value) {
    if (x.kind !== 'video') continue
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

  for (const f of userFiles.value) {
    const url = String(f?.url || '')
    if (!url) continue
    const mime = String((f as any)?.mime_type || '').toLowerCase()
    const isVideo = mime ? mime.startsWith('video/') : true
    if (!isVideo) continue
    push({
      key: url,
      kind: 'video',
      thumbSrc: String(f?.thumb_url || url),
      originalSrc: url,
      posterSrc: String(f?.thumb_url || url),
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
const { generatingCount, runTask, disconnectAll } = useTaskRunManager()

const trimmedPrompt = computed(() => String(form.prompt || '').trim())
const isSubmitting = computed(() => loading.value)
const isGenerating = computed(() => generatingCount.value > 0)
const submitDisabledReason = computed(() => {
  if (isSubmitting.value) return t('common.submittingTask')
  if (isGenerating.value) return t('common.processingWait')
  if (!form.modelKey) return modelLoading.value ? t('common.modelsLoading') : t('common.selectModel')
  if (!modelLoading.value && compatibleModelOptions.value.length === 0) return t('video.noModelsForMode')
  if (!selectedCompatibleModel.value) return t('video.modelIncompatible')
  if (!String(selectedCompatibleModel.value.model_key || '').trim()) return t('video.missingModelKey')
  if (isInputRequired('prompt') && !trimmedPrompt.value) return t('common.enterPrompt')
  if (isInputRequired('url') && !poseRefList.value.length) {
    return t('common.uploadSelect', {
      label: currentTaskMode.value.key === 'inp' ? t('video.firstFrame') : t('video.referenceImage'),
    })
  }
  if (isInputRequired('ref_video') && !refVideoList.value.length) {
    return t('common.uploadSelect', { label: inputLabel('ref_video', 'video.referenceVideo') })
  }
  if (isInputRequired('face_video') && !faceVideoList.value.length) {
    return t('common.uploadSelect', { label: inputLabel('face_video', 'video.faceVideo') })
  }
  if (isInputRequired('image_file2') && !poseCtrlList.value.length) {
    return t('common.uploadSelect', { label: inputLabel('image_file2', 'video.lastFrame') })
  }
  if (isInputRequired('prompt_wav_path') && !form.prompt_wav_path) return t('video.uploadAudioFirst')
  if (isInputRequired('direction') && !String(form.direction || '').trim()) return t('video.selectCameraDirection')
  if (isInputRequired('speed')) {
    const speed = Number(form.speed)
    if (!Number.isFinite(speed) || speed <= 0) return t('video.enterValidSpeed')
  }
  if (currentTaskMode.value.controls.aspect_ratio && !form.aspectRatio) return t('common.selectAspectRatio')
  if (currentTaskMode.value.controls.resolution && !form.resolution) return t('common.selectResolution')
  if (currentTaskMode.value.controls.num_images) {
    const n = Number(form.numImages)
    if (!Number.isFinite(n) || n < 1) return t('common.selectOutputCount')
  }
  return ''
})
const submitDisabled = computed(() => Boolean(submitDisabledReason.value))
const submitButtonText = computed(() => {
  if (isSubmitting.value) return t('common.submitting')
  if (isGenerating.value) return t('common.processing')
  return t('video.start')
})

const normalizeMediaUrl = (url: string): string => {
  if (!url) return url
  if (url.startsWith('http://') || url.startsWith('https://')) return url
  const origin = window.location.origin
  const path = url.startsWith('/') ? url : `/${url}`
  return `${origin}${path}`
}

const handleGenerate = async () => {
  if (submitDisabled.value) {
    showTopSnack(submitDisabledReason.value || t('common.cannotSubmit'))
    return
  }

  loading.value = true
  try {
    const want = Math.max(1, Math.min(9, Math.floor(Number(form.numImages || 1))))
    const { width, height } = computeVideoSizeByResolutionAndAspect(Number(form.resolution || currentResolutions.value[0] || 480), form.aspectRatio)
    const selectedModelOption = selectedCompatibleModel.value
    const selectedModelKey = String(selectedModelOption?.model_key || '').trim()
    if (!selectedModelKey) {
      showTopSnack(t('video.missingModelKey'))
      return
    }
    const loras = serializeLoras(form.loras as any)

    const req: TaskCreateRequest = {
      task_type: 'video',
      job_type: String(currentTaskMode.value.job_type || form.job_type || 'MKV'),
      video_task_mode: selectedTaskMode.value,
      mkv_mode: currentTaskMode.value.mkv_mode || undefined,
      prompt: isInputVisible('prompt') ? String(trimmedPrompt.value || '') : '',
      negative_prompt: String(form.negativePrompt || ''),
      width,
      height,
      generate_num: want,
      aspect_ratio: String(form.aspectRatio || ''),

      model_key: selectedModelKey,
      load_name: selectedModelOption?.load_name,
      family: selectedModelOption?.family,

      duration: Math.max(5, Math.min(10, Math.floor(Number(form.duration) || 5))),
      fps: Number(form.fps) === 15 ? 15 : 24,
      resolution: String(form.resolution || ''),

      url: isInputVisible('url') ? normalizeMediaUrl(String(form.url || '')) : undefined,
      ref_video: isInputVisible('ref_video') ? normalizeMediaUrl(String(form.ref_video || '')) : undefined,
      face_video: isInputVisible('face_video') ? normalizeMediaUrl(String(form.face_video || '')) : undefined,
      image_file2: isInputVisible('image_file2') ? normalizeMediaUrl(String(form.image_file2 || '')) : undefined,
      prompt_wav_path: isInputVisible('prompt_wav_path') ? normalizeMediaUrl(String(form.prompt_wav_path || '')) : undefined,
      direction: isInputVisible('direction') ? String(form.direction || '') : undefined,
      speed: isInputVisible('speed') ? Number(form.speed) : undefined,

      guidance_scale: (() => {
        const v = Number(form.guidanceScale)
        return Number.isFinite(v) ? v : 7.5
      })(),
      num_inference_steps: Math.max(1, Math.min(100, Math.floor(Number(form.numInferenceSteps) || 25))),
      seed: form.seedMode === 'custom' ? Math.max(1, Math.floor(Number(form.seed) || 1)) : -1,

      ...(loras ? { loras } : {}),
    }

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

    if (taskId && selectedModelOption) persistSelectedModel(selectedModelOption)
  } catch (error: any) {
    const msg = error?.response?.data?.detail || error?.message || String(error)
    showTopSnackError(t('common.generateFailed', { msg }))
  } finally {
    loading.value = false
  }
}

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
  showTopSnack(success ? t('common.downloadSuccess') : t('common.downloadFailed'))
}

function basenameFromUrl(url: string): string {
  try {
    const parsed = new URL(url)
    return decodeURIComponent(parsed.pathname.split('/').pop() || t('video.uploadedAudio'))
  } catch {
    return decodeURIComponent(String(url || '').split('/').pop() || t('video.uploadedAudio'))
  }
}

function deriveTaskModeFromCache(cached: TaskCreateRequest): VideoTaskModeKey {
  const explicit = String((cached as any).video_task_mode || '').trim() as VideoTaskModeKey
  if (explicit && fallbackVideoTaskModes.some((item) => item.key === explicit)) return explicit
  const jobType = String(cached.job_type || '').trim().toUpperCase()
  if (jobType === 'S2V') return 's2v'
  if (jobType === 'INP') return 'inp'
  if (jobType === 'CCV') return 'ccv'
  const hasUrl = Boolean(String((cached as any).url || '').trim())
  const hasRefVideo = Boolean(String((cached as any).ref_video || '').trim())
  const hasFaceVideo = Boolean(String((cached as any).face_video || '').trim())
  const hasPrompt = Boolean(String(cached.prompt || '').trim())
  if (hasUrl && hasRefVideo && hasFaceVideo) return 'ivv2v'
  if (hasUrl && hasRefVideo) return 'vicv'
  if (hasUrl && hasPrompt) return 'ti2v'
  if (hasUrl) return 'i2v'
  return 't2v'
}

async function handleAudioPicked(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  audioUploading.value = true
  try {
    const url = await uploadFile(file)
    form.prompt_wav_path = url
    audioFileName.value = file.name
    showTopSnack(t('common.uploadSuccess'))
  } catch (error: any) {
    showTopSnackError(error?.message || t('common.uploadFailed', { msg: error?.message || t('common.unknownError') }))
  } finally {
    audioUploading.value = false
    input.value = ''
  }
}

function clearAudio() {
  form.prompt_wav_path = ''
  audioFileName.value = ''
}

async function handleOpenLoraSelector() {
  if (!currentTaskMode.value.controls.lora) return
  if (!String(selectedCheckpointFamily.value || '').trim()) return
  await fetchLoras()
}

onBeforeUnmount(() => {
  disconnectAll()
})

onMounted(() => {
  const cached = getLocalCache<TaskCreateRequest>(taskCreateCacheKeyByType('video'))
  if (cached) {
    selectedTaskMode.value = deriveTaskModeFromCache(cached)
    form.job_type = currentTaskMode.value.job_type
    if (typeof cached.prompt === 'string') form.prompt = cached.prompt
    if (typeof cached.model_key === 'string') form.modelKey = cached.model_key
    if (typeof (cached as any).negative_prompt === 'string') form.negativePrompt = String((cached as any).negative_prompt || '')
    if ((cached as any).guidance_scale !== undefined) {
      const v = Number((cached as any).guidance_scale)
      if (!Number.isNaN(v)) form.guidanceScale = v
    }
    if ((cached as any).num_inference_steps !== undefined) form.numInferenceSteps = Number((cached as any).num_inference_steps) || form.numInferenceSteps

    const seed = Number((cached as any).seed || 0)
    if (seed > 0) {
      form.seedMode = 'custom'
      form.seed = seed
    } else {
      form.seedMode = 'random'
      form.seed = -1
    }
    if ((cached as any).remove_bg !== undefined) form.removeBg = Boolean((cached as any).remove_bg)
    if ((cached as any).face_enhance !== undefined) form.faceEnhance = Boolean((cached as any).face_enhance)
    const gen = Number((cached as any).generate_num)
    if (Number.isFinite(gen) && gen >= 1) form.numImages = Math.max(1, Math.min(9, Math.floor(gen)))
    const cachedDuration = Number((cached as any).duration)
    if (Number.isFinite(cachedDuration) && cachedDuration >= 5 && cachedDuration <= 10) {
      form.duration = Math.floor(cachedDuration)
    }
    const cachedFps = Number((cached as any).fps ?? (cached as any)?.model_cfg?.fps)
    if (cachedFps === 15 || cachedFps === 24) form.fps = cachedFps
    const cachedResolution = Number((cached as any).resolution)
    if (cachedResolution === 480 || cachedResolution === 720) form.resolution = cachedResolution
    const ar = (cached as any).aspect_ratio
    if (typeof ar === 'string' && ar.trim()) form.aspectRatio = ar.trim()
    if (typeof (cached as any).url === 'string' && (cached as any).url.trim()) poseRefList.value = [String((cached as any).url)]
    if (typeof (cached as any).ref_video === 'string' && (cached as any).ref_video.trim()) refVideoList.value = [String((cached as any).ref_video)]
    if (typeof (cached as any).face_video === 'string' && (cached as any).face_video.trim()) faceVideoList.value = [String((cached as any).face_video)]
    if (typeof (cached as any).image_file2 === 'string' && (cached as any).image_file2.trim()) poseCtrlList.value = [String((cached as any).image_file2)]
    if (typeof (cached as any).prompt_wav_path === 'string' && (cached as any).prompt_wav_path.trim()) {
      form.prompt_wav_path = String((cached as any).prompt_wav_path)
      audioFileName.value = basenameFromUrl(form.prompt_wav_path)
    }
    if (typeof (cached as any).direction === 'string') form.direction = String((cached as any).direction || '')
    if ((cached as any).speed !== undefined) {
      const speed = Number((cached as any).speed)
      if (Number.isFinite(speed) && speed > 0) form.speed = speed
    }
  }

  restoreCachedModel()
  fetchModels()
  loadUserFiles(true)
})
</script>
