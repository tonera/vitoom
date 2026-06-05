<template>
  <div class="flex h-full min-h-0 flex-col overflow-hidden bg-gray-50 text-gray-950 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100">
    <div class="shrink-0 border-b border-gray-200 bg-white px-5 py-4 [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
      <div class="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 class="text-xl font-semibold">{{ t('audio.title') }}</h1>
          <p class="mt-1 text-sm text-gray-500 [.dark_&]:text-gray-400">{{ t('audio.subtitle') }}</p>
        </div>

        <div class="flex flex-wrap gap-2" role="tablist" :aria-label="t('audio.servicesAriaLabel')">
          <button
            v-for="tab in tabs"
            :key="tab.value"
            type="button"
            class="rounded-full px-4 py-2 text-sm font-medium transition-colors cursor-pointer"
            :class="activeTab === tab.value ? activeTabClass : inactiveTabClass"
            role="tab"
            :aria-selected="activeTab === tab.value"
            @click="activeTab = tab.value"
          >
            {{ tab.label }}
          </button>
        </div>
      </div>
    </div>

    <div class="grid min-h-0 flex-1 grid-cols-1 overflow-hidden lg:grid-cols-[420px_minmax(0,1fr)]">
      <aside class="min-h-0 overflow-y-auto border-b border-gray-200 bg-white p-5 [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900 lg:border-b-0 lg:border-r">
        <div class="space-y-5">
          <section class="rounded-2xl border border-gray-200 bg-gray-50/80 p-4 [.dark_&]:border-gray-800 [.dark_&]:bg-gray-950/40">
            <h2 class="text-sm font-semibold text-gray-950 [.dark_&]:text-white">{{ currentTabTitle }}</h2>
            <p class="mt-1 text-xs leading-5 text-gray-500 [.dark_&]:text-gray-400">{{ currentTabDescription }}</p>
          </section>

          <template v-if="activeTab === 'asr'">
            <label class="block">
              <span class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.asrModel') }}</span>
              <select v-model="asrForm.modelName" class="mt-1 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-indigo-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100">
                <option v-for="model in asrModelOptions" :key="model.value" :value="model.value" :disabled="!model.load_name">
                  {{ model.label }}{{ model.load_name ? '' : t('audio.notRegistered') }}
                </option>
              </select>
            </label>

            <AudioUploadField
              :model-value="asrForm.audioUrl"
              :label="t('audio.audioToRecognize')"
              :description="t('audio.asrUploadDescription')"
              :hint="t('audio.asrUploadHint')"
              :file-name="asrForm.audioName"
              :uploading="uploadingField === 'asr'"
              @upload="(file) => handleAudioUpload(file, 'asr')"
              @clear="clearAsrAudio"
            />

            <label class="block">
              <span class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.recognizeLanguage') }}</span>
              <select v-model="asrForm.language" class="mt-1 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-indigo-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100">
                <option value="">{{ t('audio.autoDetect') }}</option>
                <option value="zh">{{ t('audio.langZh') }}</option>
                <option value="en">{{ t('audio.langEn') }}</option>
                <option value="ja">{{ t('audio.langJa') }}</option>
                <option value="ko">{{ t('audio.langKo') }}</option>
              </select>
            </label>

            <label class="flex items-center gap-2 text-sm text-gray-600 [.dark_&]:text-gray-300">
              <input v-model="asrForm.timestamps" type="checkbox" class="h-4 w-4 rounded border-gray-300 text-indigo-600">
              {{ t('audio.outputTimestamps') }}
            </label>
          </template>

          <template v-else>
            <section v-if="activeTab !== 'dialog'" class="space-y-3">
              <div class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.modelFamily') }}</div>
              <div class="grid grid-cols-2 gap-2">
                <button
                  v-for="option in ttsFamilyOptions"
                  :key="option.value"
                  type="button"
                  class="rounded-xl border px-3 py-3 text-sm font-medium transition cursor-pointer"
                  :class="ttsForm.family === option.value ? 'border-indigo-500 bg-indigo-50 text-indigo-700 [.dark_&]:border-indigo-400 [.dark_&]:bg-indigo-950/40 [.dark_&]:text-indigo-200' : 'border-gray-200 bg-white text-gray-600 hover:bg-gray-50 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800'"
                  @click="ttsForm.family = option.value"
                >
                  {{ option.label }}
                </button>
              </div>
            </section>

            <section v-if="activeTab === 'dialog'" class="space-y-3">
              <div class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.dialogMode') }}</div>
              <div class="grid grid-cols-2 gap-2">
                <button
                  v-for="option in dialogVoiceModeOptions"
                  :key="option.value"
                  type="button"
                  class="rounded-xl border px-3 py-3 text-sm font-medium transition cursor-pointer"
                  :class="dialogForm.mode === option.value ? 'border-indigo-500 bg-indigo-50 text-indigo-700 [.dark_&]:border-indigo-400 [.dark_&]:bg-indigo-950/40 [.dark_&]:text-indigo-200' : 'border-gray-200 bg-white text-gray-600 hover:bg-gray-50 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800'"
                  @click="dialogForm.mode = option.value"
                >
                  <span class="block">{{ option.label }}</span>
                  <span class="mt-1 block text-xs font-normal opacity-75">{{ option.description }}</span>
                </button>
              </div>
            </section>

            <section v-if="ttsForm.family === 'qwen' && qwenSupportsQuality" class="space-y-3">
              <div class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.quality') }}</div>
              <div class="grid grid-cols-2 gap-2">
                <button
                  v-for="option in qwenQualityOptions"
                  :key="option.value"
                  type="button"
                  class="rounded-xl border px-3 py-3 text-sm font-medium transition cursor-pointer"
                  :class="ttsForm.quality === option.value ? 'border-indigo-500 bg-indigo-50 text-indigo-700 [.dark_&]:border-indigo-400 [.dark_&]:bg-indigo-950/40 [.dark_&]:text-indigo-200' : 'border-gray-200 bg-white text-gray-600 hover:bg-gray-50 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800'"
                  @click="ttsForm.quality = option.value"
                >
                  <span class="block">{{ option.label }}</span>
                  <span class="mt-1 block text-xs font-normal opacity-75">{{ option.description }}</span>
                </button>
              </div>
            </section>

            <label v-if="ttsForm.family === 'voxcpm'" class="block">
              <span class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.voxcpmModel') }}</span>
              <select v-model="ttsForm.voxcpmModelName" class="mt-1 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-indigo-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100">
                <option v-for="model in voxcpmModelOptions" :key="model.value" :value="model.value" :disabled="!model.load_name">
                  {{ model.label }}{{ model.load_name ? '' : t('audio.notRegistered') }}
                </option>
              </select>
            </label>

            <section class="rounded-2xl border border-gray-200 bg-gray-50/80 p-4 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950/40">
              <div class="flex items-start justify-between gap-3">
                <div>
                  <div class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.autoSelectModel') }}</div>
                  <div class="mt-1 text-sm font-semibold text-gray-950 [.dark_&]:text-white">{{ autoTtsModelDisplayName }}</div>
                  <p class="mt-1 text-xs leading-5 text-gray-500 [.dark_&]:text-gray-400">{{ autoTtsModelReason }}</p>
                </div>
                <span class="shrink-0 rounded-full bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-700 [.dark_&]:bg-indigo-950/40 [.dark_&]:text-indigo-300">
                  {{ t('common.auto') }}
                </span>
              </div>
            </section>

            <template v-if="activeTab === 'tts'">
              <label class="block">
                <span class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.generateMode') }}</span>
                <select v-model="ttsForm.mode" class="mt-1 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-indigo-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100">
                  <option v-for="option in ttsModeOptions" :key="option.value" :value="option.value">{{ option.label }}</option>
                </select>
              </label>

              <label v-if="ttsForm.family === 'qwen' && ttsForm.mode === 'random'" class="block">
                <span class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.speaker') }}</span>
                <select v-model="ttsForm.speakerName" class="mt-1 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-indigo-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100">
                  <option v-for="speaker in qwenSpeakerOptions" :key="speaker.value" :value="speaker.value">
                    {{ speaker.label }}
                  </option>
                </select>
              </label>

              <label v-if="ttsForm.family === 'voxcpm' && ttsForm.mode === 'random'" class="block">
                <span class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.speaker') }}</span>
                <select v-model="ttsForm.voxcpmSpeakerName" class="mt-1 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-indigo-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100">
                  <option v-for="speaker in voxcpmSpeakerOptions" :key="speaker.value" :value="speaker.value">
                    {{ speaker.label }}
                  </option>
                </select>
              </label>

              <div class="block">
                <span class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.textToSynthesize') }}</span>
                <div
                  class="relative mt-1 rounded-xl"
                  :class="textFileDragActive ? 'ring-2 ring-indigo-400/50 ring-offset-2 ring-offset-white [.dark_&]:ring-offset-gray-900' : ''"
                  @dragenter.prevent="handleTextFileDragEnter"
                  @dragover.prevent="handleTextFileDragOver"
                  @dragleave.prevent="handleTextFileDragLeave"
                  @drop.prevent="handleTextFileDrop"
                >
                  <div
                    v-if="textFileDragActive"
                    class="pointer-events-none absolute inset-0 z-10 flex items-center justify-center rounded-xl bg-white/80 px-4 text-center text-sm font-medium text-indigo-700 backdrop-blur-[2px] [.dark_&]:bg-gray-950/70 [.dark_&]:text-indigo-200"
                    aria-hidden="true"
                  >
                    {{ t('audio.textDropHint') }}
                  </div>
                  <textarea v-model="ttsForm.prompt" class="min-h-36 w-full resize-y rounded-xl border border-gray-200 bg-white px-3 py-2 pr-12 text-sm leading-6 outline-none transition focus:border-indigo-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100" :placeholder="t('audio.textPlaceholder')" />
                  <button
                    type="button"
                    class="absolute bottom-2 right-2 flex h-9 w-9 items-center justify-center rounded-full text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-950 disabled:pointer-events-none disabled:opacity-40 [.dark_&]:text-gray-400 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-gray-100 cursor-pointer"
                    :aria-label="t('audio.uploadTextFile')"
                    :title="t('audio.uploadTextFileTitle')"
                    :disabled="textFileReading || loading || isGenerating"
                    @click="textFileInputRef?.click()"
                  >
                    <svg
                      v-if="!textFileReading"
                      class="h-5 w-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      aria-hidden="true"
                    >
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
                    </svg>
                    <span v-else class="h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-gray-700 [.dark_&]:border-gray-600 [.dark_&]:border-t-gray-200" aria-hidden="true" />
                  </button>
                  <input
                    ref="textFileInputRef"
                    type="file"
                    class="hidden"
                    accept=".txt,.md,.markdown,.docx,text/plain,text/markdown,text/x-markdown,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    @change="handleTextFilePicked"
                  >
                </div>
              </div>

              <label v-if="needsInstruct" class="block">
                <span class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.instructLabel') }}</span>
                <textarea v-model="ttsForm.instruct" class="mt-1 min-h-24 w-full resize-y rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm leading-6 outline-none transition focus:border-indigo-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100" :placeholder="t('audio.instructPlaceholder')" />
              </label>

              <AudioUploadField
                v-if="needsReferenceAudio"
                :model-value="ttsForm.refAudioUrl"
                :label="t('audio.referenceAudio')"
                :description="t('audio.referenceAudioDesc')"
                :hint="t('audio.referenceAudioHint')"
                :file-name="ttsForm.refAudioName"
                :uploading="uploadingField === 'tts-ref'"
                @upload="(file) => handleAudioUpload(file, 'tts-ref')"
                @clear="clearTtsRefAudio"
              />

              <label v-if="needsReferenceText" class="block">
                <span class="text-xs font-medium text-gray-600 [.dark_&]:text-gray-300">{{ t('audio.referenceText') }}</span>
                <textarea v-model="ttsForm.refText" class="mt-1 min-h-24 w-full resize-y rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm leading-6 outline-none transition focus:border-indigo-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100" :placeholder="t('audio.referenceTextPlaceholder')" />
              </label>
            </template>

            <template v-else>
              <SpeakerDialogEditor v-model="dialogLines" :mode="dialogForm.mode" />
            </template>
          </template>

          <button
            type="button"
            class="w-full rounded-2xl bg-gray-950 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-50 [.dark_&]:bg-white [.dark_&]:text-gray-950 [.dark_&]:hover:bg-gray-200 cursor-pointer"
            :disabled="submitDisabled"
            @click="submitAudioTask"
          >
            {{ submitButtonText }}
          </button>
          <p v-if="submitDisabledReason" class="text-center text-xs text-gray-500 [.dark_&]:text-gray-400">{{ submitDisabledReason }}</p>
        </div>
      </aside>

      <main class="min-h-0 overflow-y-auto p-5">
        <div class="mx-auto max-w-4xl space-y-4">
          <section class="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
            <div class="flex items-center justify-between gap-3">
              <div>
                <h2 class="text-base font-semibold">{{ t('audio.resultsPreview') }}</h2>
                <p class="mt-1 text-sm text-gray-500 [.dark_&]:text-gray-400">{{ t('audio.resultsPreviewDesc') }}</p>
              </div>
              <button
                v-if="results.length"
                type="button"
                class="rounded-xl border border-gray-200 px-3 py-1.5 text-xs font-medium text-gray-600 transition hover:bg-gray-50 [.dark_&]:border-gray-700 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800 cursor-pointer"
                @click="results = []"
              >
                {{ t('common.clear') }}
              </button>
            </div>
          </section>

          <div v-if="results.length === 0" class="flex min-h-[360px] flex-col items-center justify-center rounded-2xl border border-dashed border-gray-300 bg-white p-8 text-center [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900">
            <div class="flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-50 text-indigo-600 [.dark_&]:bg-indigo-950/40 [.dark_&]:text-indigo-300">
              AUD
            </div>
            <h3 class="mt-4 text-base font-semibold text-gray-950 [.dark_&]:text-white">{{ t('audio.noResults') }}</h3>
            <p class="mt-2 max-w-md text-sm text-gray-500 [.dark_&]:text-gray-400">{{ t('audio.noResultsDesc') }}</p>
          </div>

          <AudioResultCard
            v-for="result in results"
            :key="result.key"
            :result="result"
            @download="downloadResult"
            @load-text="loadTextResult"
          />
        </div>
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import mammoth from 'mammoth'
import { useTaskRunManager, type RunTaskHooks } from '../../composables/useTaskRunManager'
import { getOriginalFileUrl, taskCreateCacheKeyByType, type TaskCreateRequest } from '../../utils/taskRunner'
import { getLocalCache } from '../../utils/localCache'
import { downloadFile, uploadFile } from '../../utils/upload'
import { showTopSnack, showTopSnackError } from '../../composables/useTopSnack'
import { listTtsSpeakers, type TtsSpeakerOption } from '../../api/audio'
import AudioUploadField from './components/AudioUploadField.vue'
import AudioResultCard from './components/AudioResultCard.vue'
import SpeakerDialogEditor from './components/SpeakerDialogEditor.vue'

const { t } = useI18n()

type AudioTab = 'asr' | 'tts' | 'dialog'
type TtsFamily = 'qwen' | 'voxcpm'
type TtsMode = 'random' | 'design' | 'clone' | 'clone_plus'
type DialogVoiceMode = 'custom_voice' | 'voice_design'
type TtsQuality = 'quality' | 'speed'
type UploadTarget = 'asr' | 'tts-ref'

type AudioResultItem = {
  key: string
  runId?: string
  kind: 'audio' | 'text'
  status: 'pending' | 'completed' | 'failed'
  progress?: number
  title: string
  subtitle?: string
  prompt?: string
  url?: string
  text?: string
  fileName?: string
  mimeType?: string
  createdAt: string
}

type SpeakerDialogLine = {
  id: string
  speaker: string
  text: string
  language?: string
  instruct?: string
}

const tabs = computed<Array<{ label: string; value: AudioTab; description: string }>>(() => [
  { label: t('audio.tabs.asr.label'), value: 'asr', description: t('audio.tabs.asr.description') },
  { label: t('audio.tabs.tts.label'), value: 'tts', description: t('audio.tabs.tts.description') },
  { label: t('audio.tabs.dialog.label'), value: 'dialog', description: t('audio.tabs.dialog.description') },
])

const ttsFamilyOptions: Array<{ label: string; value: TtsFamily }> = [
  { label: 'Qwen TTS', value: 'qwen' },
  { label: 'VoxCPM', value: 'voxcpm' },
]

const asrModelOptions = [
  { label: 'Qwen3-ASR-0.6B', value: 'Qwen3-ASR-0.6B', load_name: 'Qwen3-ASR-0.6B' },
  { label: 'Qwen3-ASR-1.7B', value: 'Qwen3-ASR-1.7B', load_name: 'Qwen3-ASR-1.7B' },
]

const qwenQualityOptions = computed<Array<{ label: string; value: TtsQuality; description: string }>>(() => [
  { label: t('audio.qualityHigh'), value: 'quality', description: t('audio.qualityHighDesc') },
  { label: t('audio.qualitySpeed'), value: 'speed', description: t('audio.qualitySpeedDesc') },
])

const dialogVoiceModeOptions = computed<Array<{ label: string; value: DialogVoiceMode; description: string }>>(() => [
  { label: t('audio.dialogCustomVoice'), value: 'custom_voice', description: t('audio.dialogCustomVoiceDesc') },
  { label: t('audio.dialogVoiceDesign'), value: 'voice_design', description: t('audio.dialogVoiceDesignDesc') },
])

const ttsModeOptions = computed<Array<{ label: string; value: TtsMode }>>(() => [
  { value: 'random', label: t('audio.modeRandom') },
  { value: 'design', label: t('audio.modeDesign') },
  { value: 'clone', label: t('audio.modeClone') },
  { value: 'clone_plus', label: t('audio.modeClonePlus') },
])

type SpeakerSelectOption = { value: string; label: string; nativeLanguage?: string }

const fallbackQwenSpeakerOptions: SpeakerSelectOption[] = [
  { value: 'Vivian', label: 'Vivian - 明亮、略带锐气的年轻女声。中文', nativeLanguage: '中文' },
  { value: 'Serena', label: 'Serena - 温暖柔和的年轻女声。中文', nativeLanguage: '中文' },
  { value: 'Uncle_Fu', label: 'Uncle_Fu - 音色低沉醇厚的成熟男声。中文', nativeLanguage: '中文' },
  { value: 'Dylan', label: 'Dylan - 清晰自然的北京青年男声。中文（北京方言）', nativeLanguage: '中文（北京方言）' },
  { value: 'Eric', label: 'Eric - 活泼、略带沙哑明亮感的成都男声。中文（四川方言）', nativeLanguage: '中文（四川方言）' },
  { value: 'Ryan', label: 'Ryan - 富有节奏感的动态男声。英语', nativeLanguage: '英语' },
  { value: 'Aiden', label: 'Aiden - 清晰中频的阳光美式男声。英语', nativeLanguage: '英语' },
  { value: 'Ono_Anna', label: 'Ono_Anna - 轻快灵活的俏皮日语女声。日语', nativeLanguage: '日语' },
  { value: 'Sohee', label: 'Sohee - 富含情感的温暖韩语女声。韩语', nativeLanguage: '韩语' },
] 

const voxcpmModelOptions = [
  { label: 'VoxCPM2', value: 'VoxCPM2', load_name: 'VoxCPM2' },
  { label: 'VoxCPM1.5', value: 'VoxCPM1.5', load_name: '' },
  { label: 'VoxCPM-0.5B', value: 'VoxCPM-0.5B', load_name: 'VoxCPM-0.5B' },
]

const fallbackVoxcpmSpeakerOptions: SpeakerSelectOption[] = [
  { value: 'linda', label: 'linda — 中文女声' },
  { value: 'bbc', label: 'bbc — 英文新闻播报风格' },
  { value: 'luoli', label: 'luoli — 中文童声/萝莉' },
  { value: 'anchen', label: 'anchen — 中文成熟男声' },
  { value: 'bowen', label: 'bowen — 中文青年男声' },
  { value: 'xinran', label: 'xinran — 中文女声' },
  { value: 'samuel', label: 'samuel — 印度口音英语男声' },
  { value: 'alice', label: 'alice — 英文女声' },
  { value: 'carter', label: 'carter — 英文男声' },
  { value: 'fanbao', label: 'fanbao — 中文女声' },
]

const qwenSpeakerOptions = ref<SpeakerSelectOption[]>([...fallbackQwenSpeakerOptions])
const voxcpmSpeakerOptions = ref<SpeakerSelectOption[]>([...fallbackVoxcpmSpeakerOptions])

type AutoTtsModelSpec = {
  label: string
  loadName: string
  family: string
  reasonKey: string
}

const qwenCustomVoiceModels: Record<TtsQuality, AutoTtsModelSpec> = {
  quality: {
    label: 'Qwen3-TTS-12Hz-1.7B-CustomVoice',
    loadName: 'Qwen3-TTS-12Hz-1.7B-CustomVoice',
    family: 'qwen-tts',
    reasonKey: 'audio.reasons.customVoiceQuality',
  },
  speed: {
    label: 'Qwen3-TTS-12Hz-0.6B-CustomVoice',
    loadName: 'Qwen3-TTS-12Hz-0.6B-CustomVoice',
    family: 'qwen-tts',
    reasonKey: 'audio.reasons.customVoiceSpeed',
  },
}

const qwenBaseModels: Record<TtsQuality, AutoTtsModelSpec> = {
  quality: {
    label: 'Qwen3-TTS-12Hz-1.7B-Base',
    loadName: '',
    family: 'qwen-tts',
    reasonKey: 'audio.reasons.cloneQuality',
  },
  speed: {
    label: 'Qwen3-TTS-12Hz-0.6B-Base',
    loadName: 'Qwen3-TTS-12Hz-0.6B-Base',
    family: 'qwen-tts',
    reasonKey: 'audio.reasons.cloneSpeed',
  },
}

const qwenVoiceDesignModel: AutoTtsModelSpec = {
  label: 'Qwen3-TTS-12Hz-1.7B-VoiceDesign',
  loadName: 'Qwen3-TTS-12Hz-1.7B-VoiceDesign',
  family: 'qwen-tts',
  reasonKey: 'audio.reasons.voiceDesign',
}

function qwenTtsModelSpec(mode: TtsMode, quality: TtsQuality): AutoTtsModelSpec {
  if (mode === 'design') return qwenVoiceDesignModel
  if (mode === 'clone' || mode === 'clone_plus') return qwenBaseModels[quality]
  return qwenCustomVoiceModels[quality]
}

function qwenDialogModelSpec(mode: DialogVoiceMode, quality: TtsQuality): AutoTtsModelSpec {
  if (mode === 'voice_design') {
    return {
      ...qwenVoiceDesignModel,
      reasonKey: 'audio.reasons.dialogVoiceDesign',
    }
  }
  return qwenCustomVoiceModels[quality]
}

function voxcpmModelSpec(modelName: string): AutoTtsModelSpec {
  const option = voxcpmModelOptions.find((model) => model.value === modelName) || voxcpmModelOptions[0]!
  return {
    label: option.label,
    loadName: option.load_name,
    family: 'voxcpm',
    reasonKey: 'audio.reasons.voxcpmShared',
  }
}

const activeTab = ref<AudioTab>('asr')
const activeTabClass = 'bg-gray-950 text-white shadow-sm [.dark_&]:bg-white [.dark_&]:text-gray-950'
const inactiveTabClass = 'bg-gray-100 text-gray-600 hover:bg-gray-200 hover:text-gray-950 [.dark_&]:bg-gray-800 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-700 [.dark_&]:hover:text-white'

const asrForm = reactive({
  modelName: 'Qwen3-ASR-1.7B',
  audioUrl: '',
  audioName: '',
  language: '',
  timestamps: true,
})

const ttsForm = reactive({
  family: 'qwen' as TtsFamily,
  quality: 'quality' as TtsQuality,
  voxcpmModelName: 'VoxCPM2',
  voxcpmSpeakerName: 'linda',
  mode: 'random' as TtsMode,
  speakerName: 'Vivian',
  prompt: '',
  instruct: '',
  refAudioUrl: '',
  refAudioName: '',
  refText: '',
})

const dialogForm = reactive({
  mode: 'custom_voice' as DialogVoiceMode,
})

const dialogLines = ref<SpeakerDialogLine[]>([
  { id: 'speaker-1', speaker: 'Vivian', text: '', language: 'Chinese', instruct: '' },
  { id: 'speaker-2', speaker: 'Ryan', text: '', language: 'English', instruct: '' },
])

const loading = ref(false)
const uploadingField = ref<UploadTarget | ''>('')
const textFileReading = ref(false)
const textFileDragActive = ref(false)
const textFileInputRef = ref<HTMLInputElement | null>(null)
const results = ref<AudioResultItem[]>([])
const { generatingCount, runTask, disconnectAll } = useTaskRunManager({ placeholderPrefix: 'audio' })

const currentTab = computed(() => tabs.value.find((tab) => tab.value === activeTab.value) || tabs.value[0]!)
const currentTabTitle = computed(() => currentTab.value.label)
const currentTabDescription = computed(() => currentTab.value.description)
const isGenerating = computed(() => generatingCount.value > 0)
const needsInstruct = computed(() => ttsForm.mode === 'design')
const needsReferenceAudio = computed(() => ttsForm.mode === 'clone' || ttsForm.mode === 'clone_plus')
const needsReferenceText = computed(() => ttsForm.mode === 'clone_plus')
const qwenSupportsQuality = computed(() => (activeTab.value === 'dialog' && dialogForm.mode === 'custom_voice') || (activeTab.value === 'tts' && ttsForm.mode !== 'design'))
const qwenAutoModelSpec = computed(() => (activeTab.value === 'dialog' ? qwenDialogModelSpec(dialogForm.mode, ttsForm.quality) : qwenTtsModelSpec(ttsForm.mode, ttsForm.quality)))
const selectedTtsModel = computed(() => (ttsForm.family === 'voxcpm' ? voxcpmModelSpec(ttsForm.voxcpmModelName) : qwenAutoModelSpec.value))
const selectedAsrModel = computed(() => asrModelOptions.find((model) => model.value === asrForm.modelName) || asrModelOptions[0]!)
const autoTtsModelDisplayName = computed(() => {
  const spec = selectedTtsModel.value
  return spec.loadName || spec.label
})
const autoTtsModelReason = computed(() => {
  const spec = selectedTtsModel.value
  let reason = t(spec.reasonKey)
  if (activeTab.value === 'dialog' && dialogForm.mode === 'custom_voice') {
    reason += t('audio.reasons.dialogBatchSuffix')
  } else if (activeTab.value === 'tts' && ttsForm.family === 'qwen' && ttsForm.mode === 'clone_plus') {
    reason += t('audio.reasons.clonePlusSuffix')
  }
  return reason
})

const submitDisabledReason = computed(() => {
  if (loading.value) return t('common.submittingTask')
  if (textFileReading.value) return t('audio.textFileReading')
  if (isGenerating.value) return t('audio.taskProcessing')
  if (uploadingField.value) return t('audio.audioUploading')
  if (activeTab.value === 'asr') {
    if (!asrForm.audioUrl) return t('audio.uploadAsrFirst')
    if (!selectedAsrModel.value.load_name) return t('audio.modelNotRegistered', { name: selectedAsrModel.value.label })
    return ''
  }
  if (activeTab.value === 'dialog') {
    const validLines = normalizedDialogLines.value
    if (ttsForm.family !== 'qwen') return t('audio.dialogQwenOnly')
    if (validLines.length === 0) return t('audio.enterDialogLine')
    if (dialogForm.mode === 'voice_design' && validLines.some((line) => !line.instruct)) return t('audio.dialogInstructRequired')
    if (!selectedTtsModel.value.loadName) return t('audio.modelNotRegistered', { name: autoTtsModelDisplayName.value })
    return ''
  }
  if (!ttsForm.prompt.trim()) return t('audio.enterTextToSynthesize')
  if (!selectedTtsModel.value.loadName) return t('audio.modelNotRegistered', { name: autoTtsModelDisplayName.value })
  if (needsInstruct.value && !ttsForm.instruct.trim()) return t('audio.enterInstruct')
  if (needsReferenceAudio.value && !ttsForm.refAudioUrl) return t('audio.uploadReferenceAudio')
  if (needsReferenceText.value && !ttsForm.refText.trim()) return t('audio.enterReferenceText')
  return ''
})

const submitDisabled = computed(() => Boolean(submitDisabledReason.value))
const submitButtonText = computed(() => {
  if (loading.value) return t('common.submitting')
  if (isGenerating.value) return t('common.processing')
  if (activeTab.value === 'asr') return t('audio.extractText')
  return t('audio.generate')
})

const normalizedDialogLines = computed(() =>
  dialogLines.value
    .map((line) => ({
      speaker: String(line.speaker || '').trim() || 'Vivian',
      text: String(line.text || '').trim(),
      language: String(line.language || '').trim(),
      instruct: String(line.instruct || '').trim(),
    }))
    .filter((line) => line.text)
)

function formatNow() {
  return new Date().toLocaleString('zh-CN', { hour12: false })
}

function fileKind(file: any): 'audio' | 'text' {
  const mime = String(file?.mime_type || '').toLowerCase()
  const name = String(file?.file_name || file?.storage_path || '').toLowerCase()
  if (mime.startsWith('text/') || name.endsWith('.txt') || name.endsWith('.json') || name.endsWith('.srt')) return 'text'
  return 'audio'
}

function normalizeUrl(url: string) {
  const raw = String(url || '').trim()
  if (!raw) return ''
  if (raw.startsWith('http://') || raw.startsWith('https://')) return raw
  return raw.startsWith('/') ? `${window.location.origin}${raw}` : `${window.location.origin}/${raw}`
}

function filenameFromUrl(url: string) {
  const raw = String(url || '').trim()
  if (!raw) return ''
  try {
    const parsed = new URL(raw, window.location.href)
    return decodeURIComponent(parsed.pathname.split('/').filter(Boolean).pop() || '')
  } catch {
    return raw.split(/[\\/]/).filter(Boolean).pop() || ''
  }
}

function applyCachedAsrTask(cached: TaskCreateRequest) {
  const loadName = String(cached.load_name || '').trim()
  const model = asrModelOptions.find((item) => item.load_name === loadName || item.value === loadName)
  if (model) asrForm.modelName = model.value

  const inputUrl = String((cached as any).input_audio_url || '').trim()
  if (inputUrl) {
    asrForm.audioUrl = inputUrl
    asrForm.audioName = filenameFromUrl(inputUrl)
  }
  if (typeof cached.language === 'string') asrForm.language = cached.language
  if (cached.timestamps !== undefined) asrForm.timestamps = Boolean(cached.timestamps)
}

function resolveTtsFamilyFromCache(cached: TaskCreateRequest): TtsFamily {
  const family = String(cached.family || '').trim().toLowerCase()
  const loadName = String(cached.load_name || '').trim()
  if (
    family === 'voxcpm' ||
    voxcpmModelOptions.some((item) => item.load_name === loadName || item.value === loadName)
  ) {
    return 'voxcpm'
  }
  return 'qwen'
}

function applyCachedTtsModel(cached: TaskCreateRequest) {
  const family = resolveTtsFamilyFromCache(cached)
  ttsForm.family = family

  const loadName = String(cached.load_name || '').trim()
  if (family === 'voxcpm') {
    const model = voxcpmModelOptions.find((item) => item.load_name === loadName || item.value === loadName)
    if (model) ttsForm.voxcpmModelName = model.value
    return
  }

  const qualityByModel = (spec: AutoTtsModelSpec) => spec.loadName === loadName
  if (qualityByModel(qwenCustomVoiceModels.speed) || qualityByModel(qwenBaseModels.speed)) {
    ttsForm.quality = 'speed'
  } else {
    ttsForm.quality = 'quality'
  }
}

function applyCachedTtsTask(cached: TaskCreateRequest) {
  applyCachedTtsModel(cached)
  if (typeof cached.prompt === 'string') ttsForm.prompt = cached.prompt

  const ttsMode = String((cached as any).tts_mode || '').trim()
  if (ttsMode === 'voice_design') ttsForm.mode = 'design'
  else if ((cached as any).ref_text !== undefined || (cached as any).prompt_text !== undefined) ttsForm.mode = 'clone_plus'
  else if ((cached as any).ref_audio !== undefined || (cached as any).prompt_wav_path !== undefined) ttsForm.mode = 'clone'
  else ttsForm.mode = 'random'

  if (typeof (cached as any).speaker_name === 'string') {
    const speakerName = String((cached as any).speaker_name || '')
    if (ttsForm.family === 'voxcpm') ttsForm.voxcpmSpeakerName = speakerName
    else ttsForm.speakerName = speakerName
  }
  if (typeof (cached as any).instruct === 'string') ttsForm.instruct = String((cached as any).instruct || '')

  const refAudio = String((cached as any).ref_audio || (cached as any).prompt_wav_path || '').trim()
  if (refAudio) {
    ttsForm.refAudioUrl = refAudio
    ttsForm.refAudioName = filenameFromUrl(refAudio)
  }
  const refText = String((cached as any).ref_text || (cached as any).prompt_text || '').trim()
  if (refText) ttsForm.refText = refText
}

function applyCachedDialogTask(cached: TaskCreateRequest) {
  applyCachedTtsModel(cached)
  const ttsMode = String((cached as any).tts_mode || 'custom_voice').trim()
  dialogForm.mode = ttsMode === 'voice_design' ? 'voice_design' : 'custom_voice'

  const drama = (cached as any).drama
  const characters = Array.isArray(drama?.characters) ? drama.characters : []
  const dialogues = Array.isArray(drama?.dialogues) ? drama.dialogues : []
  const characterById = new Map(characters.map((item: any) => [String(item?.id || ''), item]))
  const lines = dialogues
    .map((dialogue: any, index: number) => {
      const character = characterById.get(String(dialogue?.speaker_id || '')) as any
      const text = String(dialogue?.text || '').trim()
      if (!text) return null
      return {
        id: `speaker-${index + 1}`,
        speaker: String(character?.name || '').trim() || t('audio.characterFallback', { index: index + 1 }),
        text,
        language: String(character?.language || '').trim() || undefined,
        instruct: String(dialogue?.instruct || character?.instruct || '').trim() || undefined,
      }
    })
    .filter(Boolean) as SpeakerDialogLine[]

  if (lines.length) dialogLines.value = lines
}

function applyCachedAudioTask(cached: TaskCreateRequest | null | undefined) {
  if (!cached || cached.task_type !== 'audio') return
  const jobType = String(cached.job_type || '').trim().toUpperCase()
  if (jobType === 'ASR') {
    activeTab.value = 'asr'
    applyCachedAsrTask(cached)
    return
  }
  if (jobType === 'TTS') {
    const hasDrama = (cached as any).drama && typeof (cached as any).drama === 'object'
    activeTab.value = hasDrama ? 'dialog' : 'tts'
    if (hasDrama) applyCachedDialogTask(cached)
    else applyCachedTtsTask(cached)
  }
}

function isSupportedPromptTextFile(file: File) {
  const name = file.name.toLowerCase()
  return name.endsWith('.txt') || name.endsWith('.md') || name.endsWith('.markdown') || name.endsWith('.docx')
}

async function readPromptTextFile(file: File) {
  const name = file.name.toLowerCase()
  if (name.endsWith('.docx')) {
    const result = await mammoth.extractRawText({ arrayBuffer: await file.arrayBuffer() })
    return result.value
  }
  return file.text()
}

async function applyPromptTextFile(file: File) {
  if (!isSupportedPromptTextFile(file)) {
    showTopSnack(t('audio.selectTextFile'))
    return
  }

  textFileReading.value = true
  try {
    const text = await readPromptTextFile(file)
    const content = text.trim()
    if (!content) {
      showTopSnack(t('audio.textFileEmpty'))
      return
    }
    ttsForm.prompt = content
    showTopSnack(t('audio.textImported', { name: file.name }))
  } catch (error: any) {
    showTopSnackError(t('audio.readTextFileFailed', { msg: error?.message || String(error) }))
  } finally {
    textFileReading.value = false
  }
}

async function handleTextFilePicked(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (!file) return
  await applyPromptTextFile(file)
}

function canHandleTextFileDrop(event: DragEvent) {
  if (textFileReading.value || loading.value || isGenerating.value) return false
  return Array.from(event.dataTransfer?.types || []).includes('Files')
}

function handleTextFileDragEnter(event: DragEvent) {
  if (!canHandleTextFileDrop(event)) return
  textFileDragActive.value = true
}

function handleTextFileDragOver(event: DragEvent) {
  if (!canHandleTextFileDrop(event)) return
  textFileDragActive.value = true
  if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy'
}

function handleTextFileDragLeave(event: DragEvent) {
  const current = event.currentTarget as HTMLElement | null
  const related = event.relatedTarget as Node | null
  if (current && related && current.contains(related)) return
  textFileDragActive.value = false
}

async function handleTextFileDrop(event: DragEvent) {
  textFileDragActive.value = false
  if (textFileReading.value || loading.value || isGenerating.value) return
  const file = event.dataTransfer?.files?.[0]
  if (!file) return
  await applyPromptTextFile(file)
}

async function handleAudioUpload(file: File, target: UploadTarget) {
  if (!file.type.startsWith('audio/') && !/\.(wav|mp3|m4a|flac|ogg|aac)$/i.test(file.name)) {
    showTopSnack(t('audio.selectAudioFile'))
    return
  }

  uploadingField.value = target
  try {
    const url = await uploadFile(file)
    if (target === 'asr') {
      asrForm.audioUrl = url
      asrForm.audioName = file.name
    } else {
      ttsForm.refAudioUrl = url
      ttsForm.refAudioName = file.name
    }
    showTopSnack(t('audio.audioUploadSuccess'))
  } catch (error: any) {
    showTopSnackError(t('common.uploadFailed', { msg: error?.message || String(error) }))
  } finally {
    uploadingField.value = ''
  }
}

function clearAsrAudio() {
  asrForm.audioUrl = ''
  asrForm.audioName = ''
}

function clearTtsRefAudio() {
  ttsForm.refAudioUrl = ''
  ttsForm.refAudioName = ''
}

function buildAsrRequest(): TaskCreateRequest {
  return {
    task_type: 'audio',
    audio_mode: 'asr',
    job_type: 'ASR',
    input_audio_url: normalizeUrl(asrForm.audioUrl),
    response_format: 'both',
    language: asrForm.language || undefined,
    timestamps: Boolean(asrForm.timestamps),
    load_name: selectedAsrModel.value.load_name,
    family: 'Qwen-asr',
  }
}

function qwenTtsMode() {
  if (ttsForm.mode === 'design') return 'voice_design'
  if (ttsForm.mode === 'clone' || ttsForm.mode === 'clone_plus') return 'voice_clone'
  return 'custom_voice'
}

function toSpeakerSelectOptions(items: TtsSpeakerOption[] | undefined): SpeakerSelectOption[] {
  return (items || []).reduce<SpeakerSelectOption[]>((acc, item) => {
    const value = String(item.name || '').trim()
    if (!value) return acc
    const description = String(item.description || '').trim()
    const language = String(item.language || '').trim()
    const detail = [description, language].filter(Boolean).join('。')
    acc.push({
      value,
      label: detail ? `${item.label || value} - ${detail}` : String(item.label || value),
      nativeLanguage: language || undefined,
    })
    return acc
  }, [])
}

async function loadSharedTtsSpeakers() {
  try {
    const catalog = await listTtsSpeakers()
    const qwenOptions = toSpeakerSelectOptions(catalog.families?.qwen?.speakers)
    const voxcpmOptions = toSpeakerSelectOptions(catalog.families?.voxcpm?.speakers)
    if (qwenOptions.length > 0) qwenSpeakerOptions.value = qwenOptions
    if (voxcpmOptions.length > 0) voxcpmSpeakerOptions.value = voxcpmOptions

    const qwenDefault = String(catalog.families?.qwen?.default_speaker || '').trim()
    const voxcpmDefault = String(catalog.families?.voxcpm?.default_speaker || '').trim()
    if (qwenDefault && qwenSpeakerOptions.value.some((item) => item.value === qwenDefault)) {
      ttsForm.speakerName = qwenDefault
    }
    if (voxcpmDefault && voxcpmSpeakerOptions.value.some((item) => item.value === voxcpmDefault)) {
      ttsForm.voxcpmSpeakerName = voxcpmDefault
    }
  } catch (error) {
    console.warn('[audio] failed to load shared tts speakers, using fallback options', error)
  }
}

function buildTtsRequest(): TaskCreateRequest {
  const model = selectedTtsModel.value
  const req: TaskCreateRequest = {
    task_type: 'audio',
    audio_mode: 'tts',
    job_type: 'TTS',
    prompt: ttsForm.prompt.trim(),
    response_format: 'audio_file',
    load_name: model.loadName,
    family: model.family,
  }

  if (ttsForm.family === 'qwen') {
    req.tts_mode = qwenTtsMode()
    if (ttsForm.mode === 'random') req.speaker_name = ttsForm.speakerName
    if (needsInstruct.value) req.instruct = ttsForm.instruct.trim()
    if (needsReferenceAudio.value) req.ref_audio = normalizeUrl(ttsForm.refAudioUrl)
    if (needsReferenceText.value) req.ref_text = ttsForm.refText.trim()
    if (ttsForm.mode === 'clone') req.x_vector_only = true
  } else {
    req.tts_mode = ttsForm.mode === 'design' ? 'voice_design' : 'custom_voice'
    if (ttsForm.mode === 'random') req.speaker_name = ttsForm.voxcpmSpeakerName
    if (needsInstruct.value) req.instruct = ttsForm.instruct.trim()
    if (needsReferenceAudio.value) req.prompt_wav_path = normalizeUrl(ttsForm.refAudioUrl)
    if (needsReferenceText.value) req.prompt_text = ttsForm.refText.trim()
  }

  return req
}

function buildDramaPayload(lines: Array<{ speaker: string; text: string; language: string; instruct: string }>) {
  const characterIds = new Map<string, string>()
  const characters: Array<Record<string, string>> = []
  const dialogues: Array<Record<string, string>> = []

  lines.forEach((line, index) => {
    const voiceMode = dialogForm.mode
    const characterKey = [
      voiceMode,
      line.speaker || `role-${index + 1}`,
      line.language,
      line.instruct,
    ].join('\u0000')
    let characterId = characterIds.get(characterKey)
    if (!characterId) {
      characterId = `role_${characters.length + 1}`
      characterIds.set(characterKey, characterId)
      characters.push({
        id: characterId,
        name: line.speaker || t('audio.characterFallback', { index: characters.length + 1 }),
        voice_mode: voiceMode,
        ...(voiceMode === 'custom_voice' ? { speaker_name: line.speaker } : {}),
        ...(line.instruct ? { instruct: line.instruct } : {}),
        ...(line.language ? { language: line.language } : {}),
      })
    }
    dialogues.push({
      speaker_id: characterId,
      text: line.text,
      ...(line.instruct ? { instruct: line.instruct } : {}),
    })
  })

  return { characters, dialogues }
}

function buildDialogRequest(): TaskCreateRequest {
  const model = selectedTtsModel.value
  const lines = normalizedDialogLines.value
  const prompt = dialogForm.mode === 'voice_design'
    ? lines.map((line) => line.text).join('\n')
    : lines.map((line) => `${line.speaker}: ${line.text}`).join('\n')
  return {
    task_type: 'audio',
    audio_mode: 'tts',
    job_type: 'TTS',
    tts_mode: dialogForm.mode,
    prompt,
    drama: {
      title: t('audio.dialogTitle'),
      synopsis: prompt,
      ...buildDramaPayload(lines),
    },
    response_format: 'audio_file',
    load_name: model.loadName,
    family: model.family,
  }
}

function createAudioRunHooks(req: TaskCreateRequest): RunTaskHooks {
  return {
    onAddPlaceholders: (keys, runId) => {
      const items = keys.map<AudioResultItem>((key) => ({
        key,
        runId,
        kind: activeTab.value === 'asr' ? 'text' : 'audio',
        status: 'pending',
        title: activeTab.value === 'asr' ? t('audio.asrProcessing') : t('audio.audioGenerating'),
        subtitle: req.family ? String(req.family) : undefined,
        prompt: String(req.prompt || ''),
        createdAt: formatNow(),
      }))
      results.value = [...items, ...results.value]
    },
    onProgress: ({ runId, progress }) => {
      if (typeof progress !== 'number' || !Number.isFinite(progress)) return
      const nextProgress = Math.max(0, Math.min(100, Math.round(progress)))
      results.value = results.value.map((item) =>
        item.status === 'pending' && item.runId === runId ? { ...item, progress: nextProgress } : item
      )
    },
    onRemoveKeys: (keys) => {
      const keySet = new Set(keys)
      results.value = results.value.filter((item) => !keySet.has(item.key))
    },
    onResultFile: ({ file, msg, replaceKey }) => {
      const url = getOriginalFileUrl(file)
      if (!url) return false
      const kind = fileKind(file)
      const item: AudioResultItem = {
        key: String(url),
        kind,
        status: 'completed',
        title: String(file?.file_name || (kind === 'audio' ? t('audio.generatedAudio') : t('audio.recognizedText'))),
        subtitle: String(file?.mime_type || (msg as any)?.family || req.family || ''),
        prompt: String(req.prompt || ''),
        url,
        text: kind === 'text' ? String(file?.text || (msg as any)?.text || (msg as any)?.transcript || '') : undefined,
        fileName: file?.file_name ? String(file.file_name) : undefined,
        mimeType: file?.mime_type ? String(file.mime_type) : undefined,
        createdAt: formatNow(),
      }

      if (replaceKey) {
        const index = results.value.findIndex((result) => result.key === replaceKey)
        if (index !== -1) {
          const next = [...results.value]
          next.splice(index, 1, item)
          results.value = next
          return true
        }
      }

      results.value = [item, ...results.value]
      return true
    },
    onError: (err) => {
      showTopSnackError(err.phase === 'create' ? t('common.submitFailed', { msg: err.message }) : err.message)
    },
  }
}

async function submitAudioTask() {
  if (submitDisabled.value) {
    showTopSnack(submitDisabledReason.value || t('common.cannotSubmit'))
    return
  }

  loading.value = true
  try {
    const req = activeTab.value === 'asr' ? buildAsrRequest() : activeTab.value === 'dialog' ? buildDialogRequest() : buildTtsRequest()
    await runTask(req, 1, createAudioRunHooks(req))
  } catch (error: any) {
    showTopSnackError(t('common.submitFailed', { msg: error?.response?.data?.detail || error?.message || String(error) }))
  } finally {
    loading.value = false
  }
}

async function downloadResult(result: AudioResultItem) {
  if (!result.url) {
    showTopSnack(t('common.noFileToDownload'))
    return
  }
  const ok = await downloadFile({
    url: result.url,
    filename: result.fileName,
    mediaType: result.kind,
    title: result.title,
  })
  showTopSnack(ok ? t('common.startDownload') : t('common.downloadFailed'))
}

async function loadTextResult(result: AudioResultItem) {
  if (!result.url) return
  try {
    const resp = await fetch(result.url)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
    const text = await resp.text()
    results.value = results.value.map((item) => (item.key === result.key ? { ...item, text } : item))
  } catch (error: any) {
    showTopSnackError(t('audio.readTextFailed', { msg: error?.message || String(error) }))
  }
}

watch(
  () => activeTab.value,
  (tab) => {
    if (tab === 'dialog') ttsForm.family = 'qwen'
  }
)

onMounted(() => {
  void loadSharedTtsSpeakers()
  const cached = getLocalCache<TaskCreateRequest>(taskCreateCacheKeyByType('audio'))
  applyCachedAudioTask(cached)
})

onBeforeUnmount(() => {
  disconnectAll()
})
</script>

