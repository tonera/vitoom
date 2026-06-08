<template>
  <Teleport to="body">
    <div v-if="open" class="fixed inset-0 z-9999 bg-black/60 flex items-center justify-center p-4">
      <div class="vt-card w-full max-w-[760px] max-h-[min(80vh,860px)] overflow-y-auto vt-scroll p-5 rounded-2xl">
        <div class="flex items-start justify-between gap-4">
          <div>
            <div class="text-lg font-semibold text-gray-900 [.dark_&]:text-white">
              {{ t('models.import.title') }}
            </div>
          </div>
          <button
            type="button"
            class="cursor-pointer rounded-lg p-2 text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-900 [.dark_&]:text-gray-400 [.dark_&]:hover:bg-gray-800/60 [.dark_&]:hover:text-white"
            :aria-label="t('common.close')"
            @click="close()"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div class="mt-4 space-y-3">
          <div class="space-y-1">
            <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.import.linkLabel') }}</div>
            <textarea
              v-model="civitaiInput"
              rows="3"
              :placeholder="t('models.import.linkPlaceholder')"
              class="w-full rounded-xl border border-gray-200 bg-gray-50 px-3 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
            />
          </div>

          <div class="flex items-center justify-end gap-2">
            <button
              type="button"
              class="cursor-pointer rounded-xl border border-gray-200 bg-white px-4 py-2 text-gray-700 transition-colors hover:bg-gray-50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/50 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800/80"
              @click="close()"
            >
              {{ t('common.cancel') }}
            </button>
            <button
              type="button"
              class="cursor-pointer rounded-xl bg-indigo-600 px-4 py-2 text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
              :disabled="civitaiLoading || !civitaiInput.trim()"
              @click="fetchImportModelInfo()"
            >
              {{ civitaiLoading ? t('models.import.fetching') : t('models.import.fetchInfo') }}
            </button>
          </div>

          <div v-if="civitaiError" class="text-sm text-rose-600 [.dark_&]:text-rose-200">
            {{ civitaiError }}
          </div>
        </div>

        <div v-if="hasPreview" class="mt-6 space-y-4">
          <div class="vt-card-muted p-4 space-y-2">
            <div class="flex items-start justify-between gap-3">
              <div class="min-w-0">
                <div class="truncate text-sm font-semibold text-gray-900 [.dark_&]:text-white">
                  {{ previewTitle }}
                </div>
                <div class="mt-1 text-xs text-gray-500 [.dark_&]:text-gray-400">
                  <span class="mr-3"
                    >{{ t('models.import.source') }}: <span class="text-gray-800 [.dark_&]:text-gray-200">{{ previewProvider }}</span></span
                  >
                  <span class="mr-3"
                    >{{ previewIdLabel }}: <span class="text-gray-800 [.dark_&]:text-gray-200">{{ previewId }}</span></span
                  >
                </div>
              </div>
              <a
                v-if="previewLink"
                class="text-xs text-indigo-600 underline underline-offset-4 hover:text-indigo-800 [.dark_&]:text-indigo-200 [.dark_&]:hover:text-white"
                :href="previewLink"
                target="_blank"
                rel="noreferrer"
              >
                {{ t('models.import.openPage') }}
              </a>
            </div>
            <div v-if="previewSub" class="whitespace-pre-wrap text-xs text-gray-600 [.dark_&]:text-gray-400">
              {{ previewSub }}
            </div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="space-y-1">
              <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.shortName') }}</div>
              <input
                v-model="civitaiForm.name"
                type="text"
                class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
              />
            </div>

            <div class="space-y-1">
              <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.name') }}</div>
              <input
                v-model="civitaiForm.load_name"
                type="text"
                class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
              />
            </div>

            <div class="space-y-1">
              <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.weightType') }}</div>
              <select
                v-model="civitaiForm.asset_type"
                class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
              >
                <option value="lora">lora</option>
                <option value="checkpoint">checkpoint</option>
              </select>
            </div>

            <div class="space-y-1">
              <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.category') }}</div>
              <select
                v-model="civitaiForm.modality"
                class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
              >
                <option v-for="opt in modalityOptions" :key="`mod-${opt.value}`" :value="opt.value">
                  {{ opt.label }}
                </option>
              </select>
            </div>

            <div class="space-y-1">
              <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.modelType') }}</div>
              <select
                v-model="civitaiForm.storage_mode"
                class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
              >
                <option value="local">{{ t('models.storageLocal') }}</option>
                <option value="cloud">{{ t('models.storageCloud') }}</option>
              </select>
            </div>

            <div class="space-y-1">
              <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.family') }}</div>
              <select
                v-model="civitaiForm.family"
                class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
              >
                <option value="">{{ t('models.form.pleaseSelect') }}</option>
                <option v-for="opt in familyOptions" :key="`mc-opt-${opt.value}`" :value="opt.value">
                  {{ opt.label }}
                </option>
              </select>
            </div>

            <div class="space-y-1 md:col-span-2">
              <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.thumbOptional') }}</div>
              <div v-if="thumbCandidates.length" class="mb-2 flex gap-2 flex-nowrap overflow-x-auto pb-2 vt-scroll">
                <button
                  v-for="(u, idx) in thumbCandidates"
                  :key="`thumb-${idx}`"
                  type="button"
                  class="flex-none cursor-pointer overflow-hidden rounded-lg border"
                  :class="
                    thumbInput === u
                      ? 'border-indigo-500/70 ring-2 ring-indigo-500/20'
                      : 'border-gray-200 hover:border-gray-400 [.dark_&]:border-gray-700/70 [.dark_&]:hover:border-gray-500/70'
                  "
                  :title="u"
                  @click="thumbInput = u"
                >
                  <img :src="u" class="h-20 w-20 object-cover" :alt="t('models.form.thumbPreviewAlt')" referrerpolicy="no-referrer" />
                </button>
              </div>
              <div class="flex gap-3">
                <div class="flex-1">
                  <input
                    v-model="thumbInput"
                    type="text"
                    :placeholder="t('models.form.thumbPlaceholder')"
                    class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
                  />
                </div>
                <div v-if="thumbPreviewUrl" class="flex-none">
                  <img
                    :src="thumbPreviewUrl"
                    :alt="t('models.form.thumbPreviewAlt')"
                    class="h-20 w-20 rounded-lg border border-gray-200 object-cover [.dark_&]:border-gray-700/70"
                    referrerpolicy="no-referrer"
                    @error="handleThumbError"
                  />
                </div>
              </div>
            </div>

            <div class="space-y-1 md:col-span-2">
              <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.triggerWords') }}</div>
              <input
                v-model="civitaiForm.trigger_words"
                type="text"
                class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
              />
            </div>

            <div class="space-y-1 md:col-span-2">
              <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.description') }}</div>
              <textarea
                v-model="civitaiForm.description"
                rows="3"
                class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
              />
            </div>
          </div>

          <div class="flex items-center justify-end gap-2">
            <button
              type="button"
              class="cursor-pointer rounded-xl border border-gray-200 bg-white px-4 py-2 text-gray-700 transition-colors hover:bg-gray-50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/50 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800/80"
              @click="close()"
            >
              {{ t('common.cancel') }}
            </button>
            <button
              type="button"
              class="cursor-pointer rounded-xl border border-gray-200 bg-white px-4 py-2 text-gray-700 transition-colors hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-60 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/50 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800/80"
              :disabled="civitaiSubmitting || !canConfirm"
              @click="confirmSave()"
            >
              {{ civitaiSubmitting && submitAction === 'save' ? t('models.import.saving') : t('common.save') }}
            </button>
            <button
              type="button"
              class="cursor-pointer rounded-xl bg-indigo-600 px-4 py-2 text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
              :disabled="civitaiSubmitting || !canConfirm"
              @click="confirmImport()"
            >
              {{ civitaiSubmitting && submitAction === 'import' ? t('models.import.importing') : t('models.import.confirmImport') }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useModelCatalogMeta } from '../../../composables/useModelCatalogMeta'
import { showTopSnack } from '../../../composables/useTopSnack'
import { createModelWithMeta, downloadActionModel, getRemoteModelInfo, type CreateModelBody } from '../../../api/models'
import { resolveModelThumbUrl } from '../../../utils/modelThumb'

type CivitaiImage = { url: string; [k: string]: any }
type CivitaiModel = { name?: string; type?: string; nsfw?: boolean; [k: string]: any }
type CivitaiFile = {
  id?: number
  sizeKB?: number
  name?: string
  type?: string
  [k: string]: any
}
type CivitaiFamily = {
  id: number
  name?: string
  createdAt?: string
  baseModel?: string
  trainedWords?: string[]
  images?: CivitaiImage[]
  files?: CivitaiFile[]
  stats?: any
  model?: CivitaiModel
  description?: string
  [k: string]: any
}

const props = defineProps<{
  open: boolean
}>()

const emit = defineEmits<{
  (e: 'close'): void
  (e: 'done'): void
}>()

const { t } = useI18n()
const { modalityOptions, familyOptions, ensureLoaded: ensureCatalogMetaLoaded, normalizeFamily } = useModelCatalogMeta()

const civitaiInput = ref('')
const civitaiLoading = ref(false)
const civitaiSubmitting = ref(false)
const submitAction = ref<'save' | 'import' | ''>('')
const civitaiError = ref('')
const civitaiInfo = ref<CivitaiFamily | null>(null)
const remoteProvider = ref<string>('')
const remoteRepoId = ref<string>('')
const remoteInfo = ref<any>(null)
const remoteThumbCandidates = ref<string[]>([])
const thumbInput = ref<string>('')
// 保护：同一次弹窗确认流程，只创建一次模型记录；若下载失败，允许重试下载而不是重复插入
const civitaiCreatedModelKey = ref<string>('')
const civitaiForm = ref<{
  name: string
  load_name: string
  modality: string
  storage_mode: string
  family: string
  asset_type: string
  trigger_words: string
  description: string
}>({
  name: '',
  load_name: '',
  modality: 'image',
  storage_mode: 'local',
  family: '',
  asset_type: 'checkpoint',
  trigger_words: '',
  description: '',
})

watch(
  () => props.open,
  (open) => {
    if (!open) return
    void ensureCatalogMetaLoaded()
    civitaiInput.value = ''
    civitaiLoading.value = false
    civitaiSubmitting.value = false
    submitAction.value = ''
    civitaiError.value = ''
    civitaiInfo.value = null
    remoteProvider.value = ''
    remoteRepoId.value = ''
    remoteInfo.value = null
    remoteThumbCandidates.value = []
    thumbInput.value = ''
    civitaiCreatedModelKey.value = ''
    civitaiForm.value = {
      name: '',
      load_name: '',
      modality: 'image',
      storage_mode: 'local',
      family: '',
      asset_type: 'checkpoint',
      trigger_words: '',
      description: '',
    }
  },
  { immediate: true }
)

function close() {
  if (civitaiLoading.value || civitaiSubmitting.value) return
  emit('close')
}

const civitaiVersionLink = computed(() => {
  const vid = civitaiInfo.value?.id
  if (!vid) return ''
  return `https://civitai.com/api/v1/model-versions/${vid}`
})

const isCivitaiPreview = computed(() => Boolean(civitaiInfo.value))
const hasPreview = computed(() => Boolean(civitaiInfo.value || remoteInfo.value))
const previewProvider = computed(() => (isCivitaiPreview.value ? 'civitai' : String(remoteProvider.value || '').trim()))
const previewIdLabel = computed(() => (isCivitaiPreview.value ? t('models.import.versionId') : t('models.import.repoId')))
const previewId = computed(() => (isCivitaiPreview.value ? String(civitaiInfo.value?.id || '') : String(remoteRepoId.value || '').trim()))
const previewTitle = computed(() => (isCivitaiPreview.value ? String(civitaiInfo.value?.model?.name || t('models.import.unknownModel')) : remoteDisplayName.value))
const previewLink = computed(() => (isCivitaiPreview.value ? civitaiVersionLink.value : remoteLink.value))
const previewSub = computed(() => {
  if (isCivitaiPreview.value) {
    const bm = String(civitaiInfo.value?.baseModel || '').trim()
    const tw = Array.isArray(civitaiInfo.value?.trainedWords) ? civitaiInfo.value?.trainedWords || [] : []
    const tws = tw.slice(0, 20).join(', ')
    const parts = [bm ? `baseModel: ${bm}` : '', tws ? `trainedWords: ${tws}${tw.length > 20 ? ' ...' : ''}` : ''].filter(Boolean)
    return parts.join('\n')
  }
  return String(remoteSummary.value || '').trim()
})

const thumbCandidates = computed(() => {
  if (isCivitaiPreview.value) {
    const imgs = civitaiInfo.value?.images || []
    return imgs.map((x) => String((x as any)?.url || '').trim()).filter(Boolean).slice(0, 10)
  }
  return (remoteThumbCandidates.value || []).map((x) => String(x || '').trim()).filter(Boolean).slice(0, 10)
})

const thumbPreviewUrl = computed(() => resolveModelThumbUrl(String(thumbInput.value || '')))

function handleThumbError(event: Event) {
  const img = event.target as HTMLImageElement
  img.style.display = 'none'
}

const canConfirm = computed(
  () =>
    hasPreview.value &&
    String(civitaiForm.value.name || '').trim().length > 0 &&
    String(civitaiForm.value.load_name || '').trim().length > 0
)

function htmlToText(input: string) {
  const s = String(input || '').trim()
  if (!s) return ''
  if (!/[<>]/.test(s)) return s
  try {
    const doc = new DOMParser().parseFromString(s, 'text/html')
    return String(doc?.body?.textContent || '').trim()
  } catch {
    return s.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim()
  }
}

function extractCivitaiVersionId(input: string): string | null {
  const s = String(input || '').trim()
  if (!s) return null
  if (/^\d+$/.test(s)) return s
  let m = s.match(/civitai:(\d+)@(\d+)/i)
  if (m?.[2]) return m[2]
  m = s.match(/familyId=(\d+)/i)
  if (m?.[1]) return m[1]
  m = s.match(/model-versions\/(\d+)/i)
  if (m?.[1]) return m[1]
  m = s.match(/@(\d+)\b/)
  if (m?.[1]) return m[1]
  return null
}

function normalizeCkTypeFromCivitai(v?: string) {
  const s = String(v || '').trim().toLowerCase()
  if (!s) return 'checkpoint'
  if (s === 'lora') return 'lora'
  if (s === 'checkpoint') return 'checkpoint'
  if (s === 'textual inversion') return 'textual inversion'
  if (s === 'controlnet') return 'controlnet'
  if (s === 'vae') return 'vae'
  return s
}

function pickCivitaiLocalPathFromFiles(files: any): string {
  const arr: CivitaiFile[] = Array.isArray(files) ? (files as any) : []
  const candidates = arr
    .map((f) => ({
      name: String((f as any)?.name || '').trim(),
      type: String((f as any)?.type || '').trim().toLowerCase(),
      sizeKB: Number((f as any)?.sizeKB || 0),
    }))
    .filter((x) => x.name.length > 0)

  if (!candidates.length) return ''

  function extScore(name: string) {
    const n = name.toLowerCase()
    if (n.endsWith('.safetensors')) return 60
    if (n.endsWith('.ckpt')) return 50
    if (n.endsWith('.pth')) return 40
    if (n.endsWith('.pt')) return 35
    if (n.endsWith('.bin')) return 30
    if (n.endsWith('.gguf')) return 20
    return 0
  }

  function typeScore(t: string) {
    // civitai 常见：Model / Training Data / Config / ...
    if (t === 'model') return 20
    return 0
  }

  // 评分：优先 Model 类型 + 常见权重后缀；同分取更大的文件（通常更像主权重）
  const best = candidates
    .map((x) => ({
      ...x,
      score: typeScore(x.type) + extScore(x.name),
    }))
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score
      if ((b.sizeKB || 0) !== (a.sizeKB || 0)) return (b.sizeKB || 0) - (a.sizeKB || 0)
      return a.name.localeCompare(b.name)
    })[0]

  return String(best?.name || '').trim()
}

const remoteLink = computed(() => {
  const p = String(remoteProvider.value || '').trim().toLowerCase()
  const rid = String(remoteRepoId.value || '').trim()
  if (!p || !rid) return ''
  if (p === 'huggingface') return `https://huggingface.co/${rid}`
  if (p === 'modelscope') return `https://modelscope.cn/models/${rid}`
  return ''
})

const remoteDisplayName = computed(() => {
  const info = remoteInfo.value
  const rid = String(remoteRepoId.value || '').trim()
  const p = String(remoteProvider.value || '').trim().toLowerCase()
  if (p === 'huggingface') {
    const n = String(info?.card_data?.model_name || info?.id || '').trim()
    return n || rid
  }
  if (p === 'modelscope') {
    const n = String(info?.Name || info?.ChineseName || info?.Path || '').trim()
    return n || rid
  }
  return rid
})

const remoteSummary = computed(() => {
  const info = remoteInfo.value
  const p = String(remoteProvider.value || '').trim().toLowerCase()
  if (!info) return ''
  if (p === 'huggingface') {
    const license = String(info?.card_data?.license_name || info?.card_data?.license || '').trim()
    const tags = Array.isArray(info?.tags) ? info.tags.slice(0, 12).join(', ') : ''
    const lm = String(info?.last_modified || '').trim()
    const dl = String(info?.downloads || '').trim()
    return [license ? `license: ${license}` : '', tags ? `tags: ${tags}` : '', lm ? `last_modified: ${lm}` : '', dl ? `downloads: ${dl}` : '']
      .filter(Boolean)
      .join('\n')
  }
  if (p === 'modelscope') {
    const license = String(info?.License || '').trim()
    const dl = String(info?.Downloads || '').trim()
    const desc = String(info?.Description || '').trim()
    return [license ? `license: ${license}` : '', dl ? `downloads: ${dl}` : '', desc ? `desc: ${desc.slice(0, 280)}` : ''].filter(Boolean).join('\n')
  }
  return ''
})

async function fetchImportModelInfo() {
  const raw = String(civitaiInput.value || '').trim()
  if (!raw) return
  civitaiLoading.value = true
  civitaiError.value = ''
  civitaiInfo.value = null
  remoteProvider.value = ''
  remoteRepoId.value = ''
  remoteInfo.value = null
  remoteThumbCandidates.value = []
  thumbInput.value = ''

  // Civitai 分支：包含 civitai.com / civitai URN / 纯数字 versionId
  const vid = extractCivitaiVersionId(raw)
  if (vid) {
    try {
      const url = `https://civitai.com/api/v1/model-versions/${encodeURIComponent(vid)}`
      const res = await fetch(url, { method: 'GET', headers: { Accept: 'application/json' } })
      if (!res.ok) throw new Error(t('models.import.civitaiFetchFailed', { status: res.status }))
      const json = (await res.json()) as CivitaiFamily
      civitaiInfo.value = json

      const firstImg = (json?.images || [])[0]?.url || ''
      if (firstImg) thumbInput.value = String(firstImg).trim()

      const modelName = String(json?.model?.name || '').trim()
      const displayName = modelName ? modelName.slice(0, 255) : `civitai-${vid}`
      const localPath = pickCivitaiLocalPathFromFiles((json as any)?.files)
      civitaiForm.value.name = displayName
      civitaiForm.value.load_name = localPath || displayName
      civitaiForm.value.asset_type = normalizeCkTypeFromCivitai(json?.model?.type)
      civitaiForm.value.family = normalizeFamily(String(json?.baseModel || '').trim())
      civitaiForm.value.trigger_words = Array.isArray(json?.trainedWords) ? json.trainedWords.join(',') : ''
      civitaiForm.value.description = htmlToText(String((json as any)?.description || ''))
    } catch (e: any) {
      console.error(e)
      civitaiInfo.value = null
      civitaiError.value =
        (e?.message ? String(e.message) : t('models.import.fetchFailed')) + t('models.import.networkHint')
    } finally {
      civitaiLoading.value = false
    }
    return
  }

  // HF / ModelScope 分支：交给后端探测并获取信息
  try {
    const res = await getRemoteModelInfo(raw)
    remoteProvider.value = String((res as any)?.provider || '')
    remoteRepoId.value = String((res as any)?.repo_id || '')
    remoteInfo.value = (res as any)?.info || null
    remoteThumbCandidates.value = Array.isArray((res as any)?.thumb_candidates) ? (res as any).thumb_candidates : []
    if (!thumbInput.value && remoteThumbCandidates.value.length) thumbInput.value = String(remoteThumbCandidates.value[0] || '').trim()

    // 填充默认表单（可编辑）
    const repoId = String(remoteRepoId.value || '').trim()
    const repoBaseName = repoId.includes('/') ? repoId.split('/').pop() || '' : repoId
    const displayName =
      String(remoteDisplayName.value || '').slice(0, 255) || repoBaseName || repoId.slice(0, 255)
    civitaiForm.value.name = displayName
    civitaiForm.value.load_name = String(repoBaseName || displayName).slice(0, 255)
    civitaiForm.value.asset_type = civitaiForm.value.asset_type || 'checkpoint'
    if (!civitaiForm.value.description) {
      civitaiForm.value.description = htmlToText(String(remoteSummary.value || ''))
    }
  } catch (e: any) {
    console.error(e)
    civitaiError.value = e?.message || t('models.import.fetchFailed')
    remoteInfo.value = null
  } finally {
    civitaiLoading.value = false
  }
}

function validateImportForm(): { ok: true; payload: ImportPayload } | { ok: false } {
  const info = civitaiInfo.value
  const isCivitai = Boolean(info)
  if (!isCivitai && !remoteInfo.value) {
    showTopSnack(t('models.import.fetchFirst'))
    return { ok: false }
  }

  const name = String(civitaiForm.value.name || '').trim()
  if (!name) {
    showTopSnack(t('models.form.nameRequired'))
    return { ok: false }
  }

  const load_name = String(civitaiForm.value.load_name || '').trim() || name
  const modality = String(civitaiForm.value.modality || '').trim() || 'image'
  const storage_mode = String(civitaiForm.value.storage_mode || '').trim() || 'local'
  const asset_type = String(civitaiForm.value.asset_type || '').trim() || 'checkpoint'
  const family = String(civitaiForm.value.family || '').trim() || undefined
  const trigger_words = String(civitaiForm.value.trigger_words || '').trim()
    ? String(civitaiForm.value.trigger_words || '').split(',').map((x) => x.trim()).filter(Boolean)
    : []
  const description = String(civitaiForm.value.description || '').trim() || undefined
  const thumb = String(thumbInput.value || '').trim() || undefined

  const provider = isCivitai ? 'civitai' : String(remoteProvider.value || '').trim()
  const repo_id = isCivitai ? String((info as any)?.id || '').trim() : String(remoteRepoId.value || '').trim()
  if (!provider || !repo_id) {
    showTopSnack(t('models.import.sourceRequired'))
    return { ok: false }
  }

  return {
    ok: true,
    payload: { name, modality, storage_mode, asset_type, family, trigger_words, description, thumb, load_name, provider, repo_id },
  }
}

type ImportPayload = {
  name: string
  modality: string
  storage_mode: string
  asset_type: string
  family?: string
  trigger_words: string[]
  description?: string
  thumb?: string
  load_name: string
  provider: string
  repo_id: string
}

async function ensureModelRecord(payload: ImportPayload): Promise<{ modelKey: string; wasExisting: boolean }> {
  let createdModelKey = String(civitaiCreatedModelKey.value || '').trim()
  if (createdModelKey) return { modelKey: createdModelKey, wasExisting: false }

  const { name, modality, storage_mode, asset_type, family, trigger_words, description, thumb, load_name, provider, repo_id } = payload
  const res = await createModelWithMeta({
    name,
    modality,
    storage_mode,
    load_name,
    source: { provider, repo_id },
    family,
    description,
    trigger_words,
    capabilities: { editable: false },
    thumb,
    asset_type,
  } satisfies CreateModelBody)

  createdModelKey = String(res.record?.model_key || '').trim()
  if (!createdModelKey) throw new Error('创建模型后未返回 model_key')
  civitaiCreatedModelKey.value = createdModelKey
  return { modelKey: createdModelKey, wasExisting: res.msg === 'exists' }
}

async function confirmSave() {
  if (civitaiSubmitting.value) return
  const validated = validateImportForm()
  if (!validated.ok) return

  civitaiSubmitting.value = true
  submitAction.value = 'save'
  try {
    const modelKey = await ensureModelRecord(validated.payload)
    showTopSnack(modelKey.wasExisting ? t('models.import.alreadyExists') : t('models.import.saveSuccess'))
    emit('done')
    emit('close')
  } catch (e: any) {
    console.error(e)
    showTopSnack(e?.message || t('models.import.importFailed'))
  } finally {
    civitaiSubmitting.value = false
    submitAction.value = ''
  }
}

async function confirmImport() {
  if (civitaiSubmitting.value) return
  const validated = validateImportForm()
  if (!validated.ok) return

  const { asset_type, provider, repo_id } = validated.payload
  civitaiSubmitting.value = true
  submitAction.value = 'import'
  try {
    const { modelKey: createdModelKey, wasExisting } = await ensureModelRecord(validated.payload)
    await downloadActionModel(createdModelKey, { action: 'start', source: { provider, repo_id }, asset_type })
    showTopSnack(wasExisting ? t('models.import.alreadyExistsDownload') : t('models.import.importSuccess'))

    emit('done')
    emit('close')
  } catch (e: any) {
    console.error(e)
    if (String(civitaiCreatedModelKey.value || '').trim()) showTopSnack((e?.message || t('models.import.downloadFailed')) + t('models.import.downloadRetryHint'))
    else showTopSnack(e?.message || t('models.import.importFailed'))
  } finally {
    civitaiSubmitting.value = false
    submitAction.value = ''
  }
}
</script>

