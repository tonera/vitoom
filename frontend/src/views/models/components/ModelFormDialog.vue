<template>
  <Teleport to="body">
    <div v-if="open" class="fixed inset-0 z-9999 bg-black/60 flex items-center justify-center p-4" @click.self="emitClose()">
      <div class="vt-card w-full max-w-[760px] max-h-[min(80vh,860px)] overflow-y-auto vt-scroll p-5 rounded-2xl">
        <div class="flex items-start justify-between gap-4">
          <div>
            <div class="text-lg font-semibold text-gray-900 [.dark_&]:text-white">
              {{ mode === 'create' ? t('models.form.createTitle') : t('models.form.editTitle') }}
            </div>
          </div>
          <button
            type="button"
            class="cursor-pointer rounded-lg p-2 text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-900 [.dark_&]:text-gray-400 [.dark_&]:hover:bg-gray-800/60 [.dark_&]:hover:text-white"
            :aria-label="t('common.close')"
            @click="emitClose()"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div class="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div class="space-y-1">
            <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.shortName') }}</div>
            <input
              v-model="form.name"
              type="text"
              class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
            />
          </div>

          <div class="space-y-1">
            <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.name') }}</div>
            <input
              v-model="form.load_name"
              type="text"
              class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
            />
          </div>

          <div class="space-y-1">
            <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.weightType') }}</div>
            <select
              v-model="form.asset_type"
              class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
            >
              <option value="lora">lora</option>
              <option value="checkpoint">checkpoint</option>
            </select>
          </div>

          <div class="space-y-1">
            <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.category') }}</div>
            <select
              v-model="form.modality"
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
              v-model="form.storage_mode"
              class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
            >
              <option value="local">{{ t('models.storageLocal') }}</option>
              <option value="cloud">{{ t('models.storageCloud') }}</option>
            </select>
          </div>

          <div class="space-y-1">
            <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.family') }}</div>
            <select
              v-model="form.family"
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
            <div class="flex gap-3">
              <div class="flex-1">
                <input
                  v-model="form.thumb"
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
                  @error="handleThumbError"
                />
              </div>
            </div>
          </div>

          <div v-if="mode === 'create'" class="space-y-1">
            <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">storage_mode=local</div>
            <label
              class="inline-flex cursor-pointer select-none items-center gap-2 text-sm text-gray-800 [.dark_&]:text-gray-200"
            >
              <input :checked="form.storage_mode === 'local'" type="checkbox" class="accent-indigo-600" @change="setStorageModeFromCheckbox" />
              {{ t('models.form.localModel') }}
            </label>
          </div>

          <div v-if="mode === 'create'" class="space-y-1">
            <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">capabilities.editable</div>
            <label
              class="inline-flex cursor-pointer select-none items-center gap-2 text-sm text-gray-800 [.dark_&]:text-gray-200"
            >
              <input v-model="form.editable" type="checkbox" class="accent-indigo-600" />
              {{ t('models.form.editableModel') }}
            </label>
          </div>

          <div class="space-y-1 md:col-span-2">
            <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.triggerWords') }}</div>
            <input
              v-model="form.trigger_words"
              type="text"
              class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
            />
          </div>

          <div class="space-y-1 md:col-span-2">
            <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.description') }}</div>
            <textarea
              v-model="form.description"
              rows="3"
              class="w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
            />
          </div>

          <div v-if="mode === 'create'" class="space-y-1 md:col-span-2">
            <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ t('models.form.runtimeConfig') }}</div>
            <textarea
              v-model="form.runtime_config_text"
              rows="5"
              placeholder="{ ... }"
              class="w-full rounded-xl border border-gray-200 bg-gray-50 font-mono text-xs px-3 py-2 text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
            />
          </div>

        </div>

        <div class="mt-5 flex items-center justify-end gap-2">
          <button
            type="button"
            class="cursor-pointer rounded-xl border border-gray-200 bg-white px-4 py-2 text-gray-700 transition-colors hover:bg-gray-50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/50 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800/80"
            :disabled="saving"
            @click="emitClose()"
          >
            {{ t('common.cancel') }}
          </button>
          <button
            type="button"
            class="px-4 py-2 rounded-xl bg-indigo-600 text-white hover:bg-indigo-500 transition-colors cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
            :disabled="saving"
            @click="submit()"
          >
            {{ saving ? t('common.submitting') : t('common.save') }}
          </button>
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
import { createModel, updateModel, type CreateModelBody, type ModelRecord, type UpdateModelBody } from '../../../api/models'
import { handleApiError } from '../../../utils/api'
import {
  formatModelThumbForForm,
  normalizeModelThumbForStore,
  resolveModelThumbUrl,
} from '../../../utils/modelThumb'

type Mode = 'create' | 'edit'

const props = defineProps<{
  open: boolean
  mode: Mode
  model?: ModelRecord | null
}>()

const emit = defineEmits<{
  (e: 'close'): void
  (e: 'saved'): void
}>()

const { t } = useI18n()

const { modalityOptions, familyOptions, ensureLoaded: ensureCatalogMetaLoaded, normalizeFamily } = useModelCatalogMeta()

const saving = ref(false)

const form = ref<{
  name: string
  modality: string
  storage_mode: string
  load_name: string
  family: string
  version: string
  description: string
  trigger_words: string
  editable: boolean
  runtime_config_text: string
  thumb: string
  asset_type: string
}>({
  name: '',
  modality: 'image',
  storage_mode: 'local',
  load_name: '',
  family: '',
  version: '',
  description: '',
  trigger_words: '',
  editable: false,
  runtime_config_text: '',
  thumb: '',
  asset_type: 'checkpoint',
})

watch(
  () => [props.open, props.mode, props.model?.id] as const,
  ([open]) => {
    if (!open) return
    void ensureCatalogMetaLoaded()
    if (props.mode === 'create') {
      form.value = {
        name: '',
        modality: 'image',
        storage_mode: 'local',
        load_name: '',
        family: '',
        version: '',
        description: '',
        trigger_words: '',
        editable: false,
        runtime_config_text: '',
        thumb: '',
        asset_type: 'checkpoint',
      }
      return
    }
    const m = props.model
    const cfg = (m as any)?.runtime_config ?? undefined
    form.value = {
      name: String(m?.name || ''),
      modality: String(m?.modality || ''),
      storage_mode: String(m?.storage_mode || ''),
      load_name: String(m?.load_name || ''),
      family: normalizeFamily(String(m?.family || '')),
      version: String(m?.version || ''),
      description: String(m?.description || ''),
      trigger_words: Array.isArray(m?.trigger_words) ? m.trigger_words.join(',') : '',
      editable: Boolean(m?.capabilities?.editable),
      runtime_config_text: safeStringify(cfg, 2),
      thumb: formatModelThumbForForm(String(m?.thumb || '')),
      asset_type: String(m?.asset_type || ''),
    }
  },
  { immediate: true }
)

function emitClose() {
  if (saving.value) return
  emit('close')
}

function setStorageModeFromCheckbox(event: Event) {
  form.value.storage_mode = (event.target as HTMLInputElement).checked ? 'local' : 'cloud'
}

function parseJsonText(text: string): any | undefined {
  const t = String(text || '').trim()
  if (!t) return undefined
  try {
    return JSON.parse(t)
  } catch {
    return undefined
  }
}

function safeStringify(v: any, spaces = 0) {
  try {
    if (v === undefined || v === null) return ''
    if (typeof v === 'string') return v
    return JSON.stringify(v, null, spaces)
  } catch {
    return ''
  }
}

const thumbPreviewUrl = computed(() => resolveModelThumbUrl(String(form.value.thumb || '')))

function handleThumbError(event: Event) {
  const img = event.target as HTMLImageElement
  img.style.display = 'none'
}

async function submit() {
  if (saving.value) return
  const name = String(form.value.name || '').trim()
  if (!name) return showTopSnack(t('models.form.nameRequired'))

  saving.value = true
  try {
    const thumbRaw = String(form.value.thumb || '').trim()
    const normalizedThumb = thumbRaw ? normalizeModelThumbForStore(thumbRaw) : undefined

    if (props.mode === 'create') {
      const body: CreateModelBody = {
        name,
        modality: String(form.value.modality || '').trim() || 'image',
        storage_mode: String(form.value.storage_mode || '').trim() || 'local',
        load_name: String(form.value.load_name || '').trim() || name,
        family: String(form.value.family || '').trim() || undefined,
        description: String(form.value.description || '').trim() || undefined,
        trigger_words: String(form.value.trigger_words || '').split(',').map((x) => x.trim()).filter(Boolean),
        capabilities: { editable: Boolean(form.value.editable) },
        runtime_config: parseJsonText(form.value.runtime_config_text),
        thumb: normalizedThumb,
        asset_type: String(form.value.asset_type || '').trim() || 'checkpoint',
      }
      await createModel(body)
      showTopSnack(t('models.form.createSuccess'))
    } else {
      const modelKey = String(props.model?.model_key || '').trim()
      if (!modelKey) return showTopSnack(t('models.form.missingModelKey'))
      const orig = props.model
      const body: UpdateModelBody = { name }
      const modality = String(form.value.modality || '').trim()
      const storageMode = String(form.value.storage_mode || '').trim()
      const family = String(form.value.family || '').trim()
      const loadName = String(form.value.load_name || '').trim()
      const description = String(form.value.description || '').trim()
      const assetType = String(form.value.asset_type || '').trim()
      const triggerWords = String(form.value.trigger_words || '')
        .split(',')
        .map((x) => x.trim())
        .filter(Boolean)
      const editable = Boolean(form.value.editable)

      if (modality && modality !== String(orig?.modality || '').trim()) body.modality = modality
      if (storageMode && storageMode !== String(orig?.storage_mode || '').trim()) body.storage_mode = storageMode
      if (family !== String(orig?.family || '').trim()) body.family = family || undefined
      if (loadName && loadName !== String(orig?.load_name || '').trim()) body.load_name = loadName
      if (description !== String(orig?.description || '').trim()) body.description = description || undefined
      if (assetType && assetType !== String(orig?.asset_type || '').trim()) body.asset_type = assetType
      if (thumbRaw) body.thumb = normalizedThumb
      if (
        JSON.stringify(triggerWords) !==
        JSON.stringify(Array.isArray(orig?.trigger_words) ? orig.trigger_words : [])
      ) {
        body.trigger_words = triggerWords
      }
      if (editable !== Boolean(orig?.capabilities?.editable)) {
        body.capabilities = { editable }
      }

      await updateModel(modelKey, body)
      showTopSnack(t('models.form.updateSuccess'))
    }

    emit('saved')
    emit('close')
  } catch (e: any) {
    console.error(e)
    const err = handleApiError(e)
    showTopSnack(err.message || t('models.form.saveFailed'))
  } finally {
    saving.value = false
  }
}
</script>

