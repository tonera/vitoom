<template>
  <div class="rounded-2xl border border-dashed border-gray-300 bg-gray-50/80 p-4 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900/40">
    <div class="flex items-start justify-between gap-3">
      <div>
        <div class="text-sm font-semibold text-gray-950 [.dark_&]:text-white">{{ displayLabel }}</div>
        <p v-if="description" class="mt-1 text-xs leading-5 text-gray-500 [.dark_&]:text-gray-400">{{ description }}</p>
      </div>
      <button
        v-if="modelValue"
        type="button"
        class="shrink-0 rounded-full px-3 py-1 text-xs font-medium text-gray-500 transition hover:bg-gray-100 hover:text-gray-900 [.dark_&]:text-gray-400 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-white cursor-pointer"
        @click="clearFile"
      >
        {{ t('audio.uploadField.clear') }}
      </button>
    </div>

    <div class="mt-4 flex flex-col gap-3">
      <button
        type="button"
        class="flex min-h-24 w-full cursor-pointer flex-col items-center justify-center rounded-xl border border-gray-200 bg-white px-4 py-5 text-center transition hover:border-indigo-300 hover:bg-indigo-50/50 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950/60 [.dark_&]:hover:border-indigo-500/50 [.dark_&]:hover:bg-indigo-950/20"
        :disabled="uploading"
        @click="openPicker"
      >
        <svg class="h-7 w-7 text-gray-400 [.dark_&]:text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19V6l12-2v13M9 19c0 1.105-1.79 2-4 2s-4-.895-4-2 1.79-2 4-2 4 .895 4 2Zm12-2c0 1.105-1.79 2-4 2s-4-.895-4-2 1.79-2 4-2 4 .895 4 2Z" />
        </svg>
        <span class="mt-2 text-sm font-medium text-gray-800 [.dark_&]:text-gray-100">
          {{ uploading ? t('audio.uploadField.uploading') : modelValue ? t('audio.uploadField.reselect') : t('audio.uploadField.selectOrUpload') }}
        </span>
        <span class="mt-1 text-xs text-gray-500 [.dark_&]:text-gray-400">{{ displayHint }}</span>
      </button>

      <div v-if="modelValue" class="rounded-xl border border-gray-200 bg-white px-3 py-2 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950/60">
        <div class="truncate text-xs font-medium text-gray-800 [.dark_&]:text-gray-100" :title="fileName || modelValue">
          {{ fileName || modelValue }}
        </div>
        <audio class="mt-2 w-full" :src="modelValue" controls preload="metadata" />
      </div>
    </div>

    <input ref="inputRef" class="hidden" type="file" :accept="accept" @change="onFileChange">
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

const props = withDefaults(
  defineProps<{
    modelValue?: string
    label?: string
    description?: string
    hint?: string
    accept?: string
    fileName?: string
    uploading?: boolean
  }>(),
  {
    modelValue: '',
    accept: 'audio/*',
    fileName: '',
    uploading: false,
  }
)

const emit = defineEmits<{
  upload: [file: File]
  clear: []
}>()

const { t } = useI18n()

const displayLabel = computed(() => props.label ?? t('audio.uploadField.label'))
const displayHint = computed(() => props.hint ?? t('audio.uploadField.hint'))

const inputRef = ref<HTMLInputElement | null>(null)

function openPicker() {
  if (inputRef.value) inputRef.value.click()
}

function onFileChange(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (file) emit('upload', file)
}

function clearFile() {
  emit('clear')
}
</script>
