<template>
  <Teleport to="body">
    <div
      v-if="qr.open"
      class="fixed inset-0 z-9999 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
      @mousedown.self="close"
    >
      <div class="vt-card w-[420px] max-w-[92vw] p-5">
        <div class="flex items-start justify-between gap-4">
          <div>
            <div class="text-base font-semibold text-gray-900 [.dark_&]:text-gray-100">
              {{ t('components.upload.qrTitle') }}
            </div>
            <p class="mt-1 text-xs leading-relaxed text-gray-500 [.dark_&]:text-gray-400">
              {{ t('components.upload.qrDesc') }}
            </p>
          </div>
          <button
            type="button"
            class="shrink-0 w-8 h-8 rounded-full border border-gray-200 hover:bg-gray-100 text-gray-600 flex items-center justify-center text-lg cursor-pointer [.dark_&]:border-transparent [.dark_&]:hover:bg-gray-700 [.dark_&]:text-gray-200"
            @click="close"
            :aria-label="t('common.close')"
          >
            ✕
          </button>
        </div>

        <div class="mt-5 flex flex-col items-center">
          <div class="rounded-2xl border border-gray-200 bg-white p-3 shadow-sm [.dark_&]:border-gray-700">
            <canvas :ref="bindQrCanvas" width="220" height="220" />
          </div>
          <div v-if="qr.loading" class="mt-4 text-sm text-gray-500 [.dark_&]:text-gray-400">
            {{ t('components.upload.generatingQr') }}
          </div>
          <div v-else-if="qr.error" class="mt-4 text-sm text-red-600 [.dark_&]:text-red-300">
            {{ qr.error }}
          </div>
          <div v-else class="mt-4 text-sm text-gray-500 [.dark_&]:text-gray-400">
            {{ qr.polling ? t('components.upload.waitingUpload') : t('components.upload.qrStopped') }}
          </div>

          <div class="mt-4 flex w-full gap-2">
            <button
              type="button"
              class="flex-1 rounded-xl border border-gray-200 bg-white px-4 py-2 text-sm text-gray-800 hover:bg-gray-50 cursor-pointer [.dark_&]:border-gray-700 [.dark_&]:bg-gray-800 [.dark_&]:hover:bg-gray-700 [.dark_&]:text-gray-200"
              @click="copyQrLink"
            >
              {{ t('components.upload.copyLink') }}
            </button>
            <button
              type="button"
              class="flex-1 rounded-xl bg-indigo-600 px-4 py-2 text-sm text-white hover:bg-indigo-700 cursor-pointer"
              @click="refresh"
            >
              {{ t('components.upload.refreshQr') }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { useQrCodeUpload, type QrUploadResult } from '../composables/useQrCodeUpload'

const props = defineProps<{
  accept: string
}>()

const emit = defineEmits<{
  uploaded: [result: QrUploadResult]
}>()

const { t } = useI18n()

const { qr, qrCanvasRef, open, close, copyQrLink, refresh } = useQrCodeUpload({
  accept: () => props.accept,
  onUploaded: (result) => emit('uploaded', result),
})

const bindQrCanvas = (el: unknown) => {
  qrCanvasRef.value = el instanceof HTMLCanvasElement ? el : null
}

defineExpose({ open, close })
</script>
