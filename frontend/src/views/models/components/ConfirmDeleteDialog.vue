<template>
  <Teleport to="body">
    <div v-if="open" class="fixed inset-0 z-9999 bg-black/60 flex items-center justify-center p-4" @click.self="emitClose()">
      <div class="vt-card w-full max-w-[520px] p-5 rounded-2xl">
        <div class="text-lg font-semibold text-gray-900 [.dark_&]:text-white">{{ t('models.delete.title') }}</div>
        <div class="mt-2 break-all text-sm text-gray-600 [.dark_&]:text-gray-300">
          {{ t('models.delete.body') }} <span class="font-semibold text-gray-900 [.dark_&]:text-white">{{ model?.name }}</span>（{{ model?.id }}）
        </div>
        <div class="mt-2 text-xs text-gray-500 [.dark_&]:text-gray-400">
          {{ t('models.delete.hint') }}
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
            class="px-4 py-2 rounded-xl bg-rose-600 text-white hover:bg-rose-500 transition-colors cursor-pointer disabled:opacity-60 disabled:cursor-not-allowed"
            :disabled="saving || !model?.model_key"
            @click="confirm()"
          >
            {{ saving ? t('models.delete.deleting') : t('common.delete') }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { deleteModel, type ModelRecord } from '../../../api/models'
import { showTopSnack } from '../../../composables/useTopSnack'

const { t } = useI18n()

const props = defineProps<{
  open: boolean
  model: ModelRecord | null
}>()

const emit = defineEmits<{
  (e: 'close'): void
  (e: 'deleted'): void
}>()

const saving = ref(false)

function emitClose() {
  if (saving.value) return
  emit('close')
}

async function confirm() {
  const modelKey = String(props.model?.model_key || '').trim()
  if (!modelKey) return
  if (saving.value) return
  saving.value = true
  try {
    await deleteModel(modelKey)
    showTopSnack(t('models.delete.success'))
    emit('deleted')
    emit('close')
  } catch (e: any) {
    console.error(e)
    showTopSnack(e?.message || t('models.delete.failed'))
  } finally {
    saving.value = false
  }
}
</script>
