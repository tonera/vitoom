<template>
  <Teleport to="body">
    <div v-if="open" class="fixed inset-0 z-50 flex items-end justify-center bg-black/45 p-4 sm:items-center" @click.self="emitClose()">
      <div class="w-full max-w-[520px] rounded-2xl border border-gray-200 bg-white p-5 shadow-xl [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900" @click.stop>
        <div class="text-lg font-semibold text-gray-950 [.dark_&]:text-white">{{ t('users.confirmDisable.title') }}</div>
        <div class="mt-2 break-all text-sm text-gray-600 [.dark_&]:text-gray-300">
          {{ t('users.confirmDisable.body') }}
          <span class="font-semibold text-gray-950 [.dark_&]:text-white">{{ user?.nickname || user?.email }}</span>
          （{{ user?.email }}）
        </div>
        <div class="mt-2 text-xs text-gray-500 [.dark_&]:text-gray-400">
          {{ t('users.confirmDisable.hint') }}
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
            class="cursor-pointer rounded-xl bg-rose-600 px-4 py-2 text-white transition-colors hover:bg-rose-500 disabled:cursor-not-allowed disabled:opacity-60"
            :disabled="saving || !user?.id"
            @click="confirm"
          >
            {{ saving ? t('users.confirmDisable.disabling') : t('users.disable') }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { disableUser, type AdminUser } from '../../../api/users'
import { showTopSnack } from '../../../composables/useTopSnack'
import { handleApiError } from '../../../utils/api'

const { t } = useI18n()

const props = defineProps<{
  open: boolean
  user: AdminUser | null
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
  const userId = props.user?.id
  if (!userId || saving.value) return

  saving.value = true
  try {
    await disableUser(userId)
    showTopSnack(t('users.confirmDisable.success'))
    emit('deleted')
    emit('close')
  } catch (error) {
    const apiError = handleApiError(error)
    showTopSnack(apiError.message || t('users.confirmDisable.failed'))
  } finally {
    saving.value = false
  }
}
</script>
