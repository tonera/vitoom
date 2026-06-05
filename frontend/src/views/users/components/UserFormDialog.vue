<template>
  <Teleport to="body">
    <div v-if="open" class="fixed inset-0 z-50 flex items-end justify-center bg-black/45 p-4 sm:items-center" @click.self="emitClose()">
      <div
        role="dialog"
        aria-modal="true"
        class="w-full max-w-lg rounded-2xl border border-gray-200 bg-white p-5 shadow-xl [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900"
        @click.stop
      >
        <h2 class="text-lg font-semibold text-gray-950 [.dark_&]:text-white">
          {{ mode === 'create' ? t('users.form.createTitle') : t('users.form.editTitle') }}
        </h2>
        <p class="mt-1 text-sm text-gray-500 [.dark_&]:text-gray-400">
          {{ mode === 'create' ? t('users.form.createDesc') : t('users.form.editDesc') }}
        </p>

        <label class="mt-4 block">
          <span class="text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('users.form.email') }}</span>
          <input
            v-model="form.email"
            type="email"
            autocomplete="off"
            class="mt-2 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-gray-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:focus:border-gray-500"
          >
        </label>

        <label class="mt-4 block">
          <span class="text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">
            {{ mode === 'create' ? t('users.form.password') : t('users.form.newPasswordOptional') }}
          </span>
          <input
            v-model="form.password"
            type="password"
            autocomplete="new-password"
            class="mt-2 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-gray-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:focus:border-gray-500"
          >
        </label>

        <label class="mt-4 block">
          <span class="text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('users.form.nickname') }}</span>
          <input
            v-model="form.nickname"
            type="text"
            maxlength="100"
            class="mt-2 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-gray-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:focus:border-gray-500"
          >
        </label>

        <fieldset class="mt-4">
          <legend class="text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('users.form.statusLegend') }}</legend>
          <div class="mt-2 flex  gap-2">
            <label class="flex cursor-pointer items-center gap-2 text-sm text-gray-800 [.dark_&]:text-gray-200">
              <input v-model="form.status" type="radio" name="user-status" value="active" class="shrink-0">
              <span>{{ t('users.statusActive') }}</span>
            </label>
            <label class="flex cursor-pointer items-center gap-2 text-sm text-gray-800 [.dark_&]:text-gray-200">
              <input v-model="form.status" type="radio" name="user-status" value="disabled" class="shrink-0">
              <span>{{ t('users.statusDisabled') }}</span>
            </label>
          </div>
        </fieldset>

        <label class="mt-4 flex cursor-pointer items-center gap-2 text-sm text-gray-800 [.dark_&]:text-gray-200">
          <input v-model="form.is_admin" type="checkbox" class="shrink-0">
          <span>{{ t('users.form.admin') }}</span>
        </label>

        <div class="mt-6 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
          <button
            type="button"
            class="rounded-xl border border-gray-200 px-4 py-2 text-sm font-medium text-gray-700 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-60 [.dark_&]:border-gray-700 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800 cursor-pointer"
            :disabled="saving"
            @click="emitClose()"
          >
            {{ t('common.cancel') }}
          </button>
          <button
            type="button"
            class="rounded-xl bg-gray-950 px-4 py-2 text-sm font-medium text-white transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-60 [.dark_&]:bg-white [.dark_&]:text-gray-950 [.dark_&]:hover:bg-gray-200 cursor-pointer"
            :disabled="saving"
            @click="submit"
          >
            {{ saving ? t('common.submitting') : t('common.save') }}
          </button>
        </div>
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { reactive, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { createUser, updateUser, type AdminUser, type AdminUserStatus } from '../../../api/users'
import { showTopSnack } from '../../../composables/useTopSnack'
import { handleApiError } from '../../../utils/api'

const { t } = useI18n()

const props = defineProps<{
  open: boolean
  mode: 'create' | 'edit'
  user: AdminUser | null
}>()

const emit = defineEmits<{
  (e: 'close'): void
  (e: 'saved'): void
}>()

const saving = ref(false)

const form = reactive({
  email: '',
  password: '',
  nickname: '',
  status: 'active' as AdminUserStatus,
  is_admin: false,
})

function normalizeUserStatus(status?: string | null): 'active' | 'disabled' {
  if (status === 'inactive' || status === 'disabled') return 'disabled'
  return 'active'
}

function resetForm() {
  form.email = props.user?.email || ''
  form.password = ''
  form.nickname = props.user?.nickname || ''
  form.status = normalizeUserStatus(props.user?.status)
  form.is_admin = Boolean(props.user?.is_admin)
}

watch(
  () => [props.open, props.mode, props.user?.id] as const,
  ([open]) => {
    if (open) {
      resetForm()
    }
  },
  { immediate: true }
)

function emitClose() {
  if (saving.value) return
  emit('close')
}

async function submit() {
  const email = form.email.trim()
  if (!email) {
    showTopSnack(t('users.form.fillEmail'))
    return
  }

  if (props.mode === 'create' && form.password.trim().length < 6) {
    showTopSnack(t('users.form.passwordMinLength'))
    return
  }

  if (props.mode === 'edit' && form.password.trim() && form.password.trim().length < 6) {
    showTopSnack(t('users.form.newPasswordMinLength'))
    return
  }

  saving.value = true
  try {
    if (props.mode === 'create') {
      await createUser({
        email,
        password: form.password,
        nickname: form.nickname.trim() || undefined,
        status: form.status,
        is_admin: form.is_admin,
      })
      showTopSnack(t('users.form.created'))
    } else if (props.user) {
      const body: {
        email: string
        nickname: string | null
        status: AdminUserStatus
        is_admin: boolean
        password?: string
      } = {
        email,
        nickname: form.nickname.trim() || null,
        status: form.status,
        is_admin: form.is_admin,
      }
      const password = form.password.trim()
      if (password) {
        body.password = password
      }
      await updateUser(props.user.id, body)
      showTopSnack(t('users.form.updated'))
    }
    emit('saved')
    emit('close')
  } catch (error) {
    const apiError = handleApiError(error)
    showTopSnack(apiError.message || t('users.form.saveFailed'))
  } finally {
    saving.value = false
  }
}
</script>
