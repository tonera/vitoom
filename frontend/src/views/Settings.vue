<template>
  <div class="flex h-full min-h-0 flex-col overflow-hidden bg-gray-50 text-gray-950 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100">
    <div class="shrink-0 border-b border-gray-200 bg-white px-5 py-4 [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
      <h1 class="text-xl font-semibold">{{ t('settings.title') }}</h1>
      <p class="mt-1 text-sm text-gray-500 [.dark_&]:text-gray-400">
        {{ t('settings.subtitle') }}
      </p>
    </div>

    <div class="min-h-0 flex-1 overflow-y-auto p-5">
      <div class="mx-auto flex w-full max-w-2xl flex-col gap-5">
        <section class="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
          <div class="border-b border-gray-100 px-5 py-4 [.dark_&]:border-gray-800">
            <h2 class="text-base font-semibold">{{ t('settings.profile.title') }}</h2>
            <p class="mt-1 text-sm text-gray-500 [.dark_&]:text-gray-400">{{ t('settings.profile.desc') }}</p>
          </div>

          <div v-if="loadingProfile" class="px-5 py-10 text-center text-sm text-gray-500 [.dark_&]:text-gray-400">
            {{ t('common.loading') }}
          </div>
          <form v-else class="space-y-4 px-5 py-5" @submit.prevent="saveNickname">
            <label class="block">
              <span class="text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('settings.profile.email') }}</span>
              <input
                :value="profile?.email || ''"
                type="email"
                disabled
                class="mt-2 w-full cursor-not-allowed rounded-xl border border-gray-200 bg-gray-50 px-3 py-2 text-sm text-gray-500 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950/60 [.dark_&]:text-gray-400"
              >
            </label>

            <label class="block">
              <span class="text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('settings.profile.nickname') }}</span>
              <input
                v-model="nicknameForm.nickname"
                type="text"
                maxlength="100"
                :placeholder="t('settings.profile.nicknamePlaceholder')"
                class="mt-2 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-gray-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:focus:border-gray-500"
              >
            </label>

            <div class="flex justify-end">
              <button
                type="submit"
                class="rounded-xl bg-gray-950 px-4 py-2 text-sm font-medium text-white transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-60 [.dark_&]:bg-white [.dark_&]:text-gray-950 [.dark_&]:hover:bg-gray-200 cursor-pointer"
                :disabled="savingNickname"
              >
                {{ savingNickname ? t('settings.profile.saving') : t('settings.profile.saveNickname') }}
              </button>
            </div>
          </form>
        </section>

        <section class="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
          <div class="border-b border-gray-100 px-5 py-4 [.dark_&]:border-gray-800">
            <h2 class="text-base font-semibold">{{ t('settings.password.title') }}</h2>
            <p class="mt-1 text-sm text-gray-500 [.dark_&]:text-gray-400">{{ t('settings.password.desc') }}</p>
          </div>

          <form class="space-y-4 px-5 py-5" @submit.prevent="savePassword">
            <label class="block">
              <span class="text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('settings.password.newPassword') }}</span>
              <input
                v-model="passwordForm.password"
                type="password"
                autocomplete="new-password"
                class="mt-2 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-gray-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:focus:border-gray-500"
              >
            </label>

            <label class="block">
              <span class="text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('settings.password.confirmPassword') }}</span>
              <input
                v-model="passwordForm.confirmPassword"
                type="password"
                autocomplete="new-password"
                class="mt-2 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-gray-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:focus:border-gray-500"
              >
            </label>

            <div class="flex justify-end">
              <button
                type="submit"
                class="rounded-xl bg-gray-950 px-4 py-2 text-sm font-medium text-white transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-60 [.dark_&]:bg-white [.dark_&]:text-gray-950 [.dark_&]:hover:bg-gray-200 cursor-pointer"
                :disabled="savingPassword"
              >
                {{ savingPassword ? t('settings.password.saving') : t('settings.password.submit') }}
              </button>
            </div>
          </form>
        </section>

        <section class="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
          <div class="border-b border-gray-100 px-5 py-4 [.dark_&]:border-gray-800">
            <h2 class="text-base font-semibold">{{ t('settings.about.title') }}</h2>
          </div>
          <div class="flex items-center justify-between px-5 py-5">
            <span class="text-sm text-gray-500 [.dark_&]:text-gray-400">{{ t('settings.about.version') }}</span>
            <span class="text-sm font-medium text-gray-900 [.dark_&]:text-gray-100">{{ appVersion || '—' }}</span>
          </div>
        </section>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { fetchHealth, getBuildTimeVersion } from '../api/system'
import { updateProfile } from '../api/profile'
import { showTopSnack } from '../composables/useTopSnack'
import { handleApiError } from '../utils/api'
import { fetchCurrentUser, type CurrentUser } from '../utils/currentUser'

const { t } = useI18n()

const loadingProfile = ref(true)
const savingNickname = ref(false)
const savingPassword = ref(false)
const appVersion = ref('')
const profile = ref<CurrentUser | null>(null)

const nicknameForm = reactive({
  nickname: '',
})

const passwordForm = reactive({
  password: '',
  confirmPassword: '',
})

async function loadAppVersion() {
  try {
    const health = await fetchHealth()
    appVersion.value = health.version || getBuildTimeVersion()
  } catch {
    appVersion.value = getBuildTimeVersion()
  }
}

async function loadProfile() {
  loadingProfile.value = true
  try {
    profile.value = await fetchCurrentUser(true)
    nicknameForm.nickname = profile.value?.nickname || ''
  } catch (error) {
    const apiError = handleApiError(error)
    showTopSnack(apiError.message || t('settings.loadProfileFailed'))
  } finally {
    loadingProfile.value = false
  }
}

async function saveNickname() {
  if (savingNickname.value) return
  savingNickname.value = true
  try {
    profile.value = await updateProfile({
      nickname: nicknameForm.nickname.trim() || null,
    })
    nicknameForm.nickname = profile.value?.nickname || ''
    await fetchCurrentUser(true)
    showTopSnack(t('settings.profile.updated'))
  } catch (error) {
    const apiError = handleApiError(error)
    showTopSnack(apiError.message || t('settings.profile.saveFailed'))
  } finally {
    savingNickname.value = false
  }
}

async function savePassword() {
  const password = passwordForm.password.trim()
  const confirmPassword = passwordForm.confirmPassword.trim()

  if (!password) {
    showTopSnack(t('settings.password.fillNewPassword'))
    return
  }
  if (password.length < 6) {
    showTopSnack(t('settings.password.minLength'))
    return
  }
  if (password !== confirmPassword) {
    showTopSnack(t('settings.password.mismatch'))
    return
  }

  if (savingPassword.value) return
  savingPassword.value = true
  try {
    await updateProfile({ new_password: password })
    passwordForm.password = ''
    passwordForm.confirmPassword = ''
    showTopSnack(t('settings.password.updated'))
  } catch (error) {
    const apiError = handleApiError(error)
    showTopSnack(apiError.message || t('settings.password.saveFailed'))
  } finally {
    savingPassword.value = false
  }
}

onMounted(() => {
  void loadAppVersion()
  loadProfile()
})
</script>
