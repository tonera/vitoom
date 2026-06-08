<template>
  <div class="min-h-screen flex items-center justify-center bg-gray-50 px-4 py-10 dark:bg-gray-950">
    <div class="max-w-md w-full space-y-8 rounded-3xl border border-gray-200 bg-white p-8 shadow-xl shadow-gray-200/70 dark:border-gray-800 dark:bg-gray-900 dark:shadow-black/30">
      <div class="text-center">
        <div class="flex items-center justify-center mb-4">
          <div class="w-12 h-12 bg-gray-950 rounded-2xl flex items-center justify-center mr-3 dark:bg-white">
            <span class="text-white font-bold text-2xl dark:text-gray-950">V</span>
          </div>
          <h1 class="text-2xl font-bold text-gray-950 dark:text-white">Vitoom</h1>
        </div>
        <h2 class="text-2xl font-bold text-gray-950 dark:text-gray-100">
          {{ t('auth.title') }}
        </h2>
        <p class="mt-2 text-sm text-gray-500 dark:text-gray-400">
          {{ t('auth.subtitle') }}
        </p>
      </div>
      <form class="mt-8 space-y-6" @submit.prevent="handleLogin">
        <div class="space-y-4">
          <div>
            <label for="email" class="mb-1.5 block text-sm font-medium text-gray-700 dark:text-gray-300">{{ t('auth.email') }}</label>
            <input
              id="email"
              v-model="form.email"
              name="email"
              type="email"
              required
              class="relative block w-full rounded-xl border border-gray-300 bg-white px-3 py-2.5 text-gray-950 placeholder-gray-400 shadow-sm focus:border-gray-950 focus:outline-none focus:ring-2 focus:ring-gray-950/10 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-100 dark:placeholder:text-gray-500 dark:focus:border-gray-300 dark:focus:ring-white/10 sm:text-sm"
              :placeholder="t('auth.emailPlaceholder')"
              autocomplete="email"
            />
          </div>
          <div>
            <label for="password" class="mb-1.5 block text-sm font-medium text-gray-700 dark:text-gray-300">{{ t('auth.password') }}</label>
            <input
              id="password"
              v-model="form.password"
              name="password"
              type="password"
              required
              class="relative block w-full rounded-xl border border-gray-300 bg-white px-3 py-2.5 text-gray-950 placeholder-gray-400 shadow-sm focus:border-gray-950 focus:outline-none focus:ring-2 focus:ring-gray-950/10 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-100 dark:placeholder:text-gray-500 dark:focus:border-gray-300 dark:focus:ring-white/10 sm:text-sm"
              :placeholder="t('auth.passwordPlaceholder')"
              autocomplete="current-password"
            />
          </div>
        </div>

        <div v-if="error" class="text-red-600 text-sm text-center">
          {{ error }}
        </div>

        <div>
          <button
            type="submit"
            :disabled="loading"
            class="group relative w-full flex justify-center rounded-xl border border-transparent bg-gray-950 px-4 py-2.5 text-sm font-medium text-white shadow-sm transition-colors hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-950/20 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-white dark:text-gray-950 dark:hover:bg-gray-200 dark:focus:ring-white/20 dark:focus:ring-offset-gray-900 cursor-pointer"
          >
            <span v-if="loading">{{ t('auth.loggingIn') }}</span>
            <span v-else>{{ t('common.login') }}</span>
          </button>
        </div>
        <p class="text-center text-sm text-gray-500 dark:text-gray-400">
          {{ t('auth.noAccount') }}
          <button type="button" class="font-medium text-gray-950 hover:underline dark:text-white cursor-pointer" @click="showRegisterPlaceholder">
            {{ t('common.register') }}
          </button>
        </p>
      </form>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter, useRoute } from 'vue-router'
import { showTopSnack } from '../composables/useTopSnack'
import { post, handleApiError } from '../utils/api'
import { setAccessToken, setRefreshToken } from '../utils/auth'

const { t } = useI18n()
const router = useRouter()
const route = useRoute()

const form = reactive({
  email: '',
  password: '',
})

const loading = ref(false)
const error = ref('')

const showRegisterPlaceholder = () => {
  showTopSnack(t('common.registerComingSoon'))
}

const handleLogin = async () => {
  loading.value = true
  error.value = ''

  try {
    const response = await post<{ access_token: string; refresh_token: string; token_type: string }>('/auth/login', {
      email: form.email,
      password: form.password,
    })

    setAccessToken(response.access_token)
    if (response.refresh_token) {
      setRefreshToken(response.refresh_token)
    }

    const redirect = (route.query.redirect as string) || '/'
    router.push(redirect)
  } catch (err: any) {
    const apiError = handleApiError(err)
    error.value = apiError.message || t('auth.loginFailed')
  } finally {
    loading.value = false
  }
}
</script>
