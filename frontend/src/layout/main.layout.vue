<template>
  <div class="flex h-screen flex-col overflow-hidden bg-white text-gray-950 vt-text-smooth [.dark_&]:bg-gray-900 [.dark_&]:text-gray-100">
    <header class="z-30 shrink-0 border-b border-gray-200 bg-white/95 shadow-sm backdrop-blur [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900/95">
      <div class="mx-auto flex h-16 max-w-[1600px] items-center gap-4 px-4 sm:px-6 lg:px-8">
        <router-link to="/" class="flex shrink-0 items-center gap-3" :aria-label="t('common.homeAriaLabel')">
          <span class="flex h-9 w-9 items-center justify-center rounded-xl bg-gray-950 text-base font-semibold text-white [.dark_&]:bg-white [.dark_&]:text-gray-950">
            V
          </span>
          <span class="text-lg font-semibold tracking-tight text-gray-950 [.dark_&]:text-white">Vitoom</span>
        </router-link>

        <nav class="hidden min-w-0 flex-1 items-center gap-1 overflow-x-auto lg:flex" :aria-label="t('common.mainNavAriaLabel')">
          <router-link
            v-for="item in navItems"
            :key="item.to"
            :to="item.to"
            :class="navLinkClass(isNavItemActive(item))"
          >
            {{ t(item.labelKey) }}
          </router-link>
        </nav>

        <nav class="-mx-1 flex min-w-0 flex-1 items-center gap-1 overflow-x-auto px-1 lg:hidden" :aria-label="t('common.mainNavAriaLabel')">
          <router-link
            v-for="item in navItems"
            :key="`mobile:${item.to}`"
            :to="item.to"
            :class="mobileNavLinkClass(isNavItemActive(item))"
          >
            {{ t(item.labelKey) }}
          </router-link>
        </nav>

        <LanguageSelect />

        <button
          type="button"
          class="flex h-9 w-9 shrink-0 items-center justify-center rounded-full text-gray-600 transition-colors hover:bg-gray-100 hover:text-gray-950 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-white cursor-pointer"
          :aria-label="isDark ? t('common.switchToLightMode') : t('common.switchToDarkMode')"
          :title="isDark ? t('common.switchToLightMode') : t('common.switchToDarkMode')"
          @click="toggleTheme"
        >
          <svg v-if="isDark" class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364-.707.707M6.343 17.657l-.707.707m12.728 0-.707-.707M6.343 6.343l-.707-.707M12 8a4 4 0 100 8 4 4 0 000-8z"
            />
          </svg>
          <svg v-else class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
            />
          </svg>
        </button>

        <div class="relative shrink-0" ref="userMenuRef">
          <div v-if="!isAuthenticated" class="flex items-center gap-1 sm:gap-2">
            <router-link
              to="/login"
              class="rounded-full px-3 py-2 text-sm font-medium text-gray-700 transition-colors hover:bg-gray-100 hover:text-gray-950 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-white sm:px-4"
            >
              {{ t('common.login') }}
            </router-link>
            <button
              type="button"
              class="rounded-full bg-gray-950 px-3 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-gray-800 [.dark_&]:bg-white [.dark_&]:text-gray-950 [.dark_&]:hover:bg-gray-200 sm:px-4"
              @click="showRegisterPlaceholder"
            >
              {{ t('common.register') }}
            </button>
          </div>

          <button
            v-else
            type="button"
            class="flex items-center gap-2 rounded-full border border-gray-200 bg-white py-1.5 pl-1.5 pr-3 text-sm font-medium text-gray-700 shadow-sm transition-colors hover:border-gray-300 hover:bg-gray-50 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900 [.dark_&]:text-gray-200 [.dark_&]:hover:border-gray-600 [.dark_&]:hover:bg-gray-800 cursor-pointer"
            @click="showUserMenu = !showUserMenu"
          >
            <span class="flex h-8 w-8 items-center justify-center rounded-full bg-gray-950 text-sm font-semibold text-white [.dark_&]:bg-white [.dark_&]:text-gray-950">
              {{ userInitial }}
            </span>
            <span class="hidden max-w-28 truncate sm:block">{{ displayName }}</span>
            <svg class="h-4 w-4 text-gray-500 [.dark_&]:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          <div
            v-if="showUserMenu"
            class="absolute right-0 mt-2 w-60 overflow-hidden rounded-2xl border border-gray-200 bg-white py-2 shadow-xl [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900"
          >
            <div class="border-b border-gray-100 px-4 pb-3 pt-2 [.dark_&]:border-gray-800">
              <p class="truncate text-sm font-semibold text-gray-950 [.dark_&]:text-white">{{ userInfo?.nickname || t('common.user') }}</p>
              <p class="mt-0.5 truncate text-xs text-gray-500 [.dark_&]:text-gray-400">{{ userInfo?.email || t('common.loggedIn') }}</p>
            </div>
            <button type="button" :class="menuItemClass" @click="goSettings">{{ t('common.settings') }}</button>
            <button v-if="isAdmin" type="button" :class="menuItemClass" @click="goUsers">{{ t('common.userManagement') }}</button>
            <button v-if="isAdmin" type="button" :class="menuItemClass" @click="goInferenceAdmin">{{ t('common.inferenceManagement') }}</button>
            <button v-if="isAdmin" type="button" :class="menuItemClass" @click="goModels">{{ t('common.models') }}</button>
            <button type="button" :class="menuItemClass" @click="goApiKeys">{{ t('common.apiKeys') }}</button>
            <button type="button" :class="[menuItemClass, 'text-red-600 hover:text-red-700 [.dark_&]:text-red-400 [.dark_&]:hover:text-red-300']" @click="handleLogout">
              {{ t('common.logout') }}
            </button>
          </div>
        </div>
      </div>
    </header>

    <main class="min-h-0 flex-1 overflow-hidden">
      <div class="h-full overflow-y-auto vt-scroll">
        <div class="mx-auto flex h-full min-h-full  flex-col px-3 py-3 sm:px-5 sm:py-5">
          <div class="min-h-0 flex-1 overflow-hidden">
            <router-view />
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import LanguageSelect from '../components/LanguageSelect.vue'
import { showTopSnack } from '../composables/useTopSnack'
import { post } from '../utils/api'
import { clearAuthTokens, hasValidAccessToken } from '../utils/auth'
import { clearCurrentUserCache, fetchCurrentUser, type CurrentUser } from '../utils/currentUser'
import { navItems, type TopNavItem } from './sidebarNav'

const router = useRouter()
const route = useRoute()
const { t } = useI18n()
const userMenuRef = ref<HTMLElement | null>(null)
const showUserMenu = ref(false)
const isAuthenticated = ref(hasValidAccessToken())
const isDark = ref(document.documentElement.classList.contains('dark'))
const userInfo = ref<CurrentUser | null>(null)

const isAdmin = computed(() => Boolean(userInfo.value?.is_admin))

const displayName = computed(() => userInfo.value?.nickname || userInfo.value?.email || t('common.user'))

const userInitial = computed(() => {
  const source = displayName.value || 'U'
  return source.charAt(0).toUpperCase()
})

const THEME_STORAGE_KEY = 'vitoom.theme'

const menuItemClass =
  'block w-full px-4 py-2.5 text-left text-sm text-gray-700 transition-colors hover:bg-gray-50 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800 cursor-pointer'

const navLinkClass = (active: boolean) => [
  'whitespace-nowrap rounded-full px-4 py-2 text-sm font-medium transition-colors',
  active
    ? 'bg-gray-950 text-white shadow-sm [.dark_&]:bg-white [.dark_&]:text-gray-950'
    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-950 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-white',
]

const mobileNavLinkClass = (active: boolean) => [
  'whitespace-nowrap rounded-full px-3 py-1.5 text-sm font-medium transition-colors',
  active
    ? 'bg-gray-950 text-white shadow-sm [.dark_&]:bg-white [.dark_&]:text-gray-950'
    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-950 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-white',
]

const isNavItemActive = (item: TopNavItem) => {
  return item.activeRouteNames.includes(route.name as string)
}

const handleClickOutside = (event: MouseEvent) => {
  if (userMenuRef.value && !userMenuRef.value.contains(event.target as Node)) {
    showUserMenu.value = false
  }
}

const fetchUserInfo = async () => {
  if (!isAuthenticated.value) return
  try {
    const info = await fetchCurrentUser(true)
    userInfo.value = info
  } catch (error) {
    console.error('Failed to fetch user info:', error)
  }
}

const goSettings = () => {
  showUserMenu.value = false
  router.push({ name: 'Settings' })
}

const goModels = () => {
  showUserMenu.value = false
  router.push({ name: 'Models' })
}

const goUsers = () => {
  showUserMenu.value = false
  router.push({ name: 'Users' })
}

const goInferenceAdmin = () => {
  showUserMenu.value = false
  router.push({ name: 'InferenceAdmin' })
}

const toggleTheme = () => {
  isDark.value = !isDark.value
  document.documentElement.classList.toggle('dark', isDark.value)
  window.localStorage.setItem(THEME_STORAGE_KEY, isDark.value ? 'dark' : 'light')
}

const showRegisterPlaceholder = () => {
  showTopSnack(t('common.registerComingSoon'))
}

const goApiKeys = () => {
  showUserMenu.value = false
  router.push({ name: 'ApiKeys' })
}

const handleLogout = async () => {
  try {
    await post('/auth/logout')
  } catch (error) {
    console.error('Logout error:', error)
  } finally {
    clearAuthTokens()
    clearCurrentUserCache()
    isAuthenticated.value = false
    showUserMenu.value = false
    router.push({ name: 'Login' })
  }
}

watch(showUserMenu, (open) => {
  if (open && isAuthenticated.value) {
    fetchUserInfo()
  }
})

onMounted(() => {
  fetchUserInfo()
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>

