<template>
  <div class="relative shrink-0" ref="menuRef">
    <button
      type="button"
      class="flex h-9 items-center gap-1 rounded-full px-2.5 text-sm font-medium text-gray-600 transition-colors hover:bg-gray-100 hover:text-gray-950 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-white cursor-pointer sm:px-3"
      :aria-label="t('common.selectLanguage')"
      :title="currentLabel"
      @click="showMenu = !showMenu"
    >
      <svg class="h-4 w-4 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path
          stroke-linecap="round"
          stroke-linejoin="round"
          stroke-width="2"
          d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129"
        />
      </svg>
      <span class="hidden sm:inline">{{ currentLabel }}</span>
      <svg class="h-3.5 w-3.5 shrink-0 text-gray-500 [.dark_&]:text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
      </svg>
    </button>

    <div
      v-if="showMenu"
      class="absolute right-0 z-40 mt-2 min-w-[8.5rem] overflow-hidden rounded-xl border border-gray-200 bg-white py-1 shadow-xl [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900"
    >
      <button
        v-for="option in languageOptions"
        :key="option.value"
        type="button"
        :class="[
          'block w-full px-4 py-2 text-left text-sm transition-colors cursor-pointer',
          option.value === currentLocale
            ? 'bg-gray-100 font-medium text-gray-950 [.dark_&]:bg-gray-800 [.dark_&]:text-white'
            : 'text-gray-700 hover:bg-gray-50 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800',
        ]"
        @click="selectLocale(option.value)"
      >
        {{ option.label }}
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { getCurrentLocale, setLocale, type SupportedLocale } from '../i18n'

const { t } = useI18n()

const languageOptions: { value: SupportedLocale; label: string }[] = [
  { value: 'zh-CN', label: '中文' },
  { value: 'en-US', label: 'English' },
  { value: 'ja-JP', label: '日本語' },
]

const menuRef = ref<HTMLElement | null>(null)
const showMenu = ref(false)

const currentLocale = computed(() => getCurrentLocale())

const currentLabel = computed(
  () => languageOptions.find((option) => option.value === currentLocale.value)?.label ?? '中文',
)

const selectLocale = async (locale: SupportedLocale) => {
  showMenu.value = false
  await setLocale(locale)
}

const handleClickOutside = (event: MouseEvent) => {
  if (menuRef.value && !menuRef.value.contains(event.target as Node)) {
    showMenu.value = false
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
})
</script>
