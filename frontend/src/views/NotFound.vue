<template>
  <div class="h-full flex items-center justify-center">
    <div class="vt-card p-8 text-center">
      <h1 class="text-5xl font-bold text-white">{{ t('home.notFound.title') }}</h1>
      <p class="mt-3 text-base text-gray-300">{{ t('home.notFound.message') }}</p>
      <router-link
        to="/"
        class="mt-6 inline-flex items-center justify-center px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
      >
        {{ t('home.notFound.backHome') }}
      </router-link>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
</script>
