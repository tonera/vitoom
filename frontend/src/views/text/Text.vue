<template>
  <div>
    <div class="vt-card p-5">
      <h1 class="text-base font-semibold text-white mb-2">{{ t('home.underDevelopment.textTitle') }}</h1>
      <p class="text-gray-300">{{ t('home.underDevelopment.text') }}</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
</script>
