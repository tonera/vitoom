<template>
  <div>
    <h1 class="text-2xl font-bold mb-4">{{ t('home.underDevelopment.videoDigitalHumanTitle') }}</h1>
    <p class="text-gray-600">{{ t('home.underDevelopment.videoDigitalHuman') }}</p>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
</script>
