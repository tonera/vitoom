<template>
  <div>
        <div class="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          <router-link
            to="/generate/image"
            class="relative group bg-gray-800 border border-gray-700 p-6 rounded-lg shadow-lg hover:shadow-xl hover:border-indigo-500 transition-all"
          >
            <div class="flex items-center">
              <div class="flex-shrink-0 bg-indigo-500 rounded-md p-3">
                <svg class="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <div class="ml-5 w-0 flex-1">
                <dl>
                  <dt class="text-sm font-medium text-gray-400 truncate">{{ t('home.cards.imageGenerate.title') }}</dt>
                  <dd class="text-lg font-medium text-white">{{ t('home.cards.imageGenerate.desc') }}</dd>
                </dl>
              </div>
            </div>
          </router-link>

          <router-link
            to="/generate/video"
            class="relative group bg-gray-800 border border-gray-700 p-6 rounded-lg shadow-lg hover:shadow-xl hover:border-indigo-500 transition-all"
          >
            <div class="flex items-center">
              <div class="flex-shrink-0 bg-green-500 rounded-md p-3">
                <svg class="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
              </div>
              <div class="ml-5 w-0 flex-1">
                <dl>
                  <dt class="text-sm font-medium text-gray-400 truncate">{{ t('home.cards.videoGenerate.title') }}</dt>
                  <dd class="text-lg font-medium text-white">{{ t('home.cards.videoGenerate.desc') }}</dd>
                </dl>
              </div>
            </div>
          </router-link>

          <router-link
            to="/agents"
            class="relative group bg-gray-800 border border-gray-700 p-6 rounded-lg shadow-lg hover:shadow-xl hover:border-indigo-500 transition-all"
          >
            <div class="flex items-center">
              <div class="flex-shrink-0 bg-purple-500 rounded-md p-3">
                <svg class="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div class="ml-5 w-0 flex-1">
                <dl>
                  <dt class="text-sm font-medium text-gray-400 truncate">{{ t('home.cards.agent.title') }}</dt>
                  <dd class="text-lg font-medium text-white">{{ t('home.cards.agent.desc') }}</dd>
                </dl>
              </div>
            </div>
          </router-link>

          <router-link
            to="/models"
            class="relative group bg-gray-800 border border-gray-700 p-6 rounded-lg shadow-lg hover:shadow-xl hover:border-indigo-500 transition-all"
          >
            <div class="flex items-center">
              <div class="flex-shrink-0 bg-blue-500 rounded-md p-3">
                <svg class="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                </svg>
              </div>
              <div class="ml-5 w-0 flex-1">
                <dl>
                  <dt class="text-sm font-medium text-gray-400 truncate">{{ t('home.cards.models.title') }}</dt>
                  <dd class="text-lg font-medium text-white">{{ t('home.cards.models.desc') }}</dd>
                </dl>
              </div>
            </div>
          </router-link>

          <router-link
            to="/files"
            class="relative group bg-gray-800 border border-gray-700 p-6 rounded-lg shadow-lg hover:shadow-xl hover:border-indigo-500 transition-all"
          >
            <div class="flex items-center">
              <div class="flex-shrink-0 bg-yellow-500 rounded-md p-3">
                <svg class="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                </svg>
              </div>
              <div class="ml-5 w-0 flex-1">
                <dl>
                  <dt class="text-sm font-medium text-gray-400 truncate">{{ t('home.cards.files.title') }}</dt>
                  <dd class="text-lg font-medium text-white">{{ t('home.cards.files.desc') }}</dd>
                </dl>
              </div>
            </div>
          </router-link>
        </div>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'

const { t } = useI18n()
</script>
