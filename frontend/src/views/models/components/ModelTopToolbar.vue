<template>
  <div class=" p-2">
    <div class="flex flex-wrap items-center gap-3">
      <button
        type="button"
        class="px-4 py-2 rounded-xl bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-500 transition-colors cursor-pointer"
        @click="emit('create')"
      >
        {{ t('models.create') }}
      </button>

      <div class="relative flex-1 min-w-[220px]">
        <input
          :value="searchKeyword"
          type="text"
          :placeholder="t('models.searchPlaceholder')"
          class="w-full rounded-xl border border-gray-200 bg-white py-2 pl-10 pr-3 text-sm text-gray-900 placeholder:text-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
          @input="onSearchInput"
        />
        <svg
          class="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400 [.dark_&]:text-gray-500"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <button
          v-if="String(searchKeyword || '').trim()"
          type="button"
          class="absolute right-2 top-1/2 -translate-y-1/2 cursor-pointer rounded-lg p-1.5 text-gray-400 hover:bg-gray-100 hover:text-gray-900 [.dark_&]:hover:bg-gray-700/60 [.dark_&]:hover:text-white"
          :aria-label="t('models.clearSearch')"
          @click="emit('update:searchKeyword', '')"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <select
        :value="ckTypeFilter"
        class="min-w-0 rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
        @change="emit('update:ckTypeFilter', ($event.target as HTMLSelectElement).value)"
      >
        <option value="all">{{ t('models.allTypes') }}</option>
        <option v-for="item in ckTypeOptions" :key="`ck-${item}`" :value="item">{{ item }}</option>
      </select>

      <select
        :value="familyFilter"
        class="min-w-0 rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
        @change="emit('update:familyFilter', ($event.target as HTMLSelectElement).value)"
      >
        <option value="all">{{ t('models.allFamilies') }}</option>
        <option v-for="c in familyOptions" :key="`mc-${c}`" :value="c">{{ c }}</option>
      </select>

      <select
        :value="typeFilter"
        class="min-w-0 rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
        @change="emit('update:typeFilter', ($event.target as HTMLSelectElement).value)"
      >
        <option value="all">{{ t('models.allTaskTypes') }}</option>
        <option v-for="item in typeOptions" :key="`tp-${item}`" :value="item">{{ item }}</option>
      </select>

      <select
        :value="storageFilter"
        class="min-w-0 rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
        @change="emit('update:storageFilter', ($event.target as HTMLSelectElement).value)"
      >
        <option value="all">{{ t('models.allStorage') }}</option>
        <option v-for="s in storageOptions" :key="`st-${s}`" :value="s">{{ storageValueToLabel(s) }}</option>
      </select>

      <select
        :value="statusFilter"
        class="min-w-0 rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 focus:outline-none focus:ring-2 focus:ring-indigo-500/50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-900/40 [.dark_&]:text-gray-100"
        @change="emit('update:statusFilter', ($event.target as HTMLSelectElement).value)"
      >
        <option value="all">{{ t('models.allStatus') }}</option>
        <option value="active">active</option>
        <option value="inactive">inactive</option>
      </select>

      <div class="flex items-center gap-2">
        <button
          type="button"
          class="cursor-pointer rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm text-gray-700 transition-colors hover:bg-gray-50 [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/50 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800/80"
          @click="emit('importModel')"
        >
          {{ t('models.importModel') }}
        </button>
        <button
          type="button"
          class="cursor-pointer text-xs text-gray-500 hover:text-gray-900 [.dark_&]:text-gray-400 [.dark_&]:hover:text-white"
          @click="emit('reset')"
        >
          {{ t('models.resetFilters') }}
        </button>
      </div>
    </div>

    <div class="mt-3 flex items-center justify-between gap-3 text-xs text-gray-500 [.dark_&]:text-gray-400">
      <div>
        <span v-if="loading">{{ t('models.loading') }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'
import { storageValueToLabel } from './modelListFormatters'

const { t } = useI18n()

defineProps<{
  loading: boolean
  total: number
  searchKeyword: string
  ckTypeFilter: string
  familyFilter: string
  typeFilter: string
  storageFilter: string
  statusFilter: string
  ckTypeOptions: string[]
  familyOptions: string[]
  storageOptions: string[]
  typeOptions: string[]
}>()

const emit = defineEmits<{
  (e: 'create'): void
  (e: 'importModel'): void
  (e: 'reset'): void
  (e: 'update:searchKeyword', v: string): void
  (e: 'update:ckTypeFilter', v: string): void
  (e: 'update:familyFilter', v: string): void
  (e: 'update:typeFilter', v: string): void
  (e: 'update:storageFilter', v: string): void
  (e: 'update:statusFilter', v: string): void
}>()

function onSearchInput(e: Event) {
  const v = (e.target as HTMLInputElement).value
  emit('update:searchKeyword', v)
}
</script>
