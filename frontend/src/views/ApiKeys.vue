<template>
  <div class="flex h-full min-h-0 flex-col overflow-hidden bg-gray-50 text-gray-950 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100">
    <div class="shrink-0 border-b border-gray-200 bg-white px-5 py-4 [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
      <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 class="text-xl font-semibold">{{ t('common.apiKeys') }}</h1>
          <p class="mt-1 text-sm text-gray-500 [.dark_&]:text-gray-400">
            {{ t('apiKeys.subtitle') }}
          </p>
        </div>
        <button
          type="button"
          class="inline-flex items-center justify-center rounded-full bg-gray-950 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-60 [.dark_&]:bg-white [.dark_&]:text-gray-950 [.dark_&]:hover:bg-gray-200 cursor-pointer"
          :disabled="createModalOpen || creating || loading"
          @click="openCreateModal"
        >
          {{ t('apiKeys.create') }}
        </button>
      </div>
    </div>

    <!-- 创建 API Key 弹窗 -->
    <Teleport to="body">
      <div
        v-if="createModalOpen"
        class="fixed inset-0 z-50 flex items-end justify-center bg-black/45 p-4 sm:items-center"
        role="presentation"
        @click.self="onCreateModalBackdropClick"
      >
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="api-key-create-title"
          class="w-full max-w-lg rounded-2xl border border-gray-200 bg-white p-5 shadow-xl sm:max-w-xl [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900"
          @click.stop
        >
          <template v-if="createModalPhase === 'form'">
            <h2 id="api-key-create-title" class="text-lg font-semibold text-gray-950 [.dark_&]:text-white">
              {{ t('apiKeys.dialog.createTitle') }}
            </h2>
            <p class="mt-1 text-sm text-gray-500 [.dark_&]:text-gray-400">
              {{ t('apiKeys.dialog.createDesc') }}
            </p>

            <label class="mt-4 block">
              <span class="text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('apiKeys.dialog.name') }}</span>
              <input
                v-model="form.name"
                type="text"
                maxlength="100"
                :placeholder="t('apiKeys.dialog.namePlaceholder')"
                class="mt-2 w-full rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-gray-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:focus:border-gray-500"
                @keydown.enter.prevent="submitCreate"
              >
            </label>

            <fieldset class="mt-4">
              <legend class="text-sm font-medium text-gray-700 [.dark_&]:text-gray-300">{{ t('apiKeys.dialog.expiryLegend') }}</legend>
              <div class="mt-2 flex flex-col gap-2">
                <label class="flex cursor-pointer items-center gap-2 text-sm text-gray-800 [.dark_&]:text-gray-200">
                  <input v-model="form.expires_in" type="radio" name="api-key-expires" value="1d" class="shrink-0">
                  <span>{{ t('apiKeys.dialog.expiry1Day') }}</span>
                </label>
                <label class="flex cursor-pointer items-center gap-2 text-sm text-gray-800 [.dark_&]:text-gray-200">
                  <input v-model="form.expires_in" type="radio" name="api-key-expires" value="1m" class="shrink-0">
                  <span>{{ t('apiKeys.dialog.expiry1Month') }}</span>
                </label>
                <label class="flex cursor-pointer items-center gap-2 text-sm text-gray-800 [.dark_&]:text-gray-200">
                  <input v-model="form.expires_in" type="radio" name="api-key-expires" value="1y" class="shrink-0">
                  <span>{{ t('apiKeys.dialog.expiry1Year') }}</span>
                </label>
                <label class="flex cursor-pointer items-center gap-2 text-sm text-gray-800 [.dark_&]:text-gray-200">
                  <input v-model="form.expires_in" type="radio" name="api-key-expires" value="never" class="shrink-0">
                  <span>{{ t('apiKeys.dialog.expiryNever') }}</span>
                </label>
              </div>
            </fieldset>

            <div class="mt-6 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
              <button
                type="button"
                class="rounded-xl border border-gray-200 px-4 py-2 text-sm font-medium text-gray-700 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-60 [.dark_&]:border-gray-700 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800 cursor-pointer"
                :disabled="creating"
                @click="closeCreateModal"
              >
                {{ t('common.cancel') }}
              </button>
              <button
                type="button"
                class="rounded-xl bg-gray-950 px-4 py-2 text-sm font-medium text-white transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-60 [.dark_&]:bg-white [.dark_&]:text-gray-950 [.dark_&]:hover:bg-gray-200 cursor-pointer"
                :disabled="creating || loading"
                @click="submitCreate"
              >
                {{ creating ? t('apiKeys.dialog.creating') : t('apiKeys.create') }}
              </button>
            </div>
          </template>

          <template v-else>
            <h2 id="api-key-create-title" class="text-lg font-semibold text-gray-950 [.dark_&]:text-white">
              {{ t('apiKeys.dialog.createdTitle') }}
            </h2>
            <div
              class="mt-4 rounded-xl border border-amber-300 bg-amber-50 p-4 text-amber-950 [.dark_&]:border-amber-800 [.dark_&]:bg-amber-950/35 [.dark_&]:text-amber-100"
            >
              <p class="text-sm font-semibold leading-snug">
                {{ t('apiKeys.dialog.createdWarning') }}
              </p>
              <div
                v-if="modalDisplayedKey"
                class="relative mt-3 max-w-full overflow-hidden rounded-lg bg-white [.dark_&]:bg-gray-950"
              >
                <button
                  type="button"
                  class="absolute right-2 top-2 z-10 flex h-8 w-8 cursor-pointer items-center justify-center rounded-lg text-amber-900/80 transition hover:bg-amber-100/90 hover:text-amber-950 [.dark_&]:text-amber-200 [.dark_&]:hover:bg-gray-800 [.dark_&]:hover:text-amber-100"
                  :aria-label="t('apiKeys.dialog.copyKeyAria')"
                  :title="t('apiKeys.dialog.copyKey')"
                  @click="copyKey(modalDisplayedKey)"
                >
                  <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                    <path
                      stroke-linecap="round"
                      stroke-linejoin="round"
                      stroke-width="2"
                      d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                    />
                  </svg>
                </button>
                <code
                  class="block max-w-full overflow-x-auto break-all px-3 pb-3 pt-3 pr-11 font-mono text-xs leading-relaxed text-gray-900 [.dark_&]:text-gray-100"
                >{{ modalDisplayedKey }}</code>
              </div>
            </div>
            <div class="mt-6 flex justify-end">
              <button
                type="button"
                class="rounded-xl bg-gray-950 px-5 py-2 text-sm font-medium text-white transition hover:bg-gray-800 [.dark_&]:bg-white [.dark_&]:text-gray-950 [.dark_&]:hover:bg-gray-200 cursor-pointer"
                @click="confirmCreateModal"
              >
                {{ t('common.confirm') }}
              </button>
            </div>
          </template>
        </div>
      </div>
    </Teleport>

    <div class="min-h-0 flex-1 overflow-y-auto p-5">
      <section class="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
        <div class="flex items-center justify-between border-b border-gray-100 px-5 py-4 [.dark_&]:border-gray-800">
          <h2 class="text-base font-semibold">{{ t('apiKeys.list.title') }}</h2>
          <button
            type="button"
            class="text-sm text-gray-500 transition hover:text-gray-950 [.dark_&]:text-gray-400 [.dark_&]:hover:text-white"
            :disabled="loading"
            @click="loadKeys"
          >
            {{ t('common.refresh') }}
          </button>
        </div>

        <div v-if="loading" class="px-5 py-10 text-center text-sm text-gray-500 [.dark_&]:text-gray-400">
          {{ t('common.loading') }}
        </div>
        <div v-else-if="items.length === 0" class="px-5 py-10 text-center text-sm text-gray-500 [.dark_&]:text-gray-400">
          {{ t('apiKeys.list.empty') }}
        </div>
        <div v-else class="divide-y divide-gray-100 [.dark_&]:divide-gray-800">
          <article
            v-for="item in items"
            :key="item.id"
            class="flex flex-col gap-4 px-5 py-4 sm:flex-row sm:items-center sm:justify-between"
          >
            <div class="min-w-0">
              <div class="flex flex-wrap items-center gap-2">
                <h3 class="truncate text-sm font-semibold">{{ item.name || t('apiKeys.list.unnamed') }}</h3>
                <span
                  class="rounded-full px-2 py-0.5 text-xs font-medium"
                  :class="item.is_expired ? 'bg-red-50 text-red-700 [.dark_&]:bg-red-950/40 [.dark_&]:text-red-300' : 'bg-emerald-50 text-emerald-700 [.dark_&]:bg-emerald-950/40 [.dark_&]:text-emerald-300'"
                >
                  {{ item.is_expired ? t('apiKeys.list.expired') : t('apiKeys.list.available') }}
                </span>
              </div>
              <p class="mt-1 font-mono text-xs text-gray-500 [.dark_&]:text-gray-400">{{ item.key_prefix }}...</p>
              <p class="mt-2 text-xs text-gray-500 [.dark_&]:text-gray-400">
                {{ t('apiKeys.list.meta', {
                  created: formatDate(item.created_at),
                  expires: formatExpiry(item.expires_at),
                  lastUsed: formatDate(item.last_used_at),
                }) }}
              </p>
            </div>
            <button
              type="button"
              class="self-start rounded-xl border border-red-200 px-3 py-2 text-sm font-medium text-red-600 transition hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-60 [.dark_&]:border-red-900/70 [.dark_&]:text-red-300 [.dark_&]:hover:bg-red-950/30 sm:self-center cursor-pointer"
              :disabled="deletingId === item.id"
              @click="handleDelete(item)"
            >
              {{ deletingId === item.id ? t('apiKeys.list.deleting') : t('common.delete') }}
            </button>
          </article>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  createApiKey,
  deleteApiKey,
  listApiKeys,
  type ApiKeyExpiration,
  type UserApiKey,
} from '../api/apiKeys'
import { showTopSnack } from '../composables/useTopSnack'
import { handleApiError } from '../utils/api'

const { t, locale } = useI18n()

const items = ref<UserApiKey[]>([])
const loading = ref(false)
const creating = ref(false)
const deletingId = ref<string | null>(null)
const createModalOpen = ref(false)
const createModalPhase = ref<'form' | 'success'>('form')
const modalDisplayedKey = ref<string | null>(null)

const form = reactive<{ name: string; expires_in: ApiKeyExpiration }>({
  name: '',
  expires_in: '1y',
})

const formatDate = (value?: string | null) => {
  if (!value) return t('apiKeys.never')
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return t('apiKeys.unknown')
  return date.toLocaleString(locale.value, { hour12: false })
}

const formatExpiry = (value?: string | null) => {
  return value ? formatDate(value) : t('apiKeys.dialog.expiryNever')
}

const loadKeys = async () => {
  loading.value = true
  try {
    const result = await listApiKeys()
    items.value = result.items || []
  } catch (error) {
    const apiError = handleApiError(error)
    showTopSnack(apiError.message || t('apiKeys.loadFailed'))
  } finally {
    loading.value = false
  }
}

const resetCreateForm = () => {
  form.name = ''
  form.expires_in = '1y'
}

const resetCreateModalState = () => {
  createModalPhase.value = 'form'
  modalDisplayedKey.value = null
  resetCreateForm()
}

const openCreateModal = () => {
  resetCreateModalState()
  createModalOpen.value = true
}

const onCreateModalBackdropClick = () => {
  if (createModalPhase.value === 'success') return
  closeCreateModal()
}

const closeCreateModal = () => {
  if (creating.value) return
  if (createModalPhase.value !== 'form') return
  createModalOpen.value = false
  resetCreateModalState()
}

const confirmCreateModal = () => {
  createModalOpen.value = false
  resetCreateModalState()
}

const submitCreate = async () => {
  creating.value = true
  try {
    const created = await createApiKey({
      name: form.name.trim() || undefined,
      expires_in: form.expires_in,
    })
    modalDisplayedKey.value = created.key
    createModalPhase.value = 'success'
    await loadKeys()
  } catch (error) {
    const apiError = handleApiError(error)
    showTopSnack(apiError.message || t('apiKeys.createFailed'))
  } finally {
    creating.value = false
  }
}

const copyKey = async (key: string | null) => {
  const s = String(key ?? '').trim()
  if (!s) return
  try {
    await navigator.clipboard.writeText(s)
    showTopSnack(t('apiKeys.copied'))
  } catch (error) {
    console.error('Failed to copy API key:', error)
    showTopSnack(t('apiKeys.copyFailed'))
  }
}

const handleDelete = async (item: UserApiKey) => {
  const name = item.name || item.key_prefix
  const confirmed = window.confirm(t('apiKeys.deleteConfirm', { name }))
  if (!confirmed) return

  deletingId.value = item.id
  try {
    await deleteApiKey(item.id)
    await loadKeys()
    showTopSnack(t('apiKeys.deleted'))
  } catch (error) {
    const apiError = handleApiError(error)
    showTopSnack(apiError.message || t('apiKeys.deleteFailed'))
  } finally {
    deletingId.value = null
  }
}

onMounted(loadKeys)
</script>
