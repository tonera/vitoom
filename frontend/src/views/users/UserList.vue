<template>
  <div class="flex h-full min-h-0 flex-col overflow-hidden bg-gray-50 text-gray-950 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100">
    <div class="shrink-0 border-b border-gray-200 bg-white px-5 py-4 [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
      <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 class="text-xl font-semibold">{{ t('users.title') }}</h1>
          <p class="mt-1 text-sm text-gray-500 [.dark_&]:text-gray-400">
            {{ t('users.subtitle') }}
          </p>
        </div>
        <button
          type="button"
          class="inline-flex items-center justify-center rounded-full bg-gray-950 px-4 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:opacity-60 [.dark_&]:bg-white [.dark_&]:text-gray-950 [.dark_&]:hover:bg-gray-200 cursor-pointer"
          :disabled="loading || formOpen"
          @click="openCreate"
        >
          {{ t('users.createUser') }}
        </button>
      </div>
    </div>

    <div class="min-h-0 flex-1 overflow-y-auto p-5">
      <section class="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
        <div class="flex flex-col gap-3 border-b border-gray-100 px-5 py-4 sm:flex-row sm:items-center sm:justify-between [.dark_&]:border-gray-800">
          <div class="flex min-w-0 flex-1 items-center gap-2">
            <input
              v-model="searchKeyword"
              type="search"
              :placeholder="t('users.searchPlaceholder')"
              class="w-full max-w-md rounded-xl border border-gray-200 bg-white px-3 py-2 text-sm outline-none transition focus:border-gray-400 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950 [.dark_&]:focus:border-gray-500"
              @keydown.enter.prevent="applySearch"
            >
            <button
              type="button"
              class="shrink-0 rounded-xl border border-gray-200 px-4 py-2 text-sm font-medium text-gray-700 transition hover:bg-gray-50 [.dark_&]:border-gray-700 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800 cursor-pointer"
              :disabled="loading"
              @click="applySearch"
            >
              {{ t('common.search') }}
            </button>
          </div>
          <button
            type="button"
            class="text-sm text-gray-500 transition hover:text-gray-950 [.dark_&]:text-gray-400 [.dark_&]:hover:text-white"
            :disabled="loading"
            @click="loadUsers"
          >
            {{ t('common.refresh') }}
          </button>
        </div>

        <div v-if="loading" class="px-5 py-10 text-center text-sm text-gray-500 [.dark_&]:text-gray-400">
          {{ t('common.loading') }}
        </div>
        <div v-else-if="items.length === 0" class="px-5 py-10 text-center text-sm text-gray-500 [.dark_&]:text-gray-400">
          {{ t('users.noData') }}
        </div>
        <div v-else class="overflow-x-auto">
          <table class="min-w-full text-left text-sm">
            <thead class="border-b border-gray-100 bg-gray-50/80 text-xs uppercase tracking-wide text-gray-500 [.dark_&]:border-gray-800 [.dark_&]:bg-gray-950/40 [.dark_&]:text-gray-400">
              <tr>
                <th class="px-5 py-3 font-medium">{{ t('users.columns.user') }}</th>
                <th class="px-5 py-3 font-medium">{{ t('users.columns.status') }}</th>
                <th class="px-5 py-3 font-medium">{{ t('users.columns.role') }}</th>
                <th class="px-5 py-3 font-medium">{{ t('users.columns.createdAt') }}</th>
                <th class="px-5 py-3 font-medium text-right">{{ t('users.columns.actions') }}</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-100 [.dark_&]:divide-gray-800">
              <tr v-for="item in items" :key="item.id" class="hover:bg-gray-50/70 [.dark_&]:hover:bg-gray-800/40">
                <td class="px-5 py-4">
                  <div class="font-medium text-gray-950 [.dark_&]:text-white">{{ item.nickname || t('users.noNickname') }}</div>
                  <div class="mt-0.5 break-all text-xs text-gray-500 [.dark_&]:text-gray-400">{{ item.email }}</div>
                </td>
                <td class="px-5 py-4">
                  <span
                    class="rounded-full px-2 py-0.5 text-xs font-medium"
                    :class="statusClass(item.status)"
                  >
                    {{ statusLabel(item.status) }}
                  </span>
                </td>
                <td class="px-5 py-4">
                  <span
                    class="rounded-full px-2 py-0.5 text-xs font-medium"
                    :class="item.is_admin ? 'bg-indigo-50 text-indigo-700 [.dark_&]:bg-indigo-950/40 [.dark_&]:text-indigo-300' : 'bg-gray-100 text-gray-600 [.dark_&]:bg-gray-800 [.dark_&]:text-gray-300'"
                  >
                    {{ item.is_admin ? t('users.roleAdmin') : t('users.roleUser') }}
                  </span>
                </td>
                <td class="px-5 py-4 text-gray-600 [.dark_&]:text-gray-300">
                  {{ formatDate(item.created_at) }}
                </td>
                <td class="px-5 py-4">
                  <div class="flex items-center justify-end gap-2">
                    <button
                      type="button"
                      class="rounded-lg px-3 py-1.5 text-sm text-gray-700 transition hover:bg-gray-100 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800 cursor-pointer"
                      @click="openEdit(item)"
                    >
                      {{ t('users.edit') }}
                    </button>
                    <button
                      v-if="item.status === 'active'"
                      type="button"
                      class="rounded-lg px-3 py-1.5 text-sm text-red-600 transition hover:bg-red-50 [.dark_&]:text-red-400 [.dark_&]:hover:bg-red-950/30 cursor-pointer"
                      @click="openDelete(item)"
                    >
                      {{ t('users.disable') }}
                    </button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div
          v-if="total > 0"
          class="flex flex-col gap-3 border-t border-gray-100 px-5 py-4 sm:flex-row sm:items-center sm:justify-between [.dark_&]:border-gray-800"
        >
          <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">
            {{ t('models.pagination', { total, page, lastPage }) }}
          </div>
          <PaginationBar :page="page" :last-page="lastPage" :per-page="perPage" :show-when-single="true" @change="onPageChange" />
        </div>
      </section>
    </div>

    <UserFormDialog
      :open="formOpen"
      :mode="formMode"
      :user="editingUser"
      @close="closeForm"
      @saved="onSaved"
    />

    <ConfirmDeleteUserDialog
      :open="deleteOpen"
      :user="deletingUser"
      @close="closeDelete"
      @deleted="onSaved"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import PaginationBar from '../../components/PaginationBar.vue'
import { listUsers, type AdminUser } from '../../api/users'
import { showTopSnack } from '../../composables/useTopSnack'
import { handleApiError } from '../../utils/api'
import ConfirmDeleteUserDialog from './components/ConfirmDeleteUserDialog.vue'
import UserFormDialog from './components/UserFormDialog.vue'

const { t } = useI18n()

const loading = ref(false)
const items = ref<AdminUser[]>([])
const total = ref(0)
const page = ref(1)
const perPage = 20
const searchKeyword = ref('')
const appliedKeyword = ref('')

const formOpen = ref(false)
const formMode = ref<'create' | 'edit'>('create')
const editingUser = ref<AdminUser | null>(null)

const deleteOpen = ref(false)
const deletingUser = ref<AdminUser | null>(null)

const lastPage = computed(() => Math.max(1, Math.ceil(total.value / perPage)))

function statusLabel(status: string) {
  if (status === 'active') return t('users.statusActive')
  return t('users.statusDisabled')
}

function statusClass(status: string) {
  if (status === 'active') {
    return 'bg-emerald-50 text-emerald-700 [.dark_&]:bg-emerald-950/40 [.dark_&]:text-emerald-300'
  }
  return 'bg-red-50 text-red-700 [.dark_&]:bg-red-950/40 [.dark_&]:text-red-300'
}

function formatDate(value?: string | null) {
  if (!value) return '-'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString()
}

async function loadUsers() {
  loading.value = true
  try {
    const result = await listUsers({
      keyword: appliedKeyword.value || undefined,
      limit: perPage,
      offset: (page.value - 1) * perPage,
    })
    items.value = result.items || []
    total.value = result.total || 0
  } catch (error) {
    const apiError = handleApiError(error)
    showTopSnack(apiError.message || t('users.loadFailed'))
  } finally {
    loading.value = false
  }
}

function applySearch() {
  appliedKeyword.value = searchKeyword.value.trim()
  page.value = 1
  loadUsers()
}

function onPageChange(payload: { page: number; perPage: number }) {
  page.value = payload.page
  loadUsers()
}

function openCreate() {
  formMode.value = 'create'
  editingUser.value = null
  formOpen.value = true
}

function openEdit(user: AdminUser) {
  formMode.value = 'edit'
  editingUser.value = user
  formOpen.value = true
}

function closeForm() {
  formOpen.value = false
  editingUser.value = null
}

function openDelete(user: AdminUser) {
  deletingUser.value = user
  deleteOpen.value = true
}

function closeDelete() {
  deleteOpen.value = false
  deletingUser.value = null
}

function onSaved() {
  loadUsers()
}

onMounted(() => {
  loadUsers()
})
</script>
