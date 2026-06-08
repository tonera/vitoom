<template>
  <div
    class="vt-surface vt-text-smooth flex h-full min-h-0 flex-col gap-4 antialiased text-gray-950 [.dark_&]:text-gray-100"
  >
    <ModelTopToolbar
      v-model:search-keyword="searchKeyword"
      v-model:ck-type-filter="ckTypeFilter"
      v-model:family-filter="familyFilter"
      v-model:type-filter="typeFilter"
      v-model:storage-filter="storageFilter"
      v-model:status-filter="statusFilter"
      :loading="loading"
      :total="total"
      :ck-type-options="ckTypeOptions"
      :family-options="familyOptions"
      :storage-options="storageOptions"
      :type-options="typeOptions"
      @create="openCreate()"
      @import-model="openImportModel()"
      @reset="resetFilters()"
    />

    <!-- Grid (scroll inside) -->
    <div class="vt-scroll min-h-0 flex-1 overflow-y-auto overflow-x-hidden p-1">
      <div v-if="!loading && models.length === 0" class="vt-card-muted p-8 text-center text-gray-500 [.dark_&]:text-gray-400">
        {{ t('models.empty') }}
      </div>

      <div v-else class="grid gap-4 items-stretch grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
        <div v-for="m in models" :key="m.model_key" class="flex">
          <ModelCard
            class="w-full"
            :model="m"
            :busy="busyIds.has(m.model_key)"
            :resolve-backend-public-url="resolveBackendPublicUrl"
            @advanced-config="openAdvancedConfig"
            @edit="openEdit"
            @delete="openDelete"
            @toggle-active="toggleActive"
            @open-download-terminal="openDownloadTerminal"
          />
        </div>
      </div>
    </div>

    <!-- Pagination -->
    <div
      class="mt-2 flex shrink-0 items-center justify-between gap-3 border-t border-gray-200 bg-white/80 px-1 py-2 backdrop-blur [.dark_&]:border-gray-800/60 [.dark_&]:bg-gray-900/50"
    >
      <div class="text-xs text-gray-500 [.dark_&]:text-gray-400">
        {{ t('models.pagination', { total, page, lastPage }) }}
      </div>
      <PaginationBar :page="page" :last-page="lastPage" :per-page="perPage" :show-when-single="true" @change="onPageChange" />
    </div>

    <ModelDownloadTerminalDialog
      :open="showDownloadTerminal"
      :model-key="downloadTerminalModelKey"
      :model="downloadTerminalModel"
      @status="onDownloadTerminalStatus"
      @close="closeDownloadTerminal()"
    />

    <ModelFormDialog :open="showFormModal" :mode="formMode" :model="editingModel" @close="closeFormModal()" @saved="onSavedAndRefresh()" />

    <ConfirmDeleteDialog :open="showDeleteModal" :model="deleting" @close="closeDeleteModal()" @deleted="onSavedAndRefresh()" />

    <ModelImportDialog :open="showImportModelModal" @close="closeImportModel()" @done="onSavedAndRefresh()" />

    <ModelAdvancedConfigDialog
      v-model:open="showAdvancedConfigModal"
      :saving="advancedConfigSaving"
      :model-name="advancedConfigModel?.name"
      :family="advancedConfigModel?.family"
      :initial-config="(advancedConfigModel as any)?.runtime_config"
      @submit="submitAdvancedConfig"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import PaginationBar from '../../components/PaginationBar.vue'
import ModelAdvancedConfigDialog from '../../components/ModelAdvancedConfigDialog.vue'
import ModelDownloadTerminalDialog from './components/ModelDownloadTerminalDialog.vue'
import ModelTopToolbar from './components/ModelTopToolbar.vue'
import ModelCard from './components/ModelCard.vue'
import ModelFormDialog from './components/ModelFormDialog.vue'
import ConfirmDeleteDialog from './components/ConfirmDeleteDialog.vue'
import ModelImportDialog from './components/ModelImportDialog.vue'
import { useModelCatalogMeta } from '../../composables/useModelCatalogMeta'
import { showTopSnack } from '../../composables/useTopSnack'
import {
  activateModel,
  deactivateModel,
  listModels,
  updateModel,
  type ModelRecord,
  type ModelsListMeta,
} from '../../api/models'
import { resolveBackendPublicUrl } from '../../utils/runtimeConfig'
import { connectModelDownload, type WebSocketMessage } from '../../utils/websocket'

const { t } = useI18n()
const { modalityFilterOptions, familyFilterOptions, ensureLoaded: ensureCatalogMetaLoaded } = useModelCatalogMeta()

const loading = ref(false)
const models = ref<ModelRecord[]>([])
const meta = ref<ModelsListMeta | null>(null)

const page = ref(1)
const perPage = ref(15)

const searchKeyword = ref('')
const ckTypeFilter = ref('all')
const familyFilter = ref('all')
const typeFilter = ref('all')
const storageFilter = ref('all')
const statusFilter = ref('all')

const busyIds = ref<Set<string>>(new Set())

// 卡片进度条需要“后台持续更新”，不能依赖下载日志弹窗是否打开
const downloadWsByModelKey = ref<Map<string, any>>(new Map())

function isDownloadRunning(st: any) {
  const s = String(st ?? '').trim().toLowerCase()
  return s === 'downloading' || s === 'pending'
}

function handleDownloadWsMessage(msg: WebSocketMessage) {
  const t = String((msg as any)?.type || '').trim()
  if (t !== 'download_status') return
  // 复用现有更新逻辑（会更新 models 列表里的字段）
  onDownloadTerminalStatus(msg as any)

  // 最终态后可以释放连接
  const st = String((msg as any)?.status || '').trim().toLowerCase()
  const modelKey = String((msg as any)?.model_key || '').trim()
  if (modelKey && (st === 'completed' || st === 'failed' || st === 'canceled')) {
    stopDownloadTracking(modelKey)
  }
}

function ensureDownloadTracking(modelKey: string) {
  const key = String(modelKey || '').trim()
  if (!key) return
  if (downloadWsByModelKey.value.has(key)) return
  const ws = connectModelDownload(key, handleDownloadWsMessage)
  downloadWsByModelKey.value.set(key, ws)
}

function stopDownloadTracking(modelKey: string) {
  const key = String(modelKey || '').trim()
  if (!key) return
  const ws = downloadWsByModelKey.value.get(key)
  if (ws) {
    try {
      ws.disconnect?.()
    } catch {}
  }
  downloadWsByModelKey.value.delete(key)
}

function syncDownloadTrackingWithList() {
  const want = new Set<string>()
  for (const m of models.value || []) {
    const modelKey = String((m as any)?.model_key || '').trim()
    if (!modelKey) continue
    if (isDownloadRunning((m as any)?.download_status)) want.add(modelKey)
  }
  // start new
  for (const modelKey of want) ensureDownloadTracking(modelKey)
  // stop those no longer shown / no longer running
  for (const modelKey of Array.from(downloadWsByModelKey.value.keys())) {
    if (!want.has(modelKey)) stopDownloadTracking(modelKey)
  }
}

// 下载监控弹窗
const showDownloadTerminal = ref(false)
const downloadTerminalModelKey = ref<string>('')
const downloadTerminalModel = ref<ModelRecord | null>(null)
function openDownloadTerminal(m: ModelRecord) {
  const modelKey = String(m?.model_key || '').trim()
  if (!modelKey) return
  downloadTerminalModelKey.value = modelKey
  downloadTerminalModel.value = m
  showDownloadTerminal.value = true
}
function closeDownloadTerminal() {
  showDownloadTerminal.value = false
  downloadTerminalModelKey.value = ''
  downloadTerminalModel.value = null
}

function onDownloadTerminalStatus(payload: any) {
  const modelKey = String(payload?.model_key || '').trim()
  if (!modelKey) return
  const st = String(payload?.status || payload?.download_status || '').trim()
  const p = Number(payload?.progress || 0)
  const bd = Number(payload?.bytes_downloaded || 0)
  const bt = Number(payload?.bytes_total || 0)

  // 1) 更新列表里的 model（用于卡片实时刷新进度条）
  const idx = models.value.findIndex((x) => String((x as any)?.model_key || '') === modelKey)
  if (idx >= 0) {
    const m: any = models.value[idx]
    if (st) m.download_status = st
    if (Number.isFinite(p)) m.download_progress = Math.max(0, Math.min(100, p))
    if (Number.isFinite(bd)) m.download_bytes_downloaded = Math.max(0, bd)
    if (Number.isFinite(bt)) m.download_bytes_total = Math.max(0, bt)
  }

  // 2) 同步更新弹窗传入的 model（避免 props.model 的显示滞后）
  if (downloadTerminalModel.value && String((downloadTerminalModel.value as any)?.model_key || '') === modelKey) {
    const m: any = downloadTerminalModel.value
    if (st) m.download_status = st
    if (Number.isFinite(p)) m.download_progress = Math.max(0, Math.min(100, p))
    if (Number.isFinite(bd)) m.download_bytes_downloaded = Math.max(0, bd)
    if (Number.isFinite(bt)) m.download_bytes_total = Math.max(0, bt)
  }

  // 关键：弹窗里点“继续下载”后，列表页也要开始后台订阅，否则关掉弹窗卡片就不会继续刷新
  const st2 = String(st || '').trim().toLowerCase()
  if (st2 === 'downloading' || st2 === 'pending') {
    ensureDownloadTracking(modelKey)
  } else if (st2 === 'completed' || st2 === 'failed' || st2 === 'canceled' || st2 === 'cancelled') {
    stopDownloadTracking(modelKey)
  }
}

watch(
  models,
  () => syncDownloadTrackingWithList(),
  // 需要 deep：因为下载状态/进度是就地修改 models[idx] 的字段，不会更换数组引用
  { immediate: true, deep: true }
)

// ----- Create / Edit -----
const showFormModal = ref(false)
const formMode = ref<'create' | 'edit'>('create')
const editingModel = ref<ModelRecord | null>(null)
function openCreate() {
  formMode.value = 'create'
  editingModel.value = null
  showFormModal.value = true
}
function openEdit(m: ModelRecord) {
  formMode.value = 'edit'
  editingModel.value = m
  showFormModal.value = true
}
function closeFormModal() {
  showFormModal.value = false
}

// ----- Delete -----
const showDeleteModal = ref(false)
const deleting = ref<ModelRecord | null>(null)
function openDelete(m: ModelRecord) {
  deleting.value = m
  showDeleteModal.value = true
}
function closeDeleteModal() {
  showDeleteModal.value = false
  deleting.value = null
}

// ----- Import Model -----
const showImportModelModal = ref(false)
function openImportModel() {
  showImportModelModal.value = true
}
function closeImportModel() {
  showImportModelModal.value = false
}

function onSavedAndRefresh() {
  fetchList()
}

// ----- Advanced Config -----
const showAdvancedConfigModal = ref(false)
const advancedConfigSaving = ref(false)
const advancedConfigModel = ref<ModelRecord | null>(null)
function openAdvancedConfig(m: ModelRecord) {
  advancedConfigModel.value = m
  showAdvancedConfigModal.value = true
}
async function submitAdvancedConfig(cfg: Record<string, any>) {
  const m = advancedConfigModel.value
  if (!m?.model_key) return showTopSnack(t('models.missingModelKey'))
  if (advancedConfigSaving.value) return

  advancedConfigSaving.value = true
  try {
    await updateModel(m.model_key, { runtime_config: cfg })
    showTopSnack(t('models.advancedConfigSaved'))
    showAdvancedConfigModal.value = false
    fetchList()
  } catch (e: any) {
    console.error(e)
    showTopSnack(e?.message || t('models.saveAdvancedConfigFailed'))
  } finally {
    advancedConfigSaving.value = false
  }
}

const total = computed(() => Number(meta.value?.total || models.value.length || 0))
const lastPage = computed(() => Math.max(1, Number(meta.value?.last_page || 1)))

function uniqSorted(arr: Array<string | undefined | null>) {
  return Array.from(new Set(arr.map((x) => String(x || '').trim()).filter((x) => x.length > 0))).sort((a, b) => a.localeCompare(b))
}

const ckTypeOptions = computed(() => uniqSorted(meta.value?.filter_options?.asset_types || []).filter((x) => x.toLowerCase() !== 'all'))
const familyOptions = familyFilterOptions
const storageOptions = computed(() => uniqSorted(meta.value?.filter_options?.storage_modes || []))
const typeOptions = modalityFilterOptions

const searchDebounceMs = 350
const searchTimer = ref<number | null>(null)
watch(
  searchKeyword,
  () => {
    if (searchTimer.value) window.clearTimeout(searchTimer.value)
    searchTimer.value = window.setTimeout(() => {
      page.value = 1
      fetchList()
    }, searchDebounceMs)
  },
  { immediate: false }
)

watch([ckTypeFilter, familyFilter, typeFilter, storageFilter, statusFilter], () => {
  page.value = 1
  fetchList()
})

async function fetchList() {
  loading.value = true
  try {
    const offset = (page.value - 1) * perPage.value
    const resp = await listModels({
      limit: perPage.value,
      offset,
      name: searchKeyword.value.trim() || undefined,
      asset_type: ckTypeFilter.value !== 'all' ? ckTypeFilter.value : undefined,
      family: ckTypeFilter.value === 'lora' ? undefined : (familyFilter.value !== 'all' ? familyFilter.value : undefined),
      lora_family: ckTypeFilter.value === 'lora' ? (familyFilter.value !== 'all' ? familyFilter.value : undefined) : undefined,
      modality: typeFilter.value !== 'all' ? typeFilter.value : undefined,
      storage_mode: storageFilter.value !== 'all' ? storageFilter.value : undefined,
      service_status: statusFilter.value !== 'all' ? statusFilter.value : undefined,
    })

    models.value = Array.isArray(resp?.data) ? (resp.data as any) : []
    meta.value = (resp?.meta || null) as any
  } catch (e: any) {
    console.error('Failed to fetch models:', e)
    showTopSnack(e?.message || t('models.fetchListFailed'))
    models.value = []
    meta.value = null
  } finally {
    loading.value = false
  }
}

function onPageChange(p: { page: number; perPage: number }) {
  page.value = p.page
  perPage.value = p.perPage
  fetchList()
}

function resetFilters() {
  ckTypeFilter.value = 'all'
  familyFilter.value = 'all'
  typeFilter.value = 'all'
  storageFilter.value = 'all'
  statusFilter.value = 'all'
  searchKeyword.value = ''
  page.value = 1
  fetchList()
}

async function toggleActive(m: ModelRecord) {
  if (!m?.model_key) return
  if (busyIds.value.has(m.model_key)) return

  const prev = m.service_status
  const next = prev === 'active' ? 'inactive' : 'active'

  busyIds.value.add(m.model_key)
  m.service_status = next
  try {
    if (next === 'active') await activateModel(m.model_key)
    else await deactivateModel(m.model_key)
  } catch (e: any) {
    m.service_status = prev
    showTopSnack(e?.message || t('models.updateStatusFailed'))
  } finally {
    busyIds.value.delete(m.model_key)
  }
}

onMounted(() => {
  void ensureCatalogMetaLoaded()
  fetchList()
})

onBeforeUnmount(() => {
  for (const modelKey of Array.from(downloadWsByModelKey.value.keys())) {
    stopDownloadTracking(modelKey)
  }
})
</script>

