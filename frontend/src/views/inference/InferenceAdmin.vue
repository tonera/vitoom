<template>
  <div class="flex h-full min-h-0 flex-col overflow-hidden bg-gray-50 text-gray-950 [.dark_&]:bg-gray-950 [.dark_&]:text-gray-100">
    <div class="shrink-0 border-b border-gray-200 bg-white px-5 py-4 [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
      <div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 class="text-xl font-semibold">{{ t('inferenceAdmin.title') }}</h1>
          <p class="mt-1 text-sm text-gray-500 [.dark_&]:text-gray-400">{{ t('inferenceAdmin.subtitle') }}</p>
        </div>
        <button type="button" class="rounded-full border border-gray-200 px-4 py-2 text-sm font-medium text-gray-700 transition hover:bg-gray-50 [.dark_&]:border-gray-700 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800 cursor-pointer" :disabled="loading" @click="loadServices">
          {{ t('inferenceAdmin.refresh') }}
        </button>
      </div>
    </div>

    <div class="min-h-0 flex-1 overflow-y-auto p-5">
      <section class="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm [.dark_&]:border-gray-800 [.dark_&]:bg-gray-900">
        <div class="flex items-center gap-2 border-b border-gray-100 px-5 py-4 [.dark_&]:border-gray-800">
          <button
            v-for="tab in tabs"
            :key="tab.value"
            type="button"
            class="rounded-full px-4 py-2 text-sm font-medium transition cursor-pointer"
            :class="activeTab === tab.value ? 'bg-gray-950 text-white [.dark_&]:bg-white [.dark_&]:text-gray-950' : 'text-gray-600 hover:bg-gray-100 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800'"
            @click="activeTab = tab.value"
          >
            {{ tab.label }} ({{ tab.count }})
          </button>
        </div>
        <div v-if="loading" class="px-5 py-10 text-center text-sm text-gray-500 [.dark_&]:text-gray-400">{{ t('common.loading') }}</div>
        <div v-else-if="visibleItems.length === 0" class="px-5 py-10 text-center text-sm text-gray-500 [.dark_&]:text-gray-400">{{ t('inferenceAdmin.empty') }}</div>
        <div v-else class="overflow-x-auto">
          <table class="min-w-full text-left text-sm">
            <thead class="border-b border-gray-100 bg-gray-50/80 text-xs uppercase tracking-wide text-gray-500 [.dark_&]:border-gray-800 [.dark_&]:bg-gray-950/40 [.dark_&]:text-gray-400">
              <tr>
                <th class="px-5 py-3 font-medium">{{ t('inferenceAdmin.columns.service') }}</th>
                <th class="px-5 py-3 font-medium">{{ t('inferenceAdmin.columns.ip') }}</th>
                <th class="px-5 py-3 font-medium">{{ t('inferenceAdmin.columns.runtime') }}</th>
                <th class="px-5 py-3 font-medium">{{ t('inferenceAdmin.columns.supervisor') }}</th>
                <th class="px-5 py-3 font-medium">{{ t('inferenceAdmin.columns.ws') }}</th>
                <th class="px-5 py-3 font-medium">{{ t('inferenceAdmin.columns.heartbeat') }}</th>
                <th class="px-5 py-3 font-medium text-right">{{ t('inferenceAdmin.columns.actions') }}</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-gray-100 [.dark_&]:divide-gray-800">
              <tr v-for="item in visibleItems" :key="item.id" class="hover:bg-gray-50/70 [.dark_&]:hover:bg-gray-800/40">
                <td class="px-5 py-4">
                  <div class="font-medium text-gray-950 [.dark_&]:text-white">{{ item.name || item.id }}</div>
                  <div class="mt-0.5 text-xs text-gray-500 [.dark_&]:text-gray-400">
                    {{ item.id }} · {{ item.service_type || '-' }}
                  </div>
                  <div class="mt-1 max-w-md truncate text-xs text-gray-400" :title="item.supervisor_url || ''">
                    {{ item.supervisor_url || '-' }}
                  </div>
                </td>
                <td class="px-5 py-4 text-gray-600 [.dark_&]:text-gray-300">
                  {{ item.client_ip || '-' }}
                </td>
                <td class="px-5 py-4">
                  <span class="rounded-full px-2 py-0.5 text-xs font-medium" :class="runtimeClass(item.runtime_state)">
                    {{ runtimeLabel(item.runtime_state) }}
                  </span>
                </td>
                <td class="px-5 py-4 text-gray-600 [.dark_&]:text-gray-300">
                  <div>{{ item.program_name || item.id }}</div>
                  <div class="mt-0.5 max-w-xs truncate text-xs text-gray-500 [.dark_&]:text-gray-400" :title="agentDetail(item)">
                    {{ supervisorLabel(item) }}
                  </div>
                </td>
                <td class="px-5 py-4">
                  <span class="rounded-full px-2 py-0.5 text-xs font-medium" :class="item.ws_online ? okBadgeClass : warnBadgeClass">
                    {{ item.ws_online ? t('inferenceAdmin.wsOnline') : t('inferenceAdmin.wsOffline') }}
                  </span>
                </td>
                <td class="px-5 py-4">
                  <span class="rounded-full px-2 py-0.5 text-xs font-medium" :class="item.heartbeat_fresh ? okBadgeClass : warnBadgeClass">
                    {{ item.heartbeat_fresh ? t('inferenceAdmin.heartbeatFresh') : t('inferenceAdmin.heartbeatStale') }}
                  </span>
                </td>
                <td class="px-5 py-4">
                  <div class="flex flex-wrap items-center justify-end gap-2">
                    <button type="button" :class="actionButtonClass" :disabled="busyId === item.id || !item.control_available" :title="controlTitle(item)" @click="runAction(item, 'start')">{{ t('inferenceAdmin.start') }}</button>
                    <button type="button" :class="actionButtonClass" :disabled="busyId === item.id || !item.control_available" :title="controlTitle(item)" @click="runAction(item, 'stop')">{{ t('inferenceAdmin.stop') }}</button>
                    <button type="button" :class="actionButtonClass" :disabled="busyId === item.id || !item.control_available" :title="controlTitle(item)" @click="runAction(item, 'restart')">{{ t('inferenceAdmin.restart') }}</button>
                    <button type="button" :class="actionButtonClass" :disabled="busyId === item.id || !item.control_available" :title="controlTitle(item)" @click="openLogs(item)">{{ t('inferenceAdmin.viewLogs') }}</button>
                    <button type="button" :class="actionButtonClass" :disabled="busyId === item.id || !item.control_available" :title="controlTitle(item)" @click="openConfig(item)">{{ t('inferenceAdmin.configure') }}</button>
                    <button v-if="!item.ws_online" type="button" :class="deleteActionButtonClass" :disabled="busyId === item.id" @click="deleteService(item)">{{ t('inferenceAdmin.delete') }}</button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>
    </div>

    <div v-if="logsOpen" class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <section class="flex max-h-[80vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl bg-white shadow-2xl [.dark_&]:bg-gray-900">
        <div class="flex items-center justify-between border-b border-gray-100 px-5 py-4 [.dark_&]:border-gray-800">
          <div>
            <h2 class="text-base font-semibold">{{ t('inferenceAdmin.logsTitle') }}</h2>
            <p class="mt-0.5 text-xs text-gray-500 [.dark_&]:text-gray-400">{{ selectedService?.id }}</p>
          </div>
          <button type="button" class="rounded-lg px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800 cursor-pointer" @click="logsOpen = false">
            {{ t('inferenceAdmin.closeLogs') }}
          </button>
        </div>
        <pre class="min-h-0 flex-1 overflow-auto bg-gray-950 p-4 text-xs leading-5 text-gray-100">{{ logsText || t('inferenceAdmin.logsEmpty') }}</pre>
      </section>
    </div>

    <div v-if="configOpen" class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <section class="flex max-h-[85vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl bg-white shadow-2xl [.dark_&]:bg-gray-900">
        <div class="flex items-center justify-between border-b border-gray-100 px-5 py-4 [.dark_&]:border-gray-800">
          <div>
            <h2 class="text-base font-semibold">{{ t('inferenceAdmin.configTitle') }}</h2>
            <p class="mt-0.5 text-xs text-gray-500 [.dark_&]:text-gray-400">{{ selectedService?.id }}</p>
          </div>
          <button type="button" class="rounded-lg px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800 cursor-pointer" @click="closeConfig">
            {{ t('inferenceAdmin.closeLogs') }}
          </button>
        </div>

        <div class="flex gap-2 border-b border-gray-100 px-5 py-3 [.dark_&]:border-gray-800">
          <button
            type="button"
            class="rounded-full px-4 py-1.5 text-sm font-medium transition cursor-pointer"
            :class="configTab === 'global' ? 'bg-gray-950 text-white [.dark_&]:bg-white [.dark_&]:text-gray-950' : 'text-gray-600 hover:bg-gray-100 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800'"
            @click="switchConfigTab('global')"
          >
            {{ t('inferenceAdmin.configTabGlobal') }}
          </button>
          <button
            type="button"
            class="rounded-full px-4 py-1.5 text-sm font-medium transition cursor-pointer"
            :class="configTab === 'service' ? 'bg-gray-950 text-white [.dark_&]:bg-white [.dark_&]:text-gray-950' : 'text-gray-600 hover:bg-gray-100 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800'"
            @click="switchConfigTab('service')"
          >
            {{ t('inferenceAdmin.configTabService') }}
          </button>
        </div>

        <div class="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          <div v-if="configLoading" class="py-10 text-center text-sm text-gray-500 [.dark_&]:text-gray-400">{{ t('common.loading') }}</div>
          <template v-else>
            <p v-if="configTab === 'global'" class="mb-4 text-xs text-gray-500 [.dark_&]:text-gray-400">
              {{ t('inferenceAdmin.configGlobalHint', { node: selectedService?.supervisor_url || '-' }) }}
            </p>
            <p v-else class="mb-4 text-xs text-gray-500 [.dark_&]:text-gray-400">
              {{ t('inferenceAdmin.configServiceHint', { serviceId: selectedService?.id || '-', serviceType: selectedService?.service_type || '-' }) }}
            </p>
            <p class="mb-4 text-xs text-amber-700 [.dark_&]:text-amber-300">{{ t('inferenceAdmin.configRestartHint') }}</p>

            <div v-if="configTab === 'service' && activeServiceFields.length === 0" class="py-8 text-center text-sm text-gray-500 [.dark_&]:text-gray-400">
              {{ t('inferenceAdmin.configServiceEmpty') }}
            </div>

            <div v-else class="space-y-4">
              <div v-for="field in activeConfigFields" :key="field.path">
                <label class="mb-1 block text-sm font-medium text-gray-700 [.dark_&]:text-gray-200">
                  {{ t(`inferenceAdmin.${field.labelKey}`) }}
                </label>
                <select
                  v-if="field.type === 'select'"
                  v-model="activeFormValues[field.path]"
                  class="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950"
                >
                  <option v-for="option in field.options || []" :key="option" :value="option">{{ option }}</option>
                </select>
                <input
                  v-else-if="field.type === 'number'"
                  v-model.number="activeFormValues[field.path]"
                  type="number"
                  step="any"
                  class="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950"
                />
                <input
                  v-else-if="field.type === 'password'"
                  v-model="activeFormValues[field.path]"
                  type="password"
                  autocomplete="new-password"
                  class="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950"
                />
                <input
                  v-else-if="field.type === 'tags'"
                  :value="tagsInputValue(field.path)"
                  type="text"
                  class="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950"
                  @input="onTagsInput(field.path, ($event.target as HTMLInputElement).value)"
                />
                <input
                  v-else
                  v-model="activeFormValues[field.path]"
                  type="text"
                  class="w-full rounded-lg border border-gray-200 bg-white px-3 py-2 text-sm [.dark_&]:border-gray-700 [.dark_&]:bg-gray-950"
                />
              </div>
            </div>
          </template>
        </div>

        <div class="flex items-center justify-end gap-2 border-t border-gray-100 px-5 py-4 [.dark_&]:border-gray-800">
          <button type="button" class="rounded-lg px-4 py-2 text-sm text-gray-600 hover:bg-gray-100 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-800 cursor-pointer" :disabled="configSaving" @click="closeConfig">
            {{ t('inferenceAdmin.closeLogs') }}
          </button>
          <button type="button" class="inline-flex items-center justify-center gap-2 rounded-lg border border-gray-200 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50 [.dark_&]:border-gray-700 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800 cursor-pointer disabled:cursor-not-allowed" :disabled="configSaving || configLoading || !canSaveConfig" @click="saveConfig(false)">
            <span
              v-if="configSavingMode === 'save'"
              class="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-gray-300 border-t-gray-700 [.dark_&]:border-gray-600 [.dark_&]:border-t-gray-200"
              aria-hidden="true"
            />
            {{ configSavingMode === 'save' ? t('inferenceAdmin.configSaving') : t('inferenceAdmin.configSaveOnly') }}
          </button>
          <button type="button" class="inline-flex items-center justify-center gap-2 rounded-lg bg-gray-950 px-4 py-2 text-sm font-medium text-white hover:bg-gray-800 disabled:opacity-50 [.dark_&]:bg-white [.dark_&]:text-gray-950 [.dark_&]:hover:bg-gray-100 cursor-pointer disabled:cursor-not-allowed" :disabled="configSaving || configLoading || !canSaveConfig" @click="saveConfig(true)">
            <span
              v-if="configSavingMode === 'restart'"
              class="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-white/40 border-t-white [.dark_&]:border-gray-950/40 [.dark_&]:border-t-gray-950"
              aria-hidden="true"
            />
            {{ configSavingMode === 'restart' ? t('inferenceAdmin.configSavingAndRestart') : t('inferenceAdmin.configSaveAndRestart') }}
          </button>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import {
  deleteAdminInferenceService,
  getAdminInferenceGlobalConfig,
  getAdminInferenceServiceConfig,
  getAdminInferenceServiceLogs,
  listAdminInferenceServices,
  restartAdminInferenceService,
  startAdminInferenceService,
  stopAdminInferenceService,
  updateAdminInferenceGlobalConfig,
  updateAdminInferenceServiceConfig,
  type AdminInferenceService,
  type InferenceRuntimeState,
} from '../../api/adminInference'
import { showTopSnack } from '../../composables/useTopSnack'
import { handleApiError } from '../../utils/api'
import {
  GLOBAL_CONFIG_FIELDS,
  buildPatchFromForm,
  formValuesFromConfig,
  serviceConfigFields,
} from './inferenceConfigSchema'

const { t } = useI18n()

const loading = ref(false)
const items = ref<AdminInferenceService[]>([])
const activeTab = ref<'active' | 'offline'>('active')
const busyId = ref('')
const logsOpen = ref(false)
const logsText = ref('')
const selectedService = ref<AdminInferenceService | null>(null)

const configOpen = ref(false)
const configTab = ref<'global' | 'service'>('global')
const configLoading = ref(false)
const configSaving = ref(false)
const configSavingMode = ref<'save' | 'restart' | null>(null)
const globalConfigOriginal = ref<Record<string, unknown>>({})
const serviceConfigOriginal = ref<Record<string, unknown>>({})
const globalFormValues = reactive<Record<string, unknown>>({})
const serviceFormValues = reactive<Record<string, unknown>>({})

const actionButtonClass =
  'rounded-lg px-3 py-1.5 text-sm text-gray-700 transition hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800 cursor-pointer'
const deleteActionButtonClass =
  'rounded-lg px-3 py-1.5 text-sm text-red-600 transition hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-50 [.dark_&]:text-red-300 [.dark_&]:hover:bg-red-950/30 cursor-pointer'
const okBadgeClass = 'bg-emerald-50 text-emerald-700 [.dark_&]:bg-emerald-950/40 [.dark_&]:text-emerald-300'
const warnBadgeClass = 'bg-amber-50 text-amber-700 [.dark_&]:bg-amber-950/40 [.dark_&]:text-amber-300'

const stateClasses: Record<string, string> = {
  online: okBadgeClass,
  degraded: warnBadgeClass,
  offline: 'bg-gray-100 text-gray-600 [.dark_&]:bg-gray-800 [.dark_&]:text-gray-300',
  agent_unreachable: 'bg-red-50 text-red-700 [.dark_&]:bg-red-950/40 [.dark_&]:text-red-300',
  agent_error: 'bg-red-50 text-red-700 [.dark_&]:bg-red-950/40 [.dark_&]:text-red-300',
  unknown: warnBadgeClass,
}

const activeItems = computed(() => items.value.filter((item) => isActiveService(item)))
const offlineItems = computed(() => items.value.filter((item) => !isActiveService(item)))
const visibleItems = computed(() => (activeTab.value === 'active' ? activeItems.value : offlineItems.value))
const tabs = computed(() => [
  { value: 'active' as const, label: t('inferenceAdmin.activeTab'), count: activeItems.value.length },
  { value: 'offline' as const, label: t('inferenceAdmin.offlineTab'), count: offlineItems.value.length },
])

const activeServiceFields = computed(() => {
  const fields = serviceConfigFields(selectedService.value?.service_type)
  return fields.filter((field) => !field.visibleWhen || field.visibleWhen(serviceFormValues))
})

const activeConfigFields = computed(() => {
  if (configTab.value === 'global') return GLOBAL_CONFIG_FIELDS
  return activeServiceFields.value
})

const activeFormValues = computed(() => (configTab.value === 'global' ? globalFormValues : serviceFormValues))

const canSaveConfig = computed(() => {
  if (configTab.value === 'service' && activeServiceFields.value.length === 0) return false
  return true
})

function runtimeLabel(state?: InferenceRuntimeState) {
  return t(`inferenceAdmin.states.${state || 'unknown'}`)
}

function runtimeClass(state?: InferenceRuntimeState) {
  return stateClasses[state || 'unknown'] || stateClasses.unknown
}

function isActiveService(item: AdminInferenceService) {
  return Boolean(item.ws_online || item.heartbeat_fresh || item.agent_reachable || item.supervisor_program)
}

function formatDetail(value: unknown) {
  if (!value) return ''
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function agentDetail(item: AdminInferenceService) {
  return formatDetail(item.agent_programs_error || item.agent_detail)
}

function supervisorLabel(item: AdminInferenceService) {
  if (item.supervisor_program?.state) return item.supervisor_program.state
  if (item.agent_programs_error) return `${t('inferenceAdmin.states.agent_error')}: ${agentDetail(item)}`
  if (item.agent_reachable) return t('inferenceAdmin.agentReachable')
  return t('inferenceAdmin.states.agent_unreachable')
}

function controlTitle(item: AdminInferenceService) {
  if (item.control_available) return ''
  return t('inferenceAdmin.controlUnavailable', {
    reason: item.control_unavailable_reason || '-',
  })
}

function resetFormValues(target: Record<string, unknown>, values: Record<string, unknown>) {
  Object.keys(target).forEach((key) => delete target[key])
  Object.assign(target, values)
}

function tagsInputValue(path: string) {
  const raw = activeFormValues.value[path]
  return Array.isArray(raw) ? raw.join(', ') : String(raw ?? '')
}

function onTagsInput(path: string, raw: string) {
  const values = activeFormValues.value
  values[path] = raw
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
}

async function loadGlobalConfig() {
  if (!selectedService.value) return
  const result = await getAdminInferenceGlobalConfig(selectedService.value.id)
  globalConfigOriginal.value = result.config || {}
  resetFormValues(globalFormValues, formValuesFromConfig(GLOBAL_CONFIG_FIELDS, globalConfigOriginal.value))
}

async function loadServiceConfig() {
  if (!selectedService.value) return
  const fields = serviceConfigFields(selectedService.value.service_type)
  if (fields.length === 0) {
    serviceConfigOriginal.value = {}
    resetFormValues(serviceFormValues, {})
    return
  }
  const result = await getAdminInferenceServiceConfig(selectedService.value.id)
  serviceConfigOriginal.value = result.config || {}
  resetFormValues(serviceFormValues, formValuesFromConfig(fields, serviceConfigOriginal.value))
}

async function loadActiveConfigTab() {
  configLoading.value = true
  try {
    if (configTab.value === 'global') {
      await loadGlobalConfig()
    } else {
      await loadServiceConfig()
    }
  } catch (error) {
    const apiError = handleApiError(error)
    showTopSnack(apiError.message || t('inferenceAdmin.configLoadFailed'))
  } finally {
    configLoading.value = false
  }
}

async function openConfig(item: AdminInferenceService) {
  if (!item.control_available) {
    showTopSnack(controlTitle(item))
    return
  }
  selectedService.value = item
  configTab.value = 'service'
  configOpen.value = true
  await loadActiveConfigTab()
}

function closeConfig() {
  configOpen.value = false
}

async function switchConfigTab(tab: 'global' | 'service') {
  if (configTab.value === tab) return
  configTab.value = tab
  await loadActiveConfigTab()
}

async function saveConfig(restartAfterSave: boolean) {
  if (!selectedService.value || !canSaveConfig.value) return
  configSaving.value = true
  configSavingMode.value = restartAfterSave ? 'restart' : 'save'
  try {
    if (configTab.value === 'global') {
      const patch = buildPatchFromForm(GLOBAL_CONFIG_FIELDS, globalFormValues, globalConfigOriginal.value)
      await updateAdminInferenceGlobalConfig(selectedService.value.id, patch)
      await loadGlobalConfig()
    } else {
      const fields = serviceConfigFields(selectedService.value.service_type)
      const patch = buildPatchFromForm(fields, serviceFormValues, serviceConfigOriginal.value)
      await updateAdminInferenceServiceConfig(selectedService.value.id, patch)
      await loadServiceConfig()
    }
    showTopSnack(t('inferenceAdmin.configSaveSuccess'))
    if (restartAfterSave) {
      await restartAdminInferenceService(selectedService.value.id)
      showTopSnack(t('inferenceAdmin.actionSuccess'))
      await loadServices()
    }
    closeConfig()
  } catch (error) {
    const apiError = handleApiError(error)
    showTopSnack(apiError.message || t('inferenceAdmin.configSaveFailed'))
  } finally {
    configSaving.value = false
    configSavingMode.value = null
  }
}

async function loadServices() {
  loading.value = true
  try {
    const result = await listAdminInferenceServices()
    items.value = result.items || []
  } catch (error) {
    const apiError = handleApiError(error)
    showTopSnack(apiError.message || t('inferenceAdmin.loadFailed'))
  } finally {
    loading.value = false
  }
}

async function runAction(item: AdminInferenceService, action: 'start' | 'stop' | 'restart') {
  if (busyId.value) return
  if (!item.control_available) {
    showTopSnack(controlTitle(item))
    return
  }
  busyId.value = item.id
  try {
    if (action === 'start') await startAdminInferenceService(item.id)
    if (action === 'stop') await stopAdminInferenceService(item.id)
    if (action === 'restart') await restartAdminInferenceService(item.id)
    showTopSnack(t('inferenceAdmin.actionSuccess'))
    await loadServices()
  } catch (error) {
    const apiError = handleApiError(error)
    showTopSnack(apiError.message || t('inferenceAdmin.actionFailed'))
  } finally {
    busyId.value = ''
  }
}

async function deleteService(item: AdminInferenceService) {
  if (busyId.value) return
  const confirmed = window.confirm(t('inferenceAdmin.deleteConfirm', { name: item.name || item.id }))
  if (!confirmed) return

  busyId.value = item.id
  try {
    await deleteAdminInferenceService(item.id)
    if (selectedService.value?.id === item.id) {
      logsOpen.value = false
      configOpen.value = false
      selectedService.value = null
    }
    showTopSnack(t('inferenceAdmin.deleteSuccess'))
    await loadServices()
  } catch (error) {
    const apiError = handleApiError(error)
    showTopSnack(apiError.message || t('inferenceAdmin.deleteFailed'))
  } finally {
    busyId.value = ''
  }
}

async function openLogs(item: AdminInferenceService) {
  if (!item.control_available) {
    showTopSnack(controlTitle(item))
    return
  }
  busyId.value = item.id
  selectedService.value = item
  try {
    const result = await getAdminInferenceServiceLogs(item.id, { stream: 'stdout', tail: 200 })
    logsText.value = (result.lines || []).join('\n')
    logsOpen.value = true
  } catch (error) {
    const apiError = handleApiError(error)
    showTopSnack(apiError.message || t('inferenceAdmin.actionFailed'))
  } finally {
    busyId.value = ''
  }
}

onMounted(() => {
  loadServices()
})
</script>
