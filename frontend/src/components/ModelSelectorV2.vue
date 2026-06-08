<template>
  <div class="w-full">
    <!-- Trigger -->
    <button
      type="button"
      class="group w-full px-3 flex items-center gap-3 transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500/80 cursor-pointer"
      :class="triggerClass"
      @click="openModelDialog"
    >
      <div
        class="shrink-0 rounded-lg overflow-hidden border border-gray-200 bg-gray-100 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/50"
        :class="thumbClass"
      >
        <img
          v-if="triggerThumbResolved && !triggerThumbFailed"
          :src="triggerThumbResolved"
          class="w-full h-full object-cover"
          :alt="triggerTitle"
          @error="triggerThumbFailed = true"
        />
        <div
          v-else
          class="w-full h-full flex items-center justify-center px-1 text-[10px] leading-tight text-center text-gray-600 font-medium line-clamp-2 [.dark_&]:text-gray-300"
        >
          {{ triggerThumbFallback }}
        </div>
      </div>

      <div class="min-w-0 flex-1 text-left">
        <div class="flex items-center gap-2 min-w-0">
          <div class="text-sm font-medium text-gray-950 truncate [.dark_&]:text-white">
            {{ triggerTitle }}
          </div>
          <span
            v-if="triggerMeta?.family"
            class="shrink-0 inline-flex items-center px-2 py-0.5 rounded-md text-[11px] leading-none bg-indigo-500/15 text-indigo-200 border border-indigo-500/20"
          >
            {{ triggerMeta.family }}
          </span>
          <span
            v-if="storageLabel(triggerMeta)"
            class="shrink-0 inline-flex items-center px-2 py-0.5 rounded-md text-[11px] leading-none bg-gray-100 text-gray-700 border border-gray-200 [.dark_&]:bg-black/40 [.dark_&]:text-gray-100 [.dark_&]:border-white/10"
          >
            {{ storageLabel(triggerMeta) }}
          </span>
        </div>
        <div v-if="showSubtitle" class="mt-0.5 text-xs text-gray-400 truncate">
          {{ triggerSubtitle }}
        </div>
      </div>

      <button
        v-if="allowClear && hasSelection"
        type="button"
        class="shrink-0 p-1.5 rounded-lg text-gray-500 hover:text-gray-950 hover:bg-gray-100 transition-all cursor-pointer [.dark_&]:text-gray-400 [.dark_&]:hover:text-white [.dark_&]:hover:bg-gray-700/60"
        :class="clearBtnClass"
        :title="t('components.modelSelector.clear')"
        @click.stop="clearValue"
      >
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>

      <div class="shrink-0 text-gray-400">
        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
        </svg>
      </div>
    </button>

    <!-- Modal -->
    <Teleport to="body">
      <div
        v-if="showDialog"
        class="fixed inset-0 z-9999 bg-black/60 flex items-center justify-center p-4"
        @click.self="closeDialog"
      >
        <div class="w-full max-w-6xl h-[85vh] overflow-hidden flex flex-col rounded-xl border border-gray-200 bg-white shadow-2xl [.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/55">
          <!-- Header -->
          <div class="px-5 py-4 border-b border-gray-200 flex items-center justify-between gap-4 [.dark_&]:border-gray-700/70">
            <div class="flex items-center gap-3 min-w-0">
              <div class="w-9 h-9 bg-indigo-600 rounded-xl flex items-center justify-center shrink-0">
                <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <div class="min-w-0">
                <div class="text-lg font-semibold text-gray-950 truncate [.dark_&]:text-white">{{ dialogTitle }}</div>
                <div class="text-xs text-gray-400 truncate">
                  {{ dialogStatusHint }} {{ dialogTotalHint }}
                </div>
              </div>
            </div>
            <button
              type="button"
              class="p-2 rounded-lg text-gray-500 hover:text-gray-950 hover:bg-gray-100 transition-colors cursor-pointer [.dark_&]:text-gray-400 [.dark_&]:hover:text-white [.dark_&]:hover:bg-gray-800/60"
              @click="closeDialog"
              :aria-label="t('components.modelSelector.close')"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <!-- Body: two columns -->
          <div class="flex-1 min-h-0 flex">
            <!-- Sidebar -->
            <aside class="w-80 shrink-0 border-r border-gray-200 bg-gray-50/80 flex flex-col min-h-0 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-900/20">
              <div class="p-4 space-y-4 overflow-y-auto vt-scroll flex-1 min-h-0">
                <!-- Search -->
                <div>
                  <div class="text-xs text-gray-400 mb-2">{{ t('components.modelSelector.search') }}</div>
                  <div class="relative">
                    <input
                      v-model="searchKeyword"
                      type="text"
                      :placeholder="isLoraMode ? t('components.modelSelector.searchLoraPlaceholder') : t('components.modelSelector.searchModelPlaceholder')"
                      class="w-full px-3 py-2 pl-10 bg-white border border-gray-200 rounded-xl text-gray-950 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500/80 [.dark_&]:bg-gray-800/60 [.dark_&]:border-gray-700/70 [.dark_&]:text-white [.dark_&]:placeholder-gray-500"
                    />
                    <svg class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                    <button
                      v-if="searchKeyword"
                      type="button"
                      class="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-lg text-gray-500 hover:text-gray-950 hover:bg-gray-100 cursor-pointer [.dark_&]:text-gray-400 [.dark_&]:hover:text-white [.dark_&]:hover:bg-gray-700/60"
                      @click="searchKeyword = ''"
                      :aria-label="t('components.modelSelector.clearSearch')"
                    >
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                </div>

                <!-- Filters -->
                <div class="rounded-xl border border-gray-200 bg-white p-3 space-y-3 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35">
                  <div class="flex items-center justify-between">
                    <div class="text-xs text-gray-400">{{ t('components.modelSelector.filter') }}</div>
                    <button
                      type="button"
                      class="text-xs text-gray-500 hover:text-gray-950 cursor-pointer [.dark_&]:text-gray-400 [.dark_&]:hover:text-white"
                      @click="resetFilters"
                    >
                      {{ t('components.modelSelector.reset') }}
                    </button>
                  </div>

                  <div>
                    <div class="text-xs text-gray-400 mb-2">{{ t('components.modelSelector.storage') }}</div>
                    <div class="flex flex-wrap gap-2">
                      <FilterChip
                        v-for="s in storageFilters"
                        :key="s"
                        :active="currentStorageFilter === s"
                        @click="handleStorageFilterChange(s)"
                      >
                        {{ s === 'all' ? t('components.modelSelector.all') : storageValueToLabel(s) }}
                      </FilterChip>
                    </div>
                  </div>

                  <div v-if="classFilters.length > 1">
                    <div class="text-xs text-gray-400 mb-2">{{ t('components.modelSelector.category') }}</div>
                    <div class="flex flex-wrap gap-2">
                      <FilterChip
                        v-for="c in classFilters"
                        :key="c"
                        :active="currentClassFilter === c"
                        @click="handleClassFilterChange(c)"
                      >
                        {{ c === 'all' ? t('components.modelSelector.all') : c }}
                      </FilterChip>
                    </div>
                  </div>
                </div>

                <!-- LoRA: weight controls (replaces old "结果" block) -->
                <div v-if="isLoraMode" class="rounded-xl border border-gray-200 bg-white p-3 space-y-3 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35">
                  <div class="flex items-center justify-between">
                    <div class="text-xs text-gray-400">{{ t('components.modelSelector.loraParams') }}</div>
                    <div class="text-xs text-gray-500">
                      {{ selectedIdList.length > 0 ? t('components.modelSelector.itemCount', { count: selectedIdList.length }) : t('components.modelSelector.notSelected') }}
                    </div>
                  </div>

                  <div v-if="selectedIdList.length === 0" class="text-[11px] text-gray-500">
                    {{ t('components.modelSelector.loraWeightHint') }}
                  </div>

                  <div v-else class="space-y-3">
                    <div
                      v-for="id in selectedIdList"
                      :key="id"
                      class="rounded-lg border border-gray-200 bg-gray-50 p-3 space-y-2 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-900/20"
                    >
                      <div class="flex items-center justify-between gap-2">
                        <div class="min-w-0 text-sm text-gray-950 truncate [.dark_&]:text-white">
                          {{ modelMap[id]?.label || id }}
                        </div>
                        <button
                          type="button"
                          class="shrink-0 p-1.5 rounded-lg text-gray-500 hover:text-gray-950 hover:bg-gray-100 transition-colors cursor-pointer [.dark_&]:text-gray-400 [.dark_&]:hover:text-white [.dark_&]:hover:bg-gray-800/60"
                          :title="t('components.modelSelector.remove')"
                          @click="removeSelected(id)"
                        >
                          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      </div>

                      <div class="flex items-center gap-3">
                        <div class="relative shrink-0">
                          <input
                            type="number"
                            :value="getLoraWeight(id)"
                            @input="(e) => setLoraWeight(id, Number((e.target as HTMLInputElement).value))"
                            :min="-1"
                            :max="2"
                            :step="0.1"
                            class="w-16 px-2 py-1 bg-white border border-gray-200 rounded text-gray-950 text-sm text-center focus:outline-none focus:ring-2 focus:ring-indigo-500/80 [.dark_&]:bg-gray-800/60 [.dark_&]:border-gray-700/70 [.dark_&]:text-white"
                          />
                        </div>
                        <div class="flex-1">
                          <input
                            type="range"
                            :value="getLoraWeight(id)"
                            @input="(e) => setLoraWeight(id, Number((e.target as HTMLInputElement).value))"
                            :min="-1"
                            :max="2"
                            :step="0.1"
                            class="w-full range range-xs range-indigo-500"
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                <!-- Lora actions -->
                <div v-if="isLoraMode" class="rounded-xl border border-gray-200 bg-white p-3 space-y-2 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35">
                  <div class="flex items-center justify-between">
                    <div class="text-xs text-gray-400">
                      {{ t('components.modelSelector.selectedCount', { count: selectedSet.size }) }}
                      <span v-if="selectedSet.size >= 3" class="text-red-300 ml-1">{{ t('components.modelSelector.maxThree') }}</span>
                    </div>
                  </div>
                  <div class="flex gap-2">
                    <button
                      type="button"
                      class="flex-1 px-3 py-2 rounded-lg bg-white border border-gray-200 text-gray-700 hover:bg-gray-50 transition-colors cursor-pointer [.dark_&]:bg-gray-800/50 [.dark_&]:border-gray-700/70 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800/80"
                      @click="closeDialog"
                    >
                      {{ t('components.modelSelector.cancel') }}
                    </button>
                    <button
                      type="button"
                      class="flex-1 px-3 py-2 rounded-lg bg-white border border-gray-200 text-gray-700 hover:bg-gray-50 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed [.dark_&]:bg-gray-800/50 [.dark_&]:border-gray-700/70 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-800/80"
                      :disabled="selectedSet.size === 0"
                      @click="clearSelection"
                    >
                      {{ t('components.modelSelector.reset') }}
                    </button>
                    <button
                      type="button"
                      class="flex-1 px-3 py-2 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                      :disabled="selectedSet.size === 0"
                      @click="confirmSelection"
                    >
                      {{ t('components.modelSelector.confirm') }}
                    </button>
                  </div>
                </div>

                <!-- Multi mode helpers -->
                <div v-if="mode === 'multiple' && !isLoraMode" class="vt-help">
                  {{ t('components.modelSelector.multiSelectHint') }}
                </div>
              </div>
            </aside>

            <!-- Main grid -->
            <main class="flex-1 min-w-0 flex flex-col">
              <div class="flex-1 min-h-0 overflow-y-auto vt-scroll p-4">
                <div v-if="filteredModels.length === 0" class="text-center py-14">
                  <div class="w-14 h-14 mx-auto mb-4 bg-gray-100 rounded-full flex items-center justify-center border border-gray-200 [.dark_&]:bg-gray-800/70 [.dark_&]:border-gray-700/70">
                    <svg class="w-7 h-7 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>
                  <div class="text-gray-700 font-medium [.dark_&]:text-gray-300">{{ t('components.modelSelector.noMatch') }}</div>
                  <div class="text-gray-500 text-sm mt-1">{{ t('components.modelSelector.tryShorterKeyword') }}</div>
                </div>

                <div v-else class="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-4">
                  <button
                    v-for="model in filteredModels"
                    :key="model.value"
                    type="button"
                    class="group text-left rounded-xl overflow-hidden border transition-all relative bg-white hover:bg-gray-50 cursor-pointer [.dark_&]:bg-gray-800/35 [.dark_&]:hover:bg-gray-800/55"
                    :class="isSelected(model.value)
                      ? 'border-indigo-500/70 ring-2 ring-indigo-500/60 ring-offset-0'
                      : 'border-gray-200 hover:border-gray-300 [.dark_&]:border-gray-700/70 [.dark_&]:hover:border-gray-600/80'"
                    @click="handleToggleModel(model)"
                  >
                    <div class="relative w-full aspect-2/3">
                      <img
                        v-if="showModelThumb(model)"
                        :src="resolveThumbUrl(model.thumb)"
                        class="w-full h-full object-cover group-hover:scale-[1.1] transition-transform duration-300"
                        :alt="model.label"
                        @error="onCardThumbError(model.value)"
                      />
                      <div
                        v-else
                        class="w-full h-full bg-gray-100 flex items-center justify-center px-2 text-gray-600 text-xs font-medium text-center leading-snug line-clamp-3 [.dark_&]:bg-gray-800 [.dark_&]:text-gray-300"
                      >
                        {{ model.label }}
                      </div>
                      <div class="absolute inset-0 bg-linear-to-t from-black/80 via-black/20 to-transparent"></div>

                      <!-- (1) top-left: 模型类型 | 存储 -->
                      <div class="absolute top-2 left-2 right-10">
                        <div class="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-black/55 border border-white/10 text-[10px] leading-none text-gray-100 backdrop-blur-sm max-w-full">
                          <span class="truncate">{{ (model.asset_type || 'checkpoint') }}</span>
                          <span class="opacity-70">|</span>
                          <span class="truncate">{{ storageLabel(model) || '-' }}</span>
                        </div>
                      </div>

                      <!-- (2) bottom: 分类(tag) + 模型名称 -->
                      <div class="absolute left-0 right-0 bottom-0 p-3">
                        <div class="mb-1.5">
                          <span
                            v-if="(model.family || '').trim()"
                            class="inline-flex items-center px-2 py-0.5 rounded-md text-[10px] leading-none bg-indigo-500/20 text-indigo-100 border border-indigo-500/25 backdrop-blur-sm"
                          >
                            {{ model.family }}
                          </span>
                        </div>
                        <div class="text-white font-semibold text-sm leading-snug line-clamp-2 drop-shadow min-h-[2.75em]">
                          {{ model.label }}
                        </div>
                      </div>

                      <div
                        v-if="isSelected(model.value)"
                        class="absolute top-2 right-2 w-6 h-6 bg-indigo-600 rounded-full flex items-center justify-center shadow"
                      >
                        <svg class="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                        </svg>
                      </div>
                    </div>
                  </button>
                </div>
              </div>

              <!-- Pagination (moved here, under model list, align right) -->
              <div v-if="lastPage > 1" class="border-t border-gray-200 bg-gray-50 px-4 py-3 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-900/10">
                <PaginationBar
                  :page="currentPage"
                  :last-page="lastPage"
                  :per-page="perPage"
                  @change="(p) => emit('pageChange', p)"
                />
              </div>

              <!-- Footer (multi-select) -->
              <div v-if="mode === 'multiple' && !isLoraMode" class="border-t border-gray-200 p-4 flex items-center justify-between gap-3 bg-gray-50 [.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-900/10">
                <div class="text-sm text-gray-600 [.dark_&]:text-gray-300">
                  {{ t('components.modelSelector.selectedCount', { count: selectedSet.size }) }}
                </div>
                <div class="flex gap-2">
                  <button
                    type="button"
                    class="px-4 py-2 rounded-lg bg-gray-100 text-gray-700 hover:bg-gray-200 transition-colors disabled:opacity-50 cursor-pointer disabled:cursor-not-allowed [.dark_&]:bg-gray-700/70 [.dark_&]:text-gray-200 [.dark_&]:hover:bg-gray-700"
                    :disabled="selectedSet.size === 0"
                    @click="clearSelection"
                  >
                    {{ t('components.modelSelector.clear') }}
                  </button>
                  <button
                    type="button"
                    class="px-4 py-2 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700 transition-colors disabled:opacity-50 cursor-pointer disabled:cursor-not-allowed"
                    :disabled="selectedSet.size === 0"
                    @click="confirmSelection"
                  >
                    {{ t('components.modelSelector.confirm') }}
                  </button>
                </div>
              </div>
            </main>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { computed, defineComponent, h, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { showTopSnack } from '../composables/useTopSnack'
import PaginationBar from './PaginationBar.vue'

const { t } = useI18n()

interface ModelOption {
  value: string
  label: string
  thumb?: string
  storage_mode?: 'local' | 'cloud' | string
  family?: string
  asset_type?: string
}

const props = withDefaults(
  defineProps<{
    modelValue?: string | string[] | Array<{ value: string; weight?: number; locked?: boolean }>
    models?: ModelOption[]
    mode?: 'single' | 'multiple'
    allowClear?: boolean
    variant?: 'default' | 'compact'
    ck_point?: 'checkpoint' | 'lora' | 'all'
    family?: string
    totalCount?: number
    meta?: {
      current_page?: number
      from?: number
      last_page?: number
      per_page?: number
      to?: number
      total?: number
      filter_options?: {
        storage_modes?: string[]
        families?: string[]
      }
    }
  }>(),
  {
    modelValue: '',
    models: () => [],
    mode: 'multiple',
    allowClear: true,
    variant: 'default',
    ck_point: 'checkpoint',
    totalCount: undefined,
    meta: undefined,
  }
)

const emit = defineEmits<{
  'update:modelValue': [value: string | string[] | Array<{ value: string; weight?: number; locked?: boolean }>]
  pageChange: [payload: { page: number; perPage: number }]
  searchChange: [keyword: string]
  filterChange: [payload: { storageType?: string; family?: string }]
  open: []
}>()

const showDialog = ref(false)
const searchKeyword = ref('')
const currentStorageFilter = ref<string>('all')
const currentClassFilter = ref<string>('all')
const selectedSet = ref<Set<string>>(new Set())
const modelMap = ref<Record<string, ModelOption>>({})
const loraMetaDraft = ref<Record<string, { weight: number; locked: boolean }>>({})

const FilterChip = defineComponent({
  name: 'FilterChip',
  props: { active: { type: Boolean, default: false } },
  emits: ['click'],
  setup(p, { emit: e, slots }) {
    return () =>
      h(
        'button',
        {
          type: 'button',
          class: [
            'px-3 py-1 rounded-full text-xs font-medium transition-colors border cursor-pointer',
            p.active
              ? 'bg-indigo-600 text-white border-indigo-500/40'
              : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50 hover:text-gray-950 [.dark_&]:bg-gray-800/40 [.dark_&]:text-gray-300 [.dark_&]:border-gray-700/70 [.dark_&]:hover:bg-gray-800/70 [.dark_&]:hover:text-white',
          ],
          onClick: () => e('click'),
        },
        slots.default ? slots.default() : ''
      )
  },
})

watch(
  () => props.models,
  (models) => {
    modelMap.value = {}
    models.forEach((m) => (modelMap.value[m.value] = m))
  },
  { immediate: true }
)

const uniqSorted = (arr: Array<string | undefined | null>) => {
  return Array.from(new Set(arr.map((x) => String(x || '').trim()).filter((x) => x.length > 0))).sort((a, b) =>
    a.localeCompare(b)
  )
}

// Filters are now provided by backend via /api/models meta.filter_options
const storageFilters = computed(() => {
  const list = uniqSorted(props.meta?.filter_options?.storage_modes || [])
  return ['all', ...list]
})

const classFilters = computed(() => {
  const list = uniqSorted(props.meta?.filter_options?.families || [])
  return ['all', ...list]
})

const isLoraMode = computed(() => props.ck_point === 'lora')
const dialogTitle = computed(() =>
  isLoraMode.value ? t('components.modelSelector.selectLora') : t('components.modelSelector.selectModel')
)
const selectedIdList = computed(() => Array.from(selectedSet.value))

watch(
  [storageFilters, classFilters],
  () => {
    if (currentStorageFilter.value !== 'all' && !storageFilters.value.includes(currentStorageFilter.value)) {
      currentStorageFilter.value = 'all'
    }
    if (currentClassFilter.value !== 'all' && !classFilters.value.includes(currentClassFilter.value)) {
      currentClassFilter.value = 'all'
    }
  },
  { immediate: true }
)

const selectedModels = computed(() => {
  if (isLoraMode.value) {
    const arr = Array.isArray(props.modelValue) ? (props.modelValue as any[]) : []
    return arr.map((x) => (x && typeof x === 'object' ? String(x.value || '') : '')).filter((v) => v && modelMap.value[v])
  }
  if (props.mode === 'multiple') {
    const values = Array.isArray(props.modelValue) ? props.modelValue : []
    return (values as any[]).filter((v) => v && typeof v === 'string' && modelMap.value[v])
  }
  const value = typeof props.modelValue === 'string' ? props.modelValue : ''
  return value && modelMap.value[value] ? [value] : []
})

const hasSelection = computed(() => selectedModels.value.length > 0)
const isCompact = computed(() => props.variant === 'compact')
const totalCountDisplay = computed(() => (typeof props.totalCount === 'number' && props.totalCount > 0 ? props.totalCount : props.models.length))
const dialogStatusHint = computed(() => {
  if (props.mode === 'multiple') {
    return t('components.modelSelector.selectedCount', { count: selectedSet.value.size })
  }
  if (selectedModels.value[0]) {
    return t('components.modelSelector.clickToSwitch')
  }
  return t('components.modelSelector.pickOneModel')
})
const dialogTotalHint = computed(() =>
  t('components.modelSelector.totalDisplay', {
    loaded: totalCountDisplay.value,
    total: typeof props.totalCount === 'number' ? props.totalCount : totalCountDisplay.value,
  })
)
const loadedCount = computed(() => props.models.length)
const currentPage = computed(() => Number(props.meta?.current_page || 1))
const lastPage = computed(() => Number(props.meta?.last_page || 1))
const perPage = computed(() => Number(props.meta?.per_page || loadedCount.value || 10))

// Debounced remote-search trigger
const searchDebounceMs = 350
const searchTimer = ref<number | null>(null)
watch(
  searchKeyword,
  (val) => {
    if (searchTimer.value) window.clearTimeout(searchTimer.value)
    searchTimer.value = window.setTimeout(() => {
      emit('searchChange', String(val || '').trim())
    }, searchDebounceMs)
  },
  { immediate: false }
)

// 移除本地筛选逻辑，改为依赖 API 返回的数据
// 筛选现在通过 API 调用实现，所以直接返回 props.models
const filteredModels = computed(() => {
  return props.models
})

const storageValueToLabel = (v: string) => {
  const s = String(v || '').toLowerCase()
  if (s === 'local') return t('components.modelSelector.local')
  if (s === 'cloud') return t('components.modelSelector.cloud')
  return v
}

const storageLabel = (m?: ModelOption) => {
  if (!m) return ''
  const st = (m.storage_mode || '').toLowerCase()
  if (st === 'local') return t('components.modelSelector.local')
  if (st === 'cloud') return t('components.modelSelector.cloud')
  return ''
}

const triggerMeta = computed(() => {
  const key = selectedModels.value[0]
  return key ? modelMap.value[key] : undefined
})

const triggerTitle = computed(() => {
  if (isLoraMode.value) {
    const n = selectedModels.value.length
    return n > 0 ? t('components.modelSelector.selectedLoras', { count: n }) : t('components.modelSelector.selectLora')
  }
  if (props.mode === 'multiple') {
    return hasSelection.value
      ? t('components.modelSelector.selectedModels', { count: selectedModels.value.length })
      : t('components.modelSelector.selectModel')
  }
  const meta = triggerMeta.value
  return meta ? meta.label : t('components.modelSelector.selectModel')
})

const triggerSubtitle = computed(() => {
  if (props.mode === 'multiple') {
    return hasSelection.value
      ? t('components.modelSelector.clickToAdjust')
      : t('components.modelSelector.pickModelsHint')
  }
  return triggerMeta.value ? triggerMeta.value.value : t('components.modelSelector.clickToPickOne')
})

function resolveThumbUrl(url?: string): string {
  const u = (url || '').trim()
  if (!u) return ''
  // absolute urls or data urls are left as-is
  if (/^(https?:)?\/\//i.test(u) || /^data:/i.test(u)) return u
  // already absolute path
  if (u.startsWith('/')) return u
  // common stored forms: "outputs/xxx" or "resources/outputs/xxx"
  if (u.startsWith('outputs/')) return `/${u}`
  if (u.startsWith('resources/outputs/')) return `/${u.replace(/^resources\//, '')}`
  // fallback: treat as a relative to outputs root
  return `/outputs/${u.replace(/^\/+/, '')}`
}

const triggerThumbFailed = ref(false)
const failedThumbIds = ref<Set<string>>(new Set())

watch(
  () => props.models,
  () => {
    failedThumbIds.value = new Set()
  }
)

const triggerThumb = computed(() => triggerMeta.value?.thumb)
const triggerThumbResolved = computed(() => {
  // 后端 /api/models 已返回 thumb 的绝对 URL，这里只做兜底格式化（兼容老数据）
  const u = triggerThumb.value
  const resolved = resolveThumbUrl(u)
  return resolved || ''
})

watch(triggerThumbResolved, () => {
  triggerThumbFailed.value = false
})

const triggerThumbFallback = computed(() => triggerMeta.value?.label || 'M')

function showModelThumb(model: ModelOption): boolean {
  if (failedThumbIds.value.has(model.value)) return false
  return !!resolveThumbUrl(model.thumb)
}

function onCardThumbError(id: string) {
  failedThumbIds.value = new Set([...failedThumbIds.value, id])
}

const showSubtitle = computed(() => !isCompact.value)

const triggerClass = computed(() => {
  // default: card-like. compact: input-like (fits narrow sidebar form)
  if (isCompact.value) {
    return [
      'min-h-11 py-2 rounded-lg border border-gray-200 bg-white',
      'hover:bg-gray-50 hover:border-gray-300',
      'active:bg-gray-100',
      '[.dark_&]:border-gray-700/70 [.dark_&]:bg-gray-800/35',
      '[.dark_&]:hover:bg-gray-800/55 [.dark_&]:hover:border-gray-600/80',
      '[.dark_&]:active:bg-gray-800/60',
    ].join(' ')
  }
  return [
    'rounded-xl border border-gray-200 bg-gray-50/80 py-2',
    'hover:border-gray-300',
    '[.dark_&]:border-gray-700/60 [.dark_&]:bg-gray-800/35 [.dark_&]:hover:border-gray-500/70',
  ].join(' ')
})

const thumbClass = computed(() => (isCompact.value ? 'w-9 h-9' : 'w-10 h-10'))

const clearBtnClass = computed(() => {
  if (!isCompact.value) return ''
  // reduce visual noise on narrow forms
  return 'opacity-0 group-hover:opacity-100 focus:opacity-100'
})

function handleStorageFilterChange(value: string) {
  currentStorageFilter.value = value
  emit('filterChange', {
    storageType: value === 'all' ? undefined : value,
    family: currentClassFilter.value === 'all' ? undefined : currentClassFilter.value,
  })
}

function handleClassFilterChange(value: string) {
  currentClassFilter.value = value
  emit('filterChange', {
    storageType: currentStorageFilter.value === 'all' ? undefined : currentStorageFilter.value,
    family: value === 'all' ? undefined : value,
  })
}

function resetFilters() {
  currentStorageFilter.value = 'all'
  currentClassFilter.value = 'all'
  searchKeyword.value = ''
  emit('searchChange', '')
  emit('filterChange', { storageType: undefined, family: undefined })
}

function openModelDialog() {
  // LoRA 模式必须先有底模 family（用于服务端按兼容集合过滤 LoRA）
  if (isLoraMode.value && !String(props.family || '').trim()) {
    showTopSnack(t('components.modelSelector.selectModelFirst'))
    return
  }

  emit('open')
  showDialog.value = true
  // 初始化选中集合
  if (isLoraMode.value) {
    const arr = Array.isArray(props.modelValue) ? (props.modelValue as any[]) : []
    const ids = arr
      .map((x) => (x && typeof x === 'object' ? String(x.value || '') : ''))
      .filter((v) => v)
    selectedSet.value = new Set(ids)
    const meta: Record<string, { weight: number; locked: boolean }> = {}
    for (const x of arr) {
      if (!x || typeof x !== 'object') continue
      const id = String(x.value || '')
      if (!id) continue
      meta[id] = { weight: typeof x.weight === 'number' ? x.weight : 0.5, locked: Boolean(x.locked) }
    }
    loraMetaDraft.value = meta
    return
  }

  if (props.mode === 'multiple') {
    const values = Array.isArray(props.modelValue) ? (props.modelValue as any[]) : []
    selectedSet.value = new Set(values.filter((v) => typeof v === 'string' && v))
  } else {
    const value = typeof props.modelValue === 'string' ? props.modelValue : ''
    selectedSet.value = value ? new Set([value]) : new Set()
  }
}

function closeDialog() {
  showDialog.value = false
}

function handleToggleModel(model: ModelOption) {
  if (props.mode === 'multiple') {
    if (selectedSet.value.has(model.value)) {
      selectedSet.value.delete(model.value)
      if (isLoraMode.value) delete loraMetaDraft.value[model.value]
      return
    }
    if (isLoraMode.value && selectedSet.value.size >= 3) return
    selectedSet.value.add(model.value)
    if (isLoraMode.value && !loraMetaDraft.value[model.value]) {
      loraMetaDraft.value[model.value] = { weight: 0.8, locked: false }
    }
    return
  }
  emit('update:modelValue', model.value)
  closeDialog()
}

function isSelected(value: string): boolean {
  if (props.mode === 'multiple') return selectedSet.value.has(value)
  return props.modelValue === value
}

function clearSelection() {
  selectedSet.value.clear()
  if (isLoraMode.value) loraMetaDraft.value = {}
}

function confirmSelection() {
  if (isLoraMode.value) {
    const ids = Array.from(selectedSet.value).slice(0, 3)
    const out = ids.map((id) => {
      const meta = loraMetaDraft.value[id] || { weight: 0.5, locked: false }
      return { value: id, weight: clampLoraWeight(meta.weight), locked: meta.locked }
    })
    emit('update:modelValue', out)
    closeDialog()
    return
  }
  emit('update:modelValue', Array.from(selectedSet.value))
  closeDialog()
}

function clearValue() {
  if (isLoraMode.value) emit('update:modelValue', [])
  else if (props.mode === 'multiple') emit('update:modelValue', [])
  else emit('update:modelValue', '')
}

function clampLoraWeight(v: number): number {
  const n = Number.isFinite(v) ? v : 0.5
  const clamped = Math.max(-1, Math.min(2, n))
  // normalize to 1 decimal step
  return Math.round(clamped * 10) / 10
}

function getLoraWeight(id: string): number {
  const meta = loraMetaDraft.value[id]
  if (!meta) return 0.5
  return clampLoraWeight(meta.weight)
}

function setLoraWeight(id: string, v: number) {
  if (!loraMetaDraft.value[id]) loraMetaDraft.value[id] = { weight: 0.5, locked: false }
  loraMetaDraft.value[id]!.weight = clampLoraWeight(v)
}

function removeSelected(id: string) {
  selectedSet.value.delete(id)
  if (isLoraMode.value) delete loraMetaDraft.value[id]
}

</script>


