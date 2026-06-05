<template>
  <div 
    ref="dropdownRef"
    class="relative w-full"
    :class="{ 'group': showTooltip && label }"
  >
    <!-- Tooltip (hover/focus only) -->
    <div
      v-if="showTooltip && label"
      class="pointer-events-none absolute -top-7 left-2 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity duration-150"
    >
      <div class="px-2 py-1 rounded-md bg-sky-50 text-sky-700 border border-sky-200 text-xs font-semibold [.dark_&]:bg-sky-500/15 [.dark_&]:text-sky-200 [.dark_&]:border-sky-500/25">
        {{ label }}
      </div>
    </div>

    <!-- 按钮 -->
      <div 
        tabindex="0" 
        role="button" 
        :class="buttonClass"
        @click="toggleDropdown"
        class="flex w-full px-3 py-2 bg-white border border-gray-200 rounded-lg text-gray-950 text-sm whitespace-nowrap items-center justify-between hover:border-indigo-500 transition-colors cursor-pointer min-h-10 [.dark_&]:bg-gray-700 [.dark_&]:border-gray-600 [.dark_&]:text-white"
      > 
        <span class="truncate flex-1 text-left">{{ selectedLabel }}</span>
        <svg 
          class="w-3 h-3 shrink-0 ml-2 transition-transform"
          :class="{ 'rotate-180': openDropdown }"
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
        </svg>
      </div>
    
    <!-- 下拉菜单浮层 -->
    <Teleport to="body">
      <!-- aspect模式：宽高比网格 -->
      <div 
        v-if="openDropdown && mode === 'aspect'"
        ref="dropdownContentRef"
        tabindex="0"
        class="fixed z-9999 mt-1 shadow-xl bg-white border border-indigo-300 rounded-lg p-4 w-[300px] max-h-[50vh] overflow-y-auto [.dark_&]:bg-gray-800 [.dark_&]:border-indigo-500"
        :style="dropdownStyle"
      >
        <div class="grid gap-2 w-full grid-cols-5">
          <div 
            v-for="(item, index) in options" 
            @click="handleSelected(item, index)" 
            :key="index+'_snb'" 
            :class="[
              'text-center cursor-pointer rounded-md p-1 h-12 w-12 transition-colors',
              getValue(item) == modelValue 
                ? 'bg-indigo-600 ring-2 ring-indigo-400' 
                : 'bg-gray-100 hover:bg-gray-200 [.dark_&]:bg-gray-700 [.dark_&]:hover:bg-gray-600'
            ]"
          >
            <div class="grid grid-rows-2 items-center justify-items-center h-full gap-1">
              <div class="bg-indigo-400 rounded-sm" :style="getSizeStyle(item)"></div>
              <div class="text-center text-xs text-gray-700 [.dark_&]:text-white">{{ getLabel(item) }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- list模式：列表 -->
      <ul 
        v-else-if="openDropdown && mode === 'list'"
        ref="dropdownContentRef"
        tabindex="0"
        class="fixed z-9999 mt-1 shadow-xl bg-white border border-indigo-300 rounded-lg p-2 w-full min-w-[200px] max-h-[50vh] overflow-y-auto [.dark_&]:bg-gray-800 [.dark_&]:border-indigo-500"
        :style="dropdownStyle"
      >
        <li 
          v-for="(item, index) in options" 
          :key="'TSD_' + index + name" 
          @click="handleSelected(item, index)"
          :class="[
            'px-3 py-2 rounded-md cursor-pointer transition-colors',
            getValue(item) == modelValue 
              ? 'bg-indigo-600 text-white' 
              : 'text-gray-700 hover:bg-gray-100 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-700'
          ]"
        >
          {{ getLabel(item) }}
        </li>
      </ul>

      <!-- panel模式：按钮面板 -->
      <div 
        v-else-if="openDropdown && mode === 'panel'"
        ref="dropdownContentRef"
        tabindex="0"
        class="fixed z-9999 mt-1 shadow-xl bg-white border border-indigo-300 rounded-lg p-4 w-[300px] max-h-[50vh] overflow-y-auto [.dark_&]:bg-gray-800 [.dark_&]:border-indigo-500"
        :style="dropdownStyle"
      >
        <div class="flex gap-2 flex-wrap items-center">
          <div 
            v-for="(item, index) in options" 
            @click="handleSelected(item, index)" 
            :key="index+'_snb'" 
            :class="[
              'text-center cursor-pointer rounded-md p-2 min-w-12 h-10 items-center justify-center flex transition-colors',
              getValue(item) == modelValue 
                ? 'bg-indigo-600 text-white ring-2 ring-indigo-400' 
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200 [.dark_&]:bg-gray-700 [.dark_&]:text-gray-300 [.dark_&]:hover:bg-gray-600'
            ]"
          >
            {{ getLabel(item) }}
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, onMounted, onUnmounted, computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'

const { t } = useI18n()

interface SelectOption {
  value?: string | number
  val?: string | number
  label?: string
  width?: number
  height?: number
}

const props = withDefaults(defineProps<{
  name: string
  modelValue: string | number
  init: SelectOption[]
  def?: string | number
  label?: string
  mode?: 'list' | 'aspect' | 'panel'
  style?: string
  buttonClass?: string
  showTooltip?: boolean
}>(), {
  name: '',
  modelValue: '',
  init: () => [],
  def: undefined,
  label: '',
  mode: 'list',
  style: 'dropdown-top dropdown-center',
  buttonClass: '',
  showTooltip: false,
})

const emit = defineEmits<{
  'update:modelValue': [value: string | number]
}>()

const openDropdown = ref(false)
const dropdownRef = ref<HTMLElement | null>(null)
const dropdownContentRef = ref<HTMLElement | null>(null)
const dropdownStyle = ref<Record<string, string>>({})

const options = computed(() => props.init)

// 确保始终有默认值
const ensureDefaultValue = () => {
  // 如果当前值无效且选项列表不为空
  if (
    (props.modelValue === '' || props.modelValue === null || props.modelValue === undefined) &&
    options.value.length > 0 &&
    options.value[0]
  ) {
    // 优先使用 def，否则使用第一个选项的值
    const defaultValue = props.def !== undefined && props.def !== '' && props.def !== null
      ? props.def
      : getValue(options.value[0])
    
    // 只有当值确实改变时才触发更新
    if (defaultValue !== props.modelValue) {
      emit('update:modelValue', defaultValue)
    }
  }
}

// 组件挂载时确保有默认值
onMounted(() => {
  ensureDefaultValue()
})

// 当选项列表变化时，如果当前值无效，设置默认值
watch(() => options.value, () => {
  ensureDefaultValue()
}, { immediate: true })

function getValue(item: SelectOption | string | number): string | number {
  if (typeof item === 'string' || typeof item === 'number') {
    return item
  }
  const value = item.value ?? item.val
  if (value !== undefined) {
    return value
  }
  // 如果都没有，返回默认值
  return String(item)
}

function getLabel(item: SelectOption | string | number): string {
  if (typeof item === 'string' || typeof item === 'number') {
    return String(item)
  }
  return item.label ?? String(getValue(item))
}

const selectedLabel = computed(() => {
  // 如果有当前值，找到对应的选项
  if (props.modelValue !== '' && props.modelValue !== null && props.modelValue !== undefined) {
    const option = options.value.find(opt => getValue(opt) === props.modelValue)
    if (option) {
      return getLabel(option)
    }
  }
  
  // 如果没有当前值，使用默认值或第一个选项
  if (props.def !== undefined && props.def !== '' && props.def !== null) {
    return String(props.def)
  }
  
  // 如果选项列表不为空，使用第一个选项的标签
  if (options.value.length > 0 && options.value[0]) {
    return getLabel(options.value[0])
  }
  
  // 最后回退到标签或占位符
  return props.label || t('components.aspectRatio.pleaseSelect')
})

function toggleDropdown() {
  openDropdown.value = !openDropdown.value
  if (openDropdown.value) {
    // 延迟一帧确保DOM已更新
    nextTick(() => {
      // 再延迟一帧确保内容已渲染
      requestAnimationFrame(() => {
        adjustDropdownPosition()
      })
    })
  }
}

const adjustDropdownPosition = () => {
  if (!dropdownRef.value || !dropdownContentRef.value) return
  
  const dropdown = dropdownRef.value
  const content = dropdownContentRef.value
  const rect = dropdown.getBoundingClientRect()
  
  const viewportWidth = window.innerWidth
  const viewportHeight = window.innerHeight
  
  // 获取实际渲染后的内容尺寸
  const contentRect = content.getBoundingClientRect()
  const contentHeight = contentRect.height || content.offsetHeight
  const contentWidth = contentRect.width || content.offsetWidth
  
  // 计算可用空间
  const spaceBelow = viewportHeight - rect.bottom
  const spaceAbove = rect.top
  
  // 决定显示在上方还是下方
  // 如果下方空间不足（小于内容高度+边距），且上方空间更大，则显示在上方
  const margin = 8 // 边距
  const showAbove = spaceBelow < contentHeight + margin && spaceAbove > spaceBelow
  
  let top: number
  if (showAbove) {
    // 显示在上方：使用固定定位，相对于视口顶部
    top = rect.top - contentHeight - margin
  } else {
    // 显示在下方：使用固定定位，相对于视口顶部
    top = rect.bottom + margin
  }
  
  // 计算水平位置（使用固定定位，相对于视口左侧）
  let left = 0
  
  if (props.mode === 'list') {
    // 列表模式：与按钮同宽，左对齐
    left = rect.left
  } else {
    // 其他模式：尝试居中，但如果空间不足则调整
    const idealLeft = rect.left + (rect.width - contentWidth) / 2
    
    // 检查右边界
    const rightEdge = idealLeft + contentWidth
    if (rightEdge > viewportWidth - margin) {
      // 超出右边界，右对齐
      left = rect.right - contentWidth
    } else if (idealLeft < margin) {
      // 超出左边界，左对齐
      left = rect.left
    } else {
      // 居中
      left = idealLeft
    }
  }
  
  dropdownStyle.value = {
    top: `${top}px`,
    left: `${left}px`,
  }
}

const handleSelected = (item?: SelectOption | string | number, _index?: number) => {
  if (document.activeElement instanceof HTMLElement) {
    document.activeElement.blur()
  }
  openDropdown.value = false
  if (item !== undefined) {
    const value = getValue(item)
    emit('update:modelValue', value)
  }
}

// 点击外部关闭下拉菜单
const handleClickOutside = (event: MouseEvent) => {
  if (
    dropdownRef.value &&
    dropdownContentRef.value &&
    !dropdownRef.value.contains(event.target as Node) &&
    !dropdownContentRef.value.contains(event.target as Node)
  ) {
    openDropdown.value = false
  }
}

// 按宽高比计算指定像素大小的方块
function getSizeStyle(item: SelectOption) {
  if (typeof item === 'string' || typeof item === 'number') {
    return {}
  }
  if (!item.width || !item.height) {
    return {}
  }
  // Render a preview rectangle that preserves the given width/height ratio.
  // The larger side is clamped to `base`.
  const base = 20
  const ratio = item.height / item.width // height per width
  let width = 0
  let height = 0
  if (ratio > 1) {
    // Tall: height is the larger side
    height = base
    width = Math.round(base / ratio)
  } else {
    // Wide: width is the larger side
    width = base
    height = Math.round(base * ratio)
  }
  return {
    height: `${height}px`,
    width: `${width}px`,
  }
}

// 监听窗口大小变化
const handleResize = () => {
  if (openDropdown.value) {
    adjustDropdownPosition()
  }
}

onMounted(() => {
  window.addEventListener('resize', handleResize)
  document.addEventListener('click', handleClickOutside)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  document.removeEventListener('click', handleClickOutside)
})
</script>


