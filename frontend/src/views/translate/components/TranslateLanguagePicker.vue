<template>
  <div class="relative">
    <button
      ref="anchorRef"
      type="button"
      class="inline-flex cursor-pointer items-center gap-1.5 rounded-xl border border-gray-200 bg-white px-3 py-1.5 text-sm text-gray-800 transition hover:border-indigo-300 hover:bg-indigo-50/60 [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900/60 [.dark_&]:text-gray-100 [.dark_&]:hover:border-indigo-500/40 [.dark_&]:hover:bg-indigo-500/10"
      :aria-label="label"
      @click="toggle"
    >
      <span class="text-xs text-gray-500 [.dark_&]:text-gray-400">{{ label }}</span>
      <span class="font-medium">{{ currentLabel }}</span>
      <svg class="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
      </svg>
    </button>

    <Teleport to="body">
      <div
        v-if="open"
        class="fixed inset-0 z-[120]"
        @click="close"
      />
      <div
        v-if="open"
        ref="popoverRef"
        class="fixed z-[121] w-56 rounded-2xl border border-gray-200 bg-white p-2 shadow-xl [.dark_&]:border-gray-700 [.dark_&]:bg-gray-900"
        :style="style"
        @click.stop
      >
        <button
          v-for="opt in options"
          :key="opt.value"
          type="button"
          class="flex w-full cursor-pointer items-center rounded-xl px-3 py-2 text-left text-sm transition hover:bg-gray-100 [.dark_&]:hover:bg-gray-800"
          :class="opt.value === modelValue ? 'bg-indigo-50 text-indigo-700 font-medium [.dark_&]:bg-indigo-500/15 [.dark_&]:text-indigo-200' : 'text-gray-800 [.dark_&]:text-gray-100'"
          @click="select(opt.value)"
        >
          {{ opt.label }}
        </button>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useAnchoredPopover } from '../../../composables/useAnchoredPopover'

const props = defineProps<{
  modelValue: string
  label: string
  options: Array<{ value: string; label: string }>
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', value: string): void
}>()

const { open, anchorRef, popoverRef, style, toggle, close } = useAnchoredPopover()
// NOTE: template refs (ref="anchorRef") are not counted by TS noUnusedLocals
void anchorRef
void popoverRef

const currentLabel = computed(() => {
  const hit = props.options.find((x) => x.value === props.modelValue)
  return hit?.label || props.modelValue || '—'
})

function select(value: string) {
  emit('update:modelValue', value)
  close()
}
</script>
