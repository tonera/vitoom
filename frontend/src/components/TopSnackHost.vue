<template>
  <Teleport to="body">
    <div
      class="pointer-events-none fixed inset-x-0 top-[5.5rem] z-9999 flex justify-center px-4 sm:top-[6rem]"
      aria-live="polite"
    >
      <Transition name="vt-snack">
        <div
          v-if="snackbar.open"
          :class="panelClass"
          role="alert"
          @click.stop
        >
          <div class="flex min-w-0 items-start gap-3">
            <span :class="iconWrapClass" aria-hidden="true">
              <svg v-if="snackbar.variant === 'error'" class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  stroke-width="2"
                  d="M12 9v4m0 4h.01M10.29 3.86 1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
                />
              </svg>
              <svg v-else-if="snackbar.variant === 'success'" class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <svg v-else-if="snackbar.variant === 'info'" class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <svg v-else class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </span>

            <p class="min-w-0 flex-1 text-sm font-medium leading-6">
              {{ snackbar.message }}
            </p>

            <button
              type="button"
              class="mt-0 flex h-6 w-6 shrink-0 items-center justify-center rounded-lg opacity-70 transition-opacity hover:opacity-100 cursor-pointer"
              :aria-label="t('common.close')"
              @click="closeTopSnack"
            >
              <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      </Transition>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useTopSnack } from '../composables/useTopSnack'

const { t } = useI18n()
const { snackbar, closeTopSnack } = useTopSnack()

const panelClass = computed(() => {
  switch (snackbar.value.variant) {
    case 'error':
      return 'vt-top-snack vt-top-snack--error pointer-events-auto w-full max-w-lg cursor-default rounded-2xl border px-4 py-3.5 shadow-xl ring-1 backdrop-blur-sm'
    case 'success':
      return 'vt-top-snack vt-top-snack--success pointer-events-auto w-full max-w-lg cursor-default rounded-2xl border px-4 py-3.5 shadow-xl ring-1 backdrop-blur-sm'
    case 'info':
      return 'vt-top-snack vt-top-snack--info pointer-events-auto w-full max-w-lg cursor-default rounded-2xl border px-4 py-3.5 shadow-xl ring-1 backdrop-blur-sm'
    default:
      return 'vt-top-snack vt-top-snack--default pointer-events-auto w-full max-w-lg cursor-default rounded-2xl border px-4 py-3.5 shadow-xl ring-1 backdrop-blur-sm'
  }
})

const iconWrapClass = computed(() => {
  const base = 'flex h-6 w-6 shrink-0 items-center justify-center rounded-full'
  switch (snackbar.value.variant) {
    case 'error':
      return `${base} bg-red-100 text-red-600 [.dark_&]:bg-red-500/20 [.dark_&]:text-red-300`
    case 'success':
      return `${base} bg-emerald-100 text-emerald-600 [.dark_&]:bg-emerald-500/20 [.dark_&]:text-emerald-300`
    case 'info':
      return `${base} bg-sky-100 text-sky-600 [.dark_&]:bg-sky-500/20 [.dark_&]:text-sky-300`
    default:
      return `${base} bg-slate-100 text-slate-600 [.dark_&]:bg-slate-500/20 [.dark_&]:text-slate-300`
  }
})
</script>
