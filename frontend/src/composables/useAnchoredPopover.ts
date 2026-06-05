import { nextTick, onBeforeUnmount, ref, watch } from 'vue'

export type UseAnchoredPopoverOptions = {
  /**
   * Overlay padding to viewport edges
   */
  paddingPx?: number
  /**
   * Gap between button and popover
   */
  gapPx?: number
}

/**
 * 给“按钮锚点 + fixed popover”提供统一的定位、RAF 节流、以及 Escape/resize/scroll 自动重算。
 * 适用于 Image* 页面里的高级选项浮层。
 */
export function useAnchoredPopover(opts: UseAnchoredPopoverOptions = {}) {
  const paddingPx = Number.isFinite(Number(opts.paddingPx)) ? Math.max(0, Math.floor(Number(opts.paddingPx))) : 12
  const gapPx = Number.isFinite(Number(opts.gapPx)) ? Math.max(0, Math.floor(Number(opts.gapPx))) : 10

  const open = ref(false)
  const anchorRef = ref<HTMLElement | null>(null)
  const popoverRef = ref<HTMLElement | null>(null)
  const style = ref<Record<string, string>>({})

  let raf = 0

  async function position() {
    if (typeof window === 'undefined') return
    await nextTick()
    const btn = anchorRef.value
    const pop = popoverRef.value
    if (!btn || !pop) return

    const btnRect = btn.getBoundingClientRect()
    const popRect = pop.getBoundingClientRect()

    let left = btnRect.left
    left = Math.max(paddingPx, Math.min(left, window.innerWidth - paddingPx - popRect.width))

    let top = btnRect.top - gapPx - popRect.height
    if (top < paddingPx) top = btnRect.bottom + gapPx
    top = Math.max(paddingPx, Math.min(top, window.innerHeight - paddingPx - popRect.height))

    style.value = { left: `${left}px`, top: `${top}px` }
  }

  function schedulePosition() {
    if (typeof window === 'undefined') return
    if (raf) cancelAnimationFrame(raf)
    raf = window.requestAnimationFrame(() => {
      raf = 0
      void position()
    })
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape' && open.value) open.value = false
  }

  function toggle() {
    open.value = !open.value
    if (open.value) schedulePosition()
  }

  function close() {
    open.value = false
  }

  watch(open, (isOpen) => {
    if (typeof window === 'undefined') return
    if (isOpen) {
      schedulePosition()
      window.addEventListener('keydown', onKeydown)
      window.addEventListener('resize', schedulePosition)
      window.addEventListener('scroll', schedulePosition, true)
    } else {
      window.removeEventListener('keydown', onKeydown)
      window.removeEventListener('resize', schedulePosition)
      window.removeEventListener('scroll', schedulePosition, true)
    }
  })

  onBeforeUnmount(() => {
    if (typeof window !== 'undefined') {
      window.removeEventListener('keydown', onKeydown)
      window.removeEventListener('resize', schedulePosition)
      window.removeEventListener('scroll', schedulePosition, true)
    }
    if (raf) cancelAnimationFrame(raf)
    raf = 0
  })

  return {
    open,
    anchorRef,
    popoverRef,
    style,
    toggle,
    close,
    schedulePosition,
  }
}

