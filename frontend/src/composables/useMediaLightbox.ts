import { ref, type ComputedRef } from 'vue'
import type { MediaLightboxItem } from '../components/MediaLightbox.vue'

export function useMediaLightbox(items: ComputedRef<MediaLightboxItem[]>) {
  const mediaOpen = ref(false)
  const mediaActiveKey = ref<string | null>(null)

  function openMediaByKey(key: string) {
    const k = String(key || '')
    if (!k) return
    if (!items.value?.some((x) => x.key === k)) return
    mediaActiveKey.value = k
    mediaOpen.value = true
  }

  return { mediaOpen, mediaActiveKey, openMediaByKey }
}


