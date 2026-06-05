export interface ResolutionConfig {
  value: string | number
  label: string
}

export const resolutionList: ResolutionConfig[] = [
  { value: 1, label: '1K' },
  { value: 2, label: '2K' },
  { value: 4, label: '4K' },
]

export const videoResolutionList: ResolutionConfig[] = [
  { value: 480, label: '480P' },
  { value: 720, label: '720P' },
]

