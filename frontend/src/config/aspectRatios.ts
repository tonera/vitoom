/**
 * 宽高比配置
 * 用于图片生成等场景
 */
export interface AspectRatioConfig {
  label: string
  height: number
  width: number
  val: string
}

export const sdAspList: AspectRatioConfig[] = [
  { label: '1:1', height: 1024, width: 1024, val: '1:1' },
  { label: '3:4', height: 1152, width: 896, val: '4:3' },
  { label: '4:3', height: 896, width: 1152, val: '3:4' },
  { label: '9:16', height: 1344, width: 768, val: '16:9' },
  { label: '2:3', height: 1216, width: 832, val: '3:2' },
  { label: '3:2', height: 832, width: 1216, val: '2:3' },
  { label: '16:9', height: 768, width: 1344, val: '9:16' },
]

