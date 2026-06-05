export function formatBytes(n: number) {
  const v = Number(n || 0)
  if (!isFinite(v) || v <= 0) return '0B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let x = v
  let i = 0
  while (x >= 1024 && i < units.length - 1) {
    x /= 1024
    i++
  }
  const fixed = i >= 2 ? 2 : 0
  return `${x.toFixed(fixed)}${units[i]}`
}

