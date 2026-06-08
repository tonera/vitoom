export type TopNavItem = {
  labelKey: string
  to: string
  activeRouteNames: string[]
}

export const navItems: TopNavItem[] = [
  { labelKey: 'nav.home', to: '/', activeRouteNames: ['Home', 'AgentChat'] },
  { labelKey: 'nav.imageGenerate', to: '/image/generate', activeRouteNames: ['ImageGenerate'] },
  { labelKey: 'nav.imageEdit', to: '/image/edit', activeRouteNames: ['ImageEdit'] },
  { labelKey: 'nav.video', to: '/video/generate', activeRouteNames: ['VideoGenerate', 'VideoDigitalHuman'] },
  { labelKey: 'nav.audio', to: '/audio', activeRouteNames: ['Audio'] },
  { labelKey: 'nav.translate', to: '/translate', activeRouteNames: ['Translate'] },
  { labelKey: 'nav.assets', to: '/assets', activeRouteNames: ['Assets'] },
]
