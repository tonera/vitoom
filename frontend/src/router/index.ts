/**
 * Vue Router配置
 */
import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'
import { hasValidAccessToken } from '../utils/auth'
import { isCurrentUserAdmin } from '../utils/currentUser'

const routes: RouteRecordRaw[] = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('../views/Login.vue'),
    meta: { requiresAuth: false, layout: false },
  },
  {
    path: '/',
    component: () => import('../layout/main.layout.vue'),
    meta: { requiresAuth: true },
    children: [
      {
        path: '',
        name: 'Home',
        component: () => import('../views/agents/AgentIndex.vue'),
      },
      // 图片相关路由
      {
        path: 'image/generate',
        name: 'ImageGenerate',
        component: () => import('../views/image/ImageGenerate.vue'),
      },
      {
        path: 'image/edit',
        name: 'ImageEdit',
        component: () => import('../views/image/ImageEdit.vue'),
      },
      // 文字路由
      {
        path: 'text',
        name: 'Text',
        component: () => import('../views/text/Text.vue'),
      },
      // 视频相关路由
      {
        path: 'video/generate',
        name: 'VideoGenerate',
        component: () => import('../views/video/VideoGenerate.vue'),
      },
      {
        path: 'video/digital-human',
        name: 'VideoDigitalHuman',
        component: () => import('../views/video/VideoDigitalHuman.vue'),
      },
      // 音频相关路由
      {
        path: 'audio',
        name: 'Audio',
        component: () => import('../views/audio/AudioIndex.vue'),
      },
      {
        path: 'translate',
        name: 'Translate',
        component: () => import('../views/translate/TranslateIndex.vue'),
      },
     
      // 资产路由
      {
        path: 'assets',
        name: 'Assets',
        component: () => import('../views/assets/Assets.vue'),
      },
      {
        path: 'agents',
        name: 'Agents',
        component: () => import('../views/agents/AgentIndex.vue'),
      },
      {
        path: 'agents/:id',
        name: 'AgentChat',
        component: () => import('../views/agents/AgentChat.vue'),
      },
      {
        path: 'models',
        name: 'Models',
        component: () => import('../views/models/ModelList.vue'),
        meta: { requiresAdmin: true },
      },
      {
        path: 'files',
        name: 'Files',
        component: () => import('../views/files/FileList.vue'),
      },
      {
        path: 'api-keys',
        name: 'ApiKeys',
        component: () => import('../views/ApiKeys.vue'),
      },
      {
        path: 'settings',
        name: 'Settings',
        component: () => import('../views/Settings.vue'),
      },
      {
        path: 'users',
        name: 'Users',
        component: () => import('../views/users/UserList.vue'),
        meta: { requiresAdmin: true },
      },
      {
        path: 'inference',
        name: 'InferenceAdmin',
        component: () => import('../views/inference/InferenceAdmin.vue'),
        meta: { requiresAdmin: true },
      },
    ],
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: () => import('../views/NotFound.vue'),
    meta: { layout: false },
  },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

// 路由守卫：检查认证状态
router.beforeEach(async (to, _from, next) => {
  const hasToken = hasValidAccessToken()
  const requiresAuth = to.meta.requiresAuth !== false

  if (requiresAuth && !hasToken) {
    next({ name: 'Login', query: { redirect: to.fullPath } })
    return
  }

  if (to.name === 'Login' && hasToken) {
    next({ name: 'Home' })
    return
  }

  if (to.matched.length > 0) {
    const parentMeta = to.matched[0]?.meta
    if (parentMeta && parentMeta.requiresAuth && !hasToken) {
      next({ name: 'Login', query: { redirect: to.fullPath } })
      return
    }
  }

  const requiresAdmin = to.matched.some((record) => record.meta.requiresAdmin === true)
  if (requiresAdmin) {
    const isAdmin = await isCurrentUserAdmin(true)
    if (!isAdmin) {
      next({ name: 'Home' })
      return
    }
  }

  next()
})

export default router

