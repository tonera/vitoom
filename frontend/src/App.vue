<template>
  <div id="app">
    <router-view />
    <TopSnackHost />
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import TopSnackHost from './components/TopSnackHost.vue'
import { hasValidAccessToken } from './utils/auth'

const router = useRouter()

onMounted(() => {
  // 检查是否有token，如果没有且不在登录页，跳转到登录页
  if (!hasValidAccessToken() && router.currentRoute.value.name !== 'Login') {
    router.push({ name: 'Login' })
  }
})
</script>

<style scoped>
#app {
  height: 100%;
}
</style>
