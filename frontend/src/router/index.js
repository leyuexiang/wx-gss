import { createRouter, createWebHistory } from 'vue-router'
import Generate from '../views/Generate.vue'
import Optimize from '../views/Optimize.vue'

const routes = [
  {
    path: '/',
    name: 'Generate',
    component: Generate,
    meta: { title: '模型生成' }
  },
  {
    path: '/optimize',
    name: 'Optimize',
    component: Optimize,
    meta: { title: '压缩优化' }
  },
  {
    path: '/viewer',
    name: 'Viewer',
    // 高斯渲染库体积较大，进入查看页时再加载，避免拖慢生成页面。
    component: () => import('../views/Viewer.vue'),
    meta: { title: '模型查看' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  document.title = to.meta.title ? `${to.meta.title} - 3DGS 平台` : '3DGS 模型管理平台'
  next()
})

export default router
