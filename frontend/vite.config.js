import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, 'src')
    }
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    proxy: {
      '/api': {
        // 本地开发前端时，将接口请求转发到远程 3DGS 后端。
        target: 'http://192.168.10.148:12524',
        changeOrigin: true
      }
    }
  }
})
