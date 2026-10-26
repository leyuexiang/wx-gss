<template>
  <div id="app">
    <div class="tech-background">
      <div class="grid-overlay"></div>
      <div class="glow-orb orb-1"></div>
      <div class="glow-orb orb-2"></div>
    </div>

    <header class="tech-header">
      <div class="header-content">
        <div class="logo-section">
          <div class="logo-icon">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
              <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
          <div class="logo-text">
            <h1>3DGS Platform</h1>
            <span class="version">v1.0</span>
          </div>
        </div>

        <nav class="tech-nav">
          <router-link to="/" class="nav-item" :class="{ active: currentRoute === '/' }">
            <div class="nav-icon">
              <el-icon><Upload /></el-icon>
            </div>
            <span class="nav-label">模型生成</span>
            <div class="nav-indicator"></div>
          </router-link>

          <router-link to="/optimize" class="nav-item" :class="{ active: currentRoute === '/optimize' }">
            <div class="nav-icon">
              <el-icon><Operation /></el-icon>
            </div>
            <span class="nav-label">压缩优化</span>
            <div class="nav-indicator"></div>
          </router-link>

          <router-link to="/viewer" class="nav-item" :class="{ active: currentRoute === '/viewer' }">
            <div class="nav-icon">
              <el-icon><View /></el-icon>
            </div>
            <span class="nav-label">模型查看</span>
            <div class="nav-indicator"></div>
          </router-link>
        </nav>

        <div class="header-actions">
          <div class="status-indicator">
            <span class="status-dot"></span>
            <span class="status-text">系统运行中</span>
          </div>
        </div>
      </div>
    </header>

    <main class="tech-main">
      <router-view v-slot="{ Component }">
        <transition name="fade-slide" mode="out-in">
          <component :is="Component" />
        </transition>
      </router-view>
    </main>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const currentRoute = computed(() => route.path)
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

:root {
  --bg-primary: #0a0e27;
  --bg-secondary: #121629;
  --bg-tertiary: #1a1f3a;

  --accent-primary: #00d9ff;
  --accent-secondary: #7b2ff7;
  --accent-gradient: linear-gradient(135deg, #00d9ff 0%, #7b2ff7 100%);

  --text-primary: #ffffff;
  --text-secondary: #a0aec0;
  --text-muted: #4a5568;

  --border-color: rgba(255, 255, 255, 0.1);
  --glass-bg: rgba(255, 255, 255, 0.05);
  --glass-border: rgba(255, 255, 255, 0.1);

  --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
  --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
  --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
  --shadow-glow: 0 0 20px rgba(0, 217, 255, 0.3);
}

html, body, #app {
  height: 100%;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  overflow-x: hidden;
}

/* 科技背景 */
.tech-background {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
  overflow: hidden;
  pointer-events: none;
}

.grid-overlay {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background-image:
    linear-gradient(rgba(0, 217, 255, 0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 217, 255, 0.03) 1px, transparent 1px);
  background-size: 50px 50px;
  animation: gridMove 20s linear infinite;
}

@keyframes gridMove {
  0% { transform: translate(0, 0); }
  100% { transform: translate(50px, 50px); }
}

.glow-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  opacity: 0.15;
  animation: float 8s ease-in-out infinite;
}

.orb-1 {
  width: 400px;
  height: 400px;
  background: var(--accent-primary);
  top: -200px;
  right: -200px;
  animation-delay: 0s;
}

.orb-2 {
  width: 500px;
  height: 500px;
  background: var(--accent-secondary);
  bottom: -250px;
  left: -250px;
  animation-delay: 4s;
}

@keyframes float {
  0%, 100% { transform: translate(0, 0) scale(1); }
  50% { transform: translate(30px, -30px) scale(1.1); }
}

/* 顶部导航 */
.tech-header {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 80px;
  background: rgba(10, 14, 39, 0.8);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border-color);
  z-index: 1000;
  box-shadow: var(--shadow-md);
}

.header-content {
  max-width: 1920px;
  height: 100%;
  margin: 0 auto;
  padding: 0 40px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.logo-section {
  display: flex;
  align-items: center;
  gap: 16px;
}

.logo-icon {
  width: 48px;
  height: 48px;
  background: var(--accent-gradient);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: var(--shadow-glow);
  animation: pulse 2s ease-in-out infinite;
}

.logo-icon svg {
  width: 28px;
  height: 28px;
  color: white;
}

@keyframes pulse {
  0%, 100% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.05); opacity: 0.9; }
}

.logo-text h1 {
  font-size: 24px;
  font-weight: 700;
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 2px;
}

.version {
  font-size: 11px;
  color: var(--text-muted);
  font-weight: 500;
  letter-spacing: 1px;
}

.tech-nav {
  display: flex;
  gap: 8px;
  flex: 1;
  justify-content: center;
  max-width: 600px;
}

.nav-item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 24px;
  border-radius: 12px;
  text-decoration: none;
  color: var(--text-secondary);
  background: transparent;
  transition: all 0.3s ease;
  overflow: hidden;
}

.nav-item::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: var(--glass-bg);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.nav-item:hover::before {
  opacity: 1;
}

.nav-item:hover {
  color: var(--accent-primary);
  transform: translateY(-2px);
}

.nav-item.active {
  color: var(--text-primary);
  background: var(--glass-bg);
  border: 1px solid var(--glass-border);
}

.nav-item.active .nav-indicator {
  width: 100%;
  opacity: 1;
}

.nav-icon {
  font-size: 20px;
  display: flex;
  align-items: center;
  z-index: 1;
}

.nav-label {
  font-size: 14px;
  font-weight: 500;
  white-space: nowrap;
  z-index: 1;
}

.nav-indicator {
  position: absolute;
  bottom: 0;
  left: 50%;
  transform: translateX(-50%);
  width: 0;
  height: 2px;
  background: var(--accent-gradient);
  opacity: 0;
  transition: all 0.3s ease;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 20px;
}

.status-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  background: var(--glass-bg);
  border-radius: 20px;
  border: 1px solid var(--glass-border);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #00ff88;
  box-shadow: 0 0 10px #00ff88;
  animation: blink 2s ease-in-out infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

.status-text {
  font-size: 12px;
  color: var(--text-secondary);
  font-weight: 500;
}

/* 主内容区 */
.tech-main {
  position: relative;
  margin-top: 80px;
  min-height: calc(100vh - 80px);
  padding: 40px 40px 60px;
  z-index: 1;
}

/* 页面过渡动画 */
.fade-slide-enter-active,
.fade-slide-leave-active {
  transition: all 0.3s ease;
}

.fade-slide-enter-from {
  opacity: 0;
  transform: translateX(20px);
}

.fade-slide-leave-to {
  opacity: 0;
  transform: translateX(-20px);
}

/* Element Plus 样式覆盖 */
.el-card {
  background: rgba(26, 31, 58, 0.6) !important;
  backdrop-filter: blur(20px);
  border: 1px solid var(--border-color) !important;
  border-radius: 16px !important;
  box-shadow: var(--shadow-md) !important;
  transition: all 0.3s ease;
}

.el-card:hover {
  border-color: rgba(0, 217, 255, 0.3) !important;
  box-shadow: var(--shadow-lg), 0 0 30px rgba(0, 217, 255, 0.1) !important;
  transform: translateY(-2px);
}

.el-card__header {
  background: rgba(255, 255, 255, 0.03) !important;
  border-bottom: 1px solid var(--border-color) !important;
  padding: 20px !important;
}

.el-card__body {
  padding: 24px !important;
}

.el-button--primary {
  background: var(--accent-gradient) !important;
  border: none !important;
  box-shadow: var(--shadow-glow) !important;
  transition: all 0.3s ease !important;
}

.el-button--primary:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 0 30px rgba(0, 217, 255, 0.5) !important;
}

.el-button--primary:active {
  transform: translateY(0) !important;
}

.el-input__wrapper {
  background: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid var(--border-color) !important;
  box-shadow: none !important;
  transition: all 0.3s ease !important;
}

.el-input__wrapper:hover {
  border-color: var(--accent-primary) !important;
}

.el-input__wrapper.is-focus {
  border-color: var(--accent-primary) !important;
  box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.1) !important;
}

.el-input__inner {
  color: var(--text-primary) !important;
}

.el-select-dropdown {
  background: var(--bg-tertiary) !important;
  border: 1px solid var(--border-color) !important;
  box-shadow: var(--shadow-lg) !important;
}

.el-select-dropdown__item {
  color: var(--text-secondary) !important;
}

.el-select-dropdown__item:hover {
  background: rgba(0, 217, 255, 0.1) !important;
  color: var(--text-primary) !important;
}

.el-select-dropdown__item.selected {
  color: var(--accent-primary) !important;
  background: rgba(0, 217, 255, 0.15) !important;
}

.el-progress__text {
  color: var(--text-primary) !important;
}

.el-progress-bar__outer {
  background: rgba(255, 255, 255, 0.05) !important;
}

.el-progress-bar__inner {
  background: var(--accent-gradient) !important;
}

.el-table {
  background: transparent !important;
  color: var(--text-primary) !important;
}

.el-table th.el-table__cell {
  background: rgba(255, 255, 255, 0.03) !important;
  color: var(--text-secondary) !important;
  border-color: var(--border-color) !important;
}

.el-table tr {
  background: transparent !important;
}

.el-table td.el-table__cell {
  border-color: var(--border-color) !important;
}

.el-table tbody tr:hover > td {
  background: rgba(0, 217, 255, 0.05) !important;
}

.el-tag {
  border: 1px solid var(--border-color) !important;
  background: rgba(255, 255, 255, 0.05) !important;
}

.el-tag--success {
  background: rgba(0, 255, 136, 0.1) !important;
  border-color: rgba(0, 255, 136, 0.3) !important;
  color: #00ff88 !important;
}

.el-tag--warning {
  background: rgba(255, 193, 7, 0.1) !important;
  border-color: rgba(255, 193, 7, 0.3) !important;
  color: #ffc107 !important;
}

.el-tag--danger {
  background: rgba(255, 77, 79, 0.1) !important;
  border-color: rgba(255, 77, 79, 0.3) !important;
  color: #ff4d4f !important;
}

.el-tag--info {
  background: rgba(0, 217, 255, 0.1) !important;
  border-color: rgba(0, 217, 255, 0.3) !important;
  color: var(--accent-primary) !important;
}

.el-upload-dragger {
  background: rgba(255, 255, 255, 0.03) !important;
  border: 2px dashed var(--border-color) !important;
  border-radius: 12px !important;
  transition: all 0.3s ease !important;
}

.el-upload-dragger:hover {
  border-color: var(--accent-primary) !important;
  background: rgba(0, 217, 255, 0.05) !important;
}

.el-tabs__nav-wrap::after {
  background: var(--border-color) !important;
}

.el-tabs__item {
  color: var(--text-secondary) !important;
}

.el-tabs__item:hover {
  color: var(--accent-primary) !important;
}

.el-tabs__item.is-active {
  color: var(--accent-primary) !important;
}

.el-tabs__active-bar {
  background: var(--accent-gradient) !important;
}

.el-slider__runway {
  background: rgba(255, 255, 255, 0.1) !important;
}

.el-slider__bar {
  background: var(--accent-gradient) !important;
}

.el-slider__button {
  border-color: var(--accent-primary) !important;
  background: var(--accent-primary) !important;
}

.el-radio-button__inner {
  background: rgba(255, 255, 255, 0.03) !important;
  border-color: var(--border-color) !important;
  color: var(--text-secondary) !important;
}

.el-radio-button__original-radio:checked + .el-radio-button__inner {
  background: var(--accent-gradient) !important;
  border-color: var(--accent-primary) !important;
  color: white !important;
  box-shadow: var(--shadow-glow) !important;
}

.el-descriptions {
  --el-descriptions-item-bordered-label-background: rgba(255, 255, 255, 0.03) !important;
}

.el-descriptions__label,
.el-descriptions__content {
  color: var(--text-primary) !important;
}

.el-descriptions__cell {
  border-color: var(--border-color) !important;
}

.el-alert {
  background: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid var(--border-color) !important;
}

.el-scrollbar__thumb {
  background: rgba(255, 255, 255, 0.2) !important;
}

/* 响应式 */
@media (max-width: 1200px) {
  .header-content {
    padding: 0 20px;
  }

  .tech-main {
    padding: 30px 20px;
  }
}

@media (max-width: 768px) {
  .logo-text h1 {
    font-size: 18px;
  }

  .nav-label {
    display: none;
  }

  .status-text {
    display: none;
  }
}
</style>
