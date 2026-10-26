<template>
  <div class="viewer-container">
    <el-card class="control-card">
      <template #header>
        <div class="card-header">
          <span>模型选择</span>
        </div>
      </template>

      <el-form :model="viewerForm" label-width="100px">
        <el-form-item label="选择模型">
          <el-select
            v-model="viewerForm.modelId"
            placeholder="请选择要查看的模型"
            style="width: 100%"
            @change="loadModel"
            filterable
          >
            <el-option
              v-for="model in models"
              :key="model.id"
              :label="model.name"
              :value="model.id"
            >
              <div style="display: flex; justify-content: space-between; align-items: center">
                <span>{{ model.name }}</span>
                <el-tag v-if="model.optimized" type="success" size="small">已优化</el-tag>
              </div>
            </el-option>
          </el-select>
        </el-form-item>

        <el-form-item label="对比模式" v-if="currentModel?.optimizedVersion">
          <el-switch
            v-model="viewerForm.compareMode"
            active-text="对比原始/优化"
            @change="handleCompareModeChange"
          />
        </el-form-item>
      </el-form>

      <div class="model-info" v-if="currentModel">
        <el-divider content-position="left">模型信息</el-divider>
        <el-descriptions :column="1" border size="small">
          <el-descriptions-item label="模型名称">{{ currentModel.name }}</el-descriptions-item>
          <el-descriptions-item label="文件大小">{{ formatFileSize(currentModel.size) }}</el-descriptions-item>
          <el-descriptions-item label="创建时间">{{ currentModel.createdAt }}</el-descriptions-item>
          <el-descriptions-item label="点云数量">{{ currentModel.pointCount || '-' }}</el-descriptions-item>
          <el-descriptions-item label="训练迭代" v-if="currentModel.iterations">
            {{ currentModel.iterations }}
          </el-descriptions-item>
          <el-descriptions-item label="分辨率" v-if="currentModel.resolution">
            {{ currentModel.resolution }}
          </el-descriptions-item>
        </el-descriptions>
      </div>

      <div class="viewer-controls">
        <el-divider content-position="left">查看器控制</el-divider>
        <el-space wrap>
          <el-button size="small" @click="resetCamera">
            <el-icon><Refresh /></el-icon>
            重置视角
          </el-button>
          <el-button v-if="currentModel?.trainingViews?.length" size="small" @click="nextTrainingView">
            训练视角 {{ trainingViewIndex + 1 }}/{{ currentModel.trainingViews.length }}
          </el-button>
          <el-button size="small" @click="toggleAutoRotate">
            <el-icon><VideoPlay /></el-icon>
            {{ autoRotate ? '停止' : '自动' }}旋转
          </el-button>
          <el-button size="small" @click="captureScreenshot">
            <el-icon><Camera /></el-icon>
            截图
          </el-button>
          <el-button size="small" @click="toggleFullscreen">
            <el-icon><FullScreen /></el-icon>
            全屏
          </el-button>
        </el-space>

        <div class="quality-control" style="margin-top: 15px">
          <el-text size="small">渲染质量:</el-text>
          <el-slider
            v-model="renderQuality"
            :min="1"
            :max="100"
            @change="handleQualityChange"
            style="width: 200px; margin-left: 10px"
          />
        </div>
      </div>
    </el-card>

    <el-card class="viewer-card" v-loading="loading">
      <div class="viewer-wrapper" ref="viewerWrapperRef">
        <div v-if="!viewerForm.compareMode" class="single-viewer">
          <div ref="viewerHostRef" class="viewer-canvas"></div>
          <div class="viewer-overlay" v-if="!currentModel || viewerError">
            <el-empty :description="viewerError || '请选择要查看的模型'" />
          </div>
          <div class="viewer-stats" v-if="stats">
            <div>Points: {{ stats.points }}</div>
          </div>
        </div>

        <div v-else class="compare-viewer">
          <div class="viewer-panel">
            <div class="panel-label">原始模型</div>
            <canvas ref="canvasOriginalRef" class="viewer-canvas"></canvas>
          </div>
          <div class="viewer-divider"></div>
          <div class="viewer-panel">
            <div class="panel-label">优化后模型</div>
            <canvas ref="canvasOptimizedRef" class="viewer-canvas"></canvas>
          </div>
        </div>

        <div class="viewer-help">
          <el-tag size="small">鼠标左键: 旋转</el-tag>
          <el-tag size="small">鼠标右键: 平移</el-tag>
          <el-tag size="small">滚轮: 缩放</el-tag>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted, nextTick, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import * as THREE from 'three'
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d'
import { getModelList, getModelInfo } from '@/api/3dgs'

const route = useRoute()
const viewerWrapperRef = ref(null)
const viewerHostRef = ref(null)
const canvasOriginalRef = ref(null)
const canvasOptimizedRef = ref(null)

const models = ref([])
const currentModel = ref(null)
const viewerError = ref('')
const loading = ref(false)
const autoRotate = ref(false)
const renderQuality = ref(100)
const trainingViewIndex = ref(-1)
const stats = ref(null)

const viewerForm = reactive({ modelId: '', compareMode: false })
let viewer = null
let renderer = null
let resizeObserver = null
let loadVersion = 0
const cameraHome = new THREE.Vector3(0, 0, 3)
const targetHome = new THREE.Vector3()
const modelCenter = new THREE.Vector3()
const cameraHomeUp = new THREE.Vector3(0, 1, 0)
let cameraHomeFov = 50

const disposeViewer = async () => {
  const oldViewer = viewer
  const oldRenderer = renderer
  viewer = null
  renderer = null
  resizeObserver?.disconnect()
  resizeObserver = null
  if (oldViewer) {
    oldViewer.stop()
    await oldViewer.dispose()
  }
  oldRenderer?.dispose()
  oldRenderer?.domElement.remove()
  stats.value = null
}

const loadModels = async () => {
  try {
    const res = await getModelList()
    models.value = res.data || []
    const modelId = route.query.modelId || route.query.optimizedId
    if (modelId) {
      viewerForm.modelId = modelId
      await loadModel(modelId)
    }
  } catch (error) {
    ElMessage.error('加载模型列表失败')
  }
}

const frameModel = () => {
  const mesh = viewer.splatMesh
  const count = mesh.getSplatCount()
  if (!count) throw new Error('模型没有可显示的高斯点')

  // 采样分位数可忽略少量离群点，同时避免遍历全部高斯造成页面卡顿。
  const axes = [[], [], []]
  const center = new THREE.Vector3()
  const step = Math.max(1, Math.floor(count / 4000))
  for (let index = 0; index < count; index += step) {
    mesh.getSplatCenter(index, center, true)
    if (Number.isFinite(center.x) && Number.isFinite(center.y) && Number.isFinite(center.z)) {
      axes[0].push(center.x)
      axes[1].push(center.y)
      axes[2].push(center.z)
    }
  }
  if (!axes[0].length) throw new Error('模型坐标无效')

  const spans = axes.map((axis, index) => {
    axis.sort((a, b) => a - b)
    const low = axis[Math.floor((axis.length - 1) * 0.05)]
    const high = axis[Math.floor((axis.length - 1) * 0.95)]
    targetHome.setComponent(index, (low + high) / 2)
    return high - low
  })
  modelCenter.copy(targetHome)
  const distance = Math.max(1.5, Math.max(...spans) / Math.tan(THREE.MathUtils.degToRad(viewer.camera.fov / 2)) * 1.3)
  cameraHome.copy(targetHome).add(new THREE.Vector3(0, 0, distance))
  cameraHomeUp.set(0, 1, 0)
  cameraHomeFov = viewer.camera.fov
  trainingViewIndex.value = -1
  resetCamera()
}

const applyTrainingView = (index) => {
  const view = currentModel.value?.trainingViews?.[index]
  if (!viewer || !view) return
  // 按重建时的相机方向查看；自由视角可能落在少视角模型没有覆盖的背面。
  cameraHome.fromArray(view.position)
  cameraHomeUp.fromArray(view.up).normalize()
  const forward = new THREE.Vector3().fromArray(view.forward).normalize()
  const distance = Math.max(1, cameraHome.distanceTo(modelCenter))
  targetHome.copy(cameraHome).addScaledVector(forward, distance)
  cameraHomeFov = view.fov || viewer.camera.fov
  trainingViewIndex.value = index
  resetCamera()
}

const nextTrainingView = () => {
  const count = currentModel.value?.trainingViews?.length || 0
  if (!count) return
  autoRotate.value = false
  viewer.controls.autoRotate = false
  applyTrainingView((trainingViewIndex.value + 1) % count)
}

const initViewer = async (modelId) => {
  const host = viewerHostRef.value
  if (!host) throw new Error('查看区域尚未就绪')

  // 使用外部渲染器，让模型画布留在当前组件内，并在切换模型时完整释放资源。
  renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true })
  renderer.domElement.style.width = '100%'
  renderer.domElement.style.height = '100%'
  host.appendChild(renderer.domElement)
  viewer = new GaussianSplats3D.Viewer({
    rootElement: host,
    renderer,
    cameraUp: [0, 1, 0],
    initialCameraPosition: [0, 0, 3],
    initialCameraLookAt: [0, 0, 0],
    sharedMemoryForWorkers: false,
    gpuAcceleratedSort: false,
    sphericalHarmonicsDegree: 0
  })

  const resize = () => {
    const width = Math.max(host.clientWidth, 1)
    const height = Math.max(host.clientHeight, 1)
    renderer.setPixelRatio(Math.max(0.25, Math.min(window.devicePixelRatio || 1, 2) * renderQuality.value / 100))
    renderer.setSize(width, height, false)
    if (viewer?.camera?.isPerspectiveCamera) {
      viewer.camera.aspect = width / height
      viewer.camera.updateProjectionMatrix()
    }
    viewer?.forceRenderNextFrame()
  }
  resizeObserver = new ResizeObserver(resize)
  resizeObserver.observe(host)
  resize()

  await viewer.addSplatScene(`/api/3dgs/model/${encodeURIComponent(modelId)}/point_cloud.splat`, {
    format: GaussianSplats3D.SceneFormat.Splat,
    splatAlphaRemovalThreshold: 5,
    showLoadingUI: false
  })
  frameModel()
  const trainingViews = currentModel.value?.trainingViews || []
  if (trainingViews.length) applyTrainingView(Math.floor(trainingViews.length / 2))
  viewer.controls.autoRotate = autoRotate.value
  viewer.start()
  stats.value = { points: viewer.splatMesh.getSplatCount().toLocaleString() }
}

const loadModel = async (modelId) => {
  const version = ++loadVersion
  loading.value = true
  viewerError.value = ''
  currentModel.value = null
  try {
    await disposeViewer()
    if (!modelId) return
    const res = await getModelInfo(modelId)
    if (version !== loadVersion) return
    currentModel.value = res.data
    await nextTick()
    await initViewer(modelId)
    if (version !== loadVersion) return
    ElMessage.success('模型加载成功')
  } catch (error) {
    if (version !== loadVersion) return
    viewerError.value = '模型文件加载失败，请检查训练结果'
    ElMessage.error(viewerError.value)
    console.error(error)
    await disposeViewer()
  } finally {
    if (version === loadVersion) loading.value = false
  }
}

const handleCompareModeChange = async (enabled) => {
  // 当前训练结果没有对比版本；保留旧入口时避免切换到空画布。
  if (enabled) {
    viewerForm.compareMode = false
    ElMessage.warning('当前模型没有可查看的优化版本')
  }
}

const resetCamera = () => {
  if (!viewer) return
  viewer.camera.position.copy(cameraHome)
  viewer.camera.up.copy(cameraHomeUp)
  viewer.camera.fov = cameraHomeFov
  viewer.camera.updateProjectionMatrix()
  viewer.controls.target.copy(targetHome)
  viewer.controls.update()
  viewer.forceRenderNextFrame()
}

const toggleAutoRotate = () => {
  if (!viewer) return
  autoRotate.value = !autoRotate.value
  viewer.controls.autoRotate = autoRotate.value
}

const captureScreenshot = () => {
  if (!viewer || !renderer) return
  viewer.render()
  const link = document.createElement('a')
  link.download = `${currentModel.value?.name || 'model'}_screenshot.png`
  link.href = renderer.domElement.toDataURL('image/png')
  link.click()
}

const toggleFullscreen = () => {
  if (!document.fullscreenElement) {
    viewerWrapperRef.value?.requestFullscreen()
  } else {
    document.exitFullscreen()
  }
}

const handleQualityChange = () => {
  if (!renderer || !viewerHostRef.value) return
  renderer.setPixelRatio(Math.max(0.25, Math.min(window.devicePixelRatio || 1, 2) * renderQuality.value / 100))
  renderer.setSize(viewerHostRef.value.clientWidth, viewerHostRef.value.clientHeight, false)
  viewer?.forceRenderNextFrame()
}

const formatFileSize = (bytes) => {
  if (!bytes) return '-'
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`
}

onMounted(loadModels)
watch(() => route.query.modelId, (modelId) => {
  if (modelId && modelId !== viewerForm.modelId) {
    viewerForm.modelId = modelId
    loadModel(modelId)
  }
})
onUnmounted(() => {
  loadVersion++
  disposeViewer().catch(console.error)
})
</script>

<style scoped>
.viewer-container {
  height: calc(100vh - 160px);
  display: flex;
  gap: 20px;
}

.control-card {
  width: 380px;
  flex-shrink: 0;
  overflow-y: auto;
  background: linear-gradient(135deg, rgba(26, 31, 58, 0.8) 0%, rgba(18, 22, 41, 0.8) 100%) !important;
  backdrop-filter: blur(20px);
}

.control-card::-webkit-scrollbar {
  width: 6px;
}

.control-card::-webkit-scrollbar-thumb {
  background: rgba(0, 217, 255, 0.3);
  border-radius: 3px;
}

.viewer-card {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: linear-gradient(135deg, rgba(18, 22, 41, 0.6) 0%, rgba(10, 14, 39, 0.9) 100%) !important;
  border: 2px solid var(--border-color) !important;
  position: relative;
  overflow: hidden;
}

.viewer-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background:
    repeating-linear-gradient(
      0deg,
      rgba(0, 217, 255, 0.03) 0px,
      transparent 1px,
      transparent 2px,
      rgba(0, 217, 255, 0.03) 3px
    );
  pointer-events: none;
  animation: scanline 8s linear infinite;
}

@keyframes scanline {
  0% { transform: translateY(0); }
  100% { transform: translateY(50px); }
}

.card-header {
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 10px;
  color: var(--text-primary);
}

.card-header i {
  color: var(--accent-primary);
}

.model-info {
  margin-top: 20px;
}

.model-info :deep(.el-descriptions__body),
.model-info :deep(.el-descriptions__cell) {
  background: rgba(16, 22, 42, 0.9) !important;
}

.model-info :deep(.el-descriptions__table) {
  table-layout: fixed;
}

.model-info :deep(.el-descriptions__label) {
  color: var(--text-secondary) !important;
  font-weight: 500;
}

.model-info :deep(.el-descriptions__content) {
  color: var(--text-primary) !important;
  overflow-wrap: anywhere;
}

.viewer-controls {
  margin-top: 24px;
}

.viewer-controls :deep(.el-space) {
  width: 100%;
  flex-wrap: wrap;
}

.viewer-controls :deep(.el-button) {
  flex: 1;
  min-width: 120px;
  border-radius: 8px;
  font-weight: 500;
  transition: all 0.3s ease;
}

.viewer-controls :deep(.el-button:hover) {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 217, 255, 0.3);
}

.quality-control {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 8px;
  margin-top: 12px;
}

.quality-control :deep(.el-text) {
  color: var(--text-secondary) !important;
  font-weight: 500;
  white-space: nowrap;
}

.quality-control :deep(.el-slider) {
  flex: 1;
}

/* 查看器区域 */
.viewer-wrapper {
  position: relative;
  width: 100%;
  height: 100%;
  background: radial-gradient(circle at center, #0f1420 0%, #0a0e27 100%);
  border-radius: 12px;
  overflow: hidden;
}

.single-viewer,
.compare-viewer {
  width: 100%;
  height: 100%;
  position: relative;
}

.compare-viewer {
  display: flex;
}

.viewer-panel {
  flex: 1;
  position: relative;
  overflow: hidden;
}

.panel-label {
  position: absolute;
  top: 20px;
  left: 20px;
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(10px);
  color: white;
  padding: 10px 20px;
  border-radius: 8px;
  font-size: 14px;
  font-weight: 600;
  z-index: 10;
  border: 1px solid rgba(0, 217, 255, 0.3);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
}

.viewer-divider {
  width: 3px;
  background: var(--accent-gradient);
  box-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
  position: relative;
}

.viewer-divider::before {
  content: '⇄';
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  background: var(--bg-primary);
  color: var(--accent-primary);
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  border: 2px solid var(--accent-primary);
  box-shadow: 0 0 15px rgba(0, 217, 255, 0.5);
}

.viewer-canvas {
  width: 100%;
  height: 100%;
  display: block;
}

.viewer-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: radial-gradient(circle at center, rgba(15, 20, 32, 0.9) 0%, rgba(10, 14, 39, 1) 100%);
}

.viewer-stats {
  position: absolute;
  top: 20px;
  right: 20px;
  background: rgba(0, 0, 0, 0.8);
  backdrop-filter: blur(10px);
  color: #00ff88;
  padding: 16px 20px;
  border-radius: 8px;
  font-family: 'Courier New', monospace;
  font-size: 13px;
  line-height: 1.8;
  border: 1px solid rgba(0, 255, 136, 0.3);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
  min-width: 120px;
}

.viewer-stats div {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}

.viewer-stats div::before {
  content: '▸';
  color: var(--accent-primary);
}

.viewer-help {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  justify-content: center;
}

.viewer-help :deep(.el-tag) {
  background: rgba(0, 0, 0, 0.8) !important;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(0, 217, 255, 0.3) !important;
  color: var(--text-primary) !important;
  padding: 8px 16px;
  font-size: 12px;
  font-weight: 500;
}

:deep(.el-card__body) {
  height: calc(100% - 57px);
  padding: 0;
}

.viewer-card :deep(.el-card__body) {
  padding: 20px;
}

:deep(.el-form-item__label) {
  color: var(--text-secondary) !important;
  font-weight: 500;
}

/* 响应式 */
@media (max-width: 1200px) {
  .viewer-container {
    flex-direction: column;
    height: auto;
  }

  .control-card {
    width: 100%;
    max-height: 400px;
  }

  .viewer-card {
    min-height: 600px;
  }
}
</style>
