<template>
  <div class="optimize-container">
    <el-card class="select-card">
      <template #header>
        <div class="card-header">
          <span>选择要优化的模型</span>
        </div>
      </template>

      <el-form :model="form" label-width="120px">
        <el-form-item label="选择模型">
          <el-select
            v-model="form.modelId"
            placeholder="请选择模型"
            style="width: 100%"
            @change="handleModelChange"
            filterable
          >
            <el-option
              v-for="model in models"
              :key="model.id"
              :label="model.name"
              :value="model.id"
            >
              <span>{{ model.name }}</span>
              <span style="float: right; color: #8492a6; font-size: 13px">
                {{ formatFileSize(model.size) }}
              </span>
            </el-option>
          </el-select>
        </el-form-item>

        <el-form-item label="优化算法">
          <el-select v-model="form.algorithm" placeholder="选择优化算法">
            <el-option label="LightGaussian - 轻量化压缩" value="lightgaussian">
              <div>
                <div><strong>LightGaussian</strong></div>
                <div style="font-size: 12px; color: #909399">基于剪枝的压缩方法，压缩率高，质量损失小</div>
              </div>
            </el-option>
            <el-option label="Mini-Splatting - 快速压缩" value="mini-splatting">
              <div>
                <div><strong>Mini-Splatting</strong></div>
                <div style="font-size: 12px; color: #909399">快速压缩算法，适合实时应用</div>
              </div>
            </el-option>
            <el-option label="Compact3D - 高质量压缩" value="compact3d">
              <div>
                <div><strong>Compact3D</strong></div>
                <div style="font-size: 12px; color: #909399">保持高质量的压缩方案，压缩时间较长</div>
              </div>
            </el-option>
          </el-select>
        </el-form-item>

        <el-form-item label="压缩级别">
          <el-radio-group v-model="form.compressionLevel">
            <el-radio-button label="low">轻度压缩 (50-70%)</el-radio-button>
            <el-radio-button label="medium">中度压缩 (30-50%)</el-radio-button>
            <el-radio-button label="high">重度压缩 (10-30%)</el-radio-button>
          </el-radio-group>
        </el-form-item>

        <el-form-item label="质量优先">
          <el-switch v-model="form.qualityFirst" />
          <span style="margin-left: 10px; color: #909399; font-size: 12px">
            开启后会优先保证渲染质量，可能降低压缩率
          </span>
        </el-form-item>
      </el-form>

      <div class="button-group">
        <el-button
          type="primary"
          @click="handleOptimize"
          :loading="optimizing"
          :disabled="!form.modelId"
        >
          <el-icon v-if="!optimizing"><Cpu /></el-icon>
          开始优化
        </el-button>
        <el-button @click="handleReset" :disabled="optimizing">重置</el-button>
      </div>
    </el-card>

    <el-card class="task-card" v-if="currentTask">
      <template #header>
        <div class="card-header">
          <span>优化任务进度</span>
        </div>
      </template>

      <div class="task-info">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="任务ID">{{ currentTask.taskId }}</el-descriptions-item>
          <el-descriptions-item label="原始模型">{{ currentTask.modelName }}</el-descriptions-item>
          <el-descriptions-item label="优化算法">{{ getAlgorithmName(currentTask.algorithm) }}</el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="getStatusType(currentTask.status)">{{ getStatusText(currentTask.status) }}</el-tag>
          </el-descriptions-item>
        </el-descriptions>

        <div style="margin-top: 20px">
          <el-progress :percentage="currentTask.progress" :status="currentTask.status === 'failed' ? 'exception' : undefined" />
        </div>

        <div class="comparison" v-if="currentTask.status === 'completed' && currentTask.comparison">
          <el-divider content-position="left">优化对比</el-divider>
          <el-row :gutter="20">
            <el-col :span="12">
              <el-statistic title="原始大小" :value="formatFileSize(currentTask.comparison.originalSize)" />
            </el-col>
            <el-col :span="12">
              <el-statistic title="优化后大小" :value="formatFileSize(currentTask.comparison.optimizedSize)" />
            </el-col>
          </el-row>
          <el-row :gutter="20" style="margin-top: 20px">
            <el-col :span="12">
              <el-statistic title="压缩率">
                <template #suffix>%</template>
                <template #default>{{ currentTask.comparison.compressionRatio }}</template>
              </el-statistic>
            </el-col>
            <el-col :span="12">
              <el-statistic title="质量分数 (PSNR)">
                <template #suffix>dB</template>
                <template #default>{{ currentTask.comparison.psnr }}</template>
              </el-statistic>
            </el-col>
          </el-row>
        </div>

        <div class="task-log" v-if="currentTask.log">
          <el-divider content-position="left">执行日志</el-divider>
          <el-scrollbar height="200px">
            <pre>{{ currentTask.log }}</pre>
          </el-scrollbar>
        </div>

        <div class="action-buttons" v-if="currentTask.status === 'completed'">
          <el-button type="success" @click="viewOptimized">查看优化后的模型</el-button>
          <el-button type="primary" @click="downloadOptimized">下载优化模型</el-button>
        </div>
      </div>
    </el-card>

    <el-card class="history-card">
      <template #header>
        <div class="card-header">
          <span>优化历史</span>
          <el-button type="primary" size="small" @click="loadHistory" :icon="Refresh">刷新</el-button>
        </div>
      </template>

      <el-table :data="history" style="width: 100%" v-loading="loadingHistory">
        <el-table-column prop="modelName" label="原始模型" />
        <el-table-column prop="algorithm" label="算法" width="150">
          <template #default="scope">
            {{ getAlgorithmName(scope.row.algorithm) }}
          </template>
        </el-table-column>
        <el-table-column label="压缩率" width="100">
          <template #default="scope">
            {{ scope.row.compressionRatio }}%
          </template>
        </el-table-column>
        <el-table-column prop="createdAt" label="优化时间" width="180" />
        <el-table-column prop="status" label="状态" width="100">
          <template #default="scope">
            <el-tag :type="getStatusType(scope.row.status)">
              {{ getStatusText(scope.row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="150">
          <template #default="scope">
            <el-button size="small" @click="viewHistoryItem(scope.row)" :disabled="scope.row.status !== 'completed'">
              查看
            </el-button>
            <el-button size="small" type="danger" @click="deleteHistoryItem(scope.row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getModelList, optimizeModel as apiOptimizeModel, checkOptimizeStatus } from '@/api/3dgs'
import { Refresh } from '@element-plus/icons-vue'

const router = useRouter()
const route = useRoute()
const models = ref([])
const history = ref([])
const loadingHistory = ref(false)
const optimizing = ref(false)
const currentTask = ref(null)
let statusCheckInterval = null

const form = reactive({
  modelId: '',
  algorithm: 'lightgaussian',
  compressionLevel: 'medium',
  qualityFirst: true
})

const handleModelChange = (modelId) => {
  const model = models.value.find(m => m.id === modelId)
  if (model) {
    console.log('选择的模型:', model.name)
  }
}

const handleOptimize = async () => {
  if (!form.modelId) {
    ElMessage.warning('请先选择要优化的模型')
    return
  }

  optimizing.value = true

  try {
    const res = await apiOptimizeModel({
      modelId: form.modelId,
      algorithm: form.algorithm,
      compressionLevel: form.compressionLevel,
      qualityFirst: form.qualityFirst
    })

    ElMessage.success('优化任务已提交')
    const selectedModel = models.value.find(m => m.id === form.modelId)
    currentTask.value = {
      taskId: res.data.taskId,
      modelName: selectedModel?.name || '',
      algorithm: form.algorithm,
      status: 'processing',
      progress: 0,
      log: ''
    }

    startStatusCheck(res.data.taskId)
  } catch (error) {
    ElMessage.error(error.message || '提交优化任务失败')
  } finally {
    optimizing.value = false
  }
}

const handleReset = () => {
  form.modelId = ''
  form.algorithm = 'lightgaussian'
  form.compressionLevel = 'medium'
  form.qualityFirst = true
}

const startStatusCheck = (taskId) => {
  if (statusCheckInterval) {
    clearInterval(statusCheckInterval)
  }

  statusCheckInterval = setInterval(async () => {
    try {
      const res = await checkOptimizeStatus(taskId)
      currentTask.value = {
        ...currentTask.value,
        status: res.data.status,
        progress: res.data.progress || 0,
        log: res.data.log || '',
        comparison: res.data.comparison
      }

      if (res.data.status === 'completed' || res.data.status === 'failed') {
        clearInterval(statusCheckInterval)
        statusCheckInterval = null
        loadHistory()

        if (res.data.status === 'completed') {
          ElMessage.success('模型优化完成')
        } else {
          ElMessage.error('模型优化失败')
        }
      }
    } catch (error) {
      console.error('检查状态失败:', error)
    }
  }, 3000)
}

const loadModels = async () => {
  try {
    const res = await getModelList()
    models.value = res.data?.filter(m => m.status === 'completed') || []

    if (route.query.modelId) {
      form.modelId = route.query.modelId
    }
  } catch (error) {
    ElMessage.error('加载模型列表失败')
  }
}

const loadHistory = async () => {
  loadingHistory.value = true
  try {
    const res = await getModelList()
    history.value = res.data?.filter(m => m.optimized) || []
  } catch (error) {
    ElMessage.error('加载优化历史失败')
  } finally {
    loadingHistory.value = false
  }
}

const viewOptimized = () => {
  if (currentTask.value?.taskId) {
    router.push({ path: '/viewer', query: { optimizedId: currentTask.value.taskId } })
  }
}

const downloadOptimized = () => {
  if (currentTask.value?.taskId) {
    window.open(`/api/3dgs/download/${currentTask.value.taskId}`, '_blank')
  }
}

const viewHistoryItem = (item) => {
  router.push({ path: '/viewer', query: { optimizedId: item.id } })
}

const deleteHistoryItem = async (item) => {
  try {
    await ElMessageBox.confirm(`确定要删除该优化记录吗？`, '警告', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })

    ElMessage.success('删除成功')
    loadHistory()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
}

const getAlgorithmName = (algorithm) => {
  const map = {
    'lightgaussian': 'LightGaussian',
    'mini-splatting': 'Mini-Splatting',
    'compact3d': 'Compact3D'
  }
  return map[algorithm] || algorithm
}

const getStatusType = (status) => {
  const map = {
    'processing': 'warning',
    'completed': 'success',
    'failed': 'danger'
  }
  return map[status] || 'info'
}

const getStatusText = (status) => {
  const map = {
    'processing': '处理中',
    'completed': '已完成',
    'failed': '失败'
  }
  return map[status] || status
}

const formatFileSize = (bytes) => {
  if (!bytes) return '-'
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`
}

onMounted(() => {
  loadModels()
  loadHistory()
})

onUnmounted(() => {
  if (statusCheckInterval) {
    clearInterval(statusCheckInterval)
  }
})
</script>

<style scoped>
.optimize-container {
  max-width: 1200px;
  margin: 0 auto;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  color: var(--text-primary);
}

/* 选择卡片增强 */
.select-card {
  background: linear-gradient(135deg, rgba(123, 47, 247, 0.1) 0%, rgba(0, 217, 255, 0.1) 100%),
              rgba(26, 31, 58, 0.6) !important;
}

.button-group {
  display: flex;
  justify-content: center;
  gap: 16px;
  margin-top: 32px;
}

.button-group :deep(.el-button) {
  padding: 14px 32px;
  font-size: 15px;
  font-weight: 600;
  border-radius: 10px;
  transition: all 0.3s ease;
}

.button-group :deep(.el-button:hover) {
  transform: translateY(-2px);
}

/* 任务卡片 */
.task-card {
  margin-top: 20px;
  border: 2px solid rgba(0, 217, 255, 0.3) !important;
  animation: pulse-glow 2s ease-in-out infinite;
}

@keyframes pulse-glow {
  0%, 100% {
    box-shadow: 0 0 20px rgba(0, 217, 255, 0.2);
  }
  50% {
    box-shadow: 0 0 40px rgba(0, 217, 255, 0.4);
  }
}

.task-info {
  padding: 10px 0;
}

.task-info :deep(.el-descriptions__label) {
  color: var(--text-secondary) !important;
  font-weight: 500;
}

.task-info :deep(.el-descriptions__content) {
  color: var(--text-primary) !important;
}

.task-info :deep(.el-progress-bar__outer) {
  height: 20px !important;
  border-radius: 10px;
}

.task-info :deep(.el-progress-bar__inner) {
  border-radius: 10px;
  background: var(--accent-gradient) !important;
}

/* 对比数据 */
.comparison {
  margin-top: 24px;
  padding: 20px;
  background: rgba(0, 217, 255, 0.05);
  border: 1px solid rgba(0, 217, 255, 0.2);
  border-radius: 12px;
}

.comparison :deep(.el-statistic__head) {
  color: var(--text-secondary) !important;
  font-size: 13px;
}

.comparison :deep(.el-statistic__content) {
  color: var(--accent-primary) !important;
  font-size: 28px;
  font-weight: 700;
}

.task-log {
  margin-top: 20px;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 16px;
  box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.3);
}

.task-log pre {
  font-family: 'Courier New', monospace;
  font-size: 12px;
  line-height: 1.6;
  color: #00ff88;
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.action-buttons {
  margin-top: 24px;
  text-align: center;
  display: flex;
  justify-content: center;
  gap: 12px;
}

.action-buttons :deep(.el-button) {
  padding: 12px 28px;
  font-weight: 600;
  border-radius: 10px;
}

/* 历史记录卡片 */
.history-card {
  margin-top: 20px;
}

.history-card :deep(.el-table) {
  background: transparent !important;
}

.history-card :deep(.el-table__body tr:hover) {
  background: rgba(0, 217, 255, 0.05) !important;
}

:deep(.el-select-dropdown__item) {
  height: auto;
  padding: 12px 20px;
  line-height: 1.5;
}

:deep(.el-form-item__label) {
  color: var(--text-secondary) !important;
  font-weight: 500;
}
</style>
