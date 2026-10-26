<template>
  <div class="generate-container">
    <el-card class="upload-card">
      <template #header>
        <div class="card-header">
          <span>上传图像数据生成 3DGS 模型</span>
        </div>
      </template>

      <el-upload
        ref="uploadRef"
        class="upload-zone"
        drag
        :auto-upload="false"
        :on-change="handleFileChange"
        :limit="1"
        accept=".zip"
        :file-list="fileList"
      >
        <el-icon class="upload-icon"><UploadFilled /></el-icon>
        <div class="upload-text">将 ZIP 文件拖到此处，或<em>点击上传</em></div>
        <template #tip>
          <div class="upload-tip">
            只能上传 zip 文件，文件内应包含图像序列
          </div>
        </template>
      </el-upload>

      <div class="config-section" v-if="fileList.length > 0">
        <el-divider content-position="left">生成配置</el-divider>
        <el-form :model="form" label-width="140px">
          <el-form-item label="模型名称">
            <el-input v-model="form.modelName" placeholder="请输入模型名称" />
          </el-form-item>
          <el-form-item label="训练迭代次数">
            <el-input-number v-model="form.iterations" :min="1000" :max="30000" :step="1000" />
          </el-form-item>
          <el-form-item label="图像降采样">
            <el-radio-group v-model="form.resolution">
              <el-radio :label="1">原始分辨率</el-radio>
              <el-radio :label="2">1/2 分辨率</el-radio>
              <el-radio :label="4">1/4 分辨率</el-radio>
            </el-radio-group>
          </el-form-item>
        </el-form>

        <el-divider content-position="left">压缩优化（可选）</el-divider>
        <el-form :model="form" label-width="140px">
          <el-form-item label="生成后自动优化">
            <el-switch v-model="form.autoOptimize" @change="handleAutoOptimizeChange" />
            <span style="margin-left: 10px; color: #909399; font-size: 12px">
              开启后会在模型生成完成后自动进行压缩优化
            </span>
          </el-form-item>

          <template v-if="form.autoOptimize">
            <el-form-item label="优化算法">
              <el-select v-model="form.algorithm" placeholder="选择优化算法" style="width: 100%">
                <el-option value="lightgaussian" label="LightGaussian - 轻量化压缩">
                  <div>
                    <div><strong>LightGaussian</strong></div>
                    <div style="font-size: 12px; color: #909399">基于剪枝的压缩方法，压缩率高，质量损失小</div>
                  </div>
                </el-option>
                <el-option value="mini-splatting" label="Mini-Splatting - 快速压缩">
                  <div>
                    <div><strong>Mini-Splatting</strong></div>
                    <div style="font-size: 12px; color: #909399">快速压缩算法，适合实时应用</div>
                  </div>
                </el-option>
                <el-option value="compact3d" label="Compact3D - 高质量压缩">
                  <div>
                    <div><strong>Compact3D</strong></div>
                    <div style="font-size: 12px; color: #909399">保持高质量的压缩方案，压缩时间较长</div>
                  </div>
                </el-option>
              </el-select>
            </el-form-item>

            <el-form-item label="压缩级别">
              <el-radio-group v-model="form.compressionLevel">
                <el-radio-button label="low">轻度 (50-70%)</el-radio-button>
                <el-radio-button label="medium">中度 (30-50%)</el-radio-button>
                <el-radio-button label="high">重度 (10-30%)</el-radio-button>
              </el-radio-group>
            </el-form-item>

            <el-form-item label="质量优先">
              <el-switch v-model="form.qualityFirst" />
              <span style="margin-left: 10px; color: #909399; font-size: 12px">
                开启后会优先保证渲染质量，可能降低压缩率
              </span>
            </el-form-item>
          </template>
        </el-form>
      </div>

      <div class="button-group">
        <el-button type="primary" @click="handleGenerate" :loading="generating" :disabled="fileList.length === 0">
          <el-icon v-if="!generating"><VideoPlay /></el-icon>
          开始生成
        </el-button>
        <el-button @click="handleClear" :disabled="generating">清空</el-button>
      </div>
    </el-card>

    <el-card class="task-card" v-if="currentTask">
      <template #header>
        <div class="card-header">
          <span>生成任务进度</span>
        </div>
      </template>

      <div class="task-info">
        <el-descriptions :column="2" border>
          <el-descriptions-item label="任务ID">{{ currentTask.taskId }}</el-descriptions-item>
          <el-descriptions-item label="模型名称">{{ currentTask.modelName }}</el-descriptions-item>
          <el-descriptions-item label="状态">
            <el-tag :type="getStatusType(currentTask.status)">{{ getStatusText(currentTask.status) }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="进度">
            <el-progress :percentage="currentTask.progress" :status="currentTask.status === 'failed' ? 'exception' : undefined" />
          </el-descriptions-item>
        </el-descriptions>

        <div class="task-log" v-if="currentTask.log">
          <el-divider content-position="left">执行日志</el-divider>
          <el-scrollbar height="200px">
            <pre>{{ currentTask.log }}</pre>
          </el-scrollbar>
        </div>
      </div>
    </el-card>

    <el-card class="models-card">
      <template #header>
        <div class="card-header">
          <span>已生成的模型</span>
          <el-button type="primary" size="small" @click="loadModels" :icon="Refresh">刷新</el-button>
        </div>
      </template>

      <el-table :data="models" style="width: 100%" v-loading="loadingModels">
        <el-table-column prop="name" label="模型名称" />
        <el-table-column prop="createdAt" label="创建时间" width="180" />
        <el-table-column prop="size" label="文件大小" width="120">
          <template #default="scope">
            {{ formatFileSize(scope.row.size) }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="100">
          <template #default="scope">
            <el-tag :type="scope.row.status === 'completed' ? 'success' : 'info'">
              {{ scope.row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200">
          <template #default="scope">
            <el-button size="small" @click="viewModel(scope.row)">查看</el-button>
            <el-button size="small" type="warning" @click="optimizeModel(scope.row)">优化</el-button>
            <el-button size="small" type="danger" @click="deleteModel(scope.row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { generate3DGS, checkGenerateStatus, getModelList, deleteModel as apiDeleteModel } from '@/api/3dgs'
import { Refresh } from '@element-plus/icons-vue'

const router = useRouter()
const uploadRef = ref(null)
const fileList = ref([])
const generating = ref(false)
const loadingModels = ref(false)
const currentTask = ref(null)
const models = ref([])
let statusCheckInterval = null

const form = reactive({
  modelName: '',
  iterations: 7000,
  resolution: 1,
  autoOptimize: false,
  algorithm: 'lightgaussian',
  compressionLevel: 'medium',
  qualityFirst: true
})

const handleAutoOptimizeChange = (value) => {
  if (value) {
    ElMessage.info('已启用自动优化，模型生成完成后将自动进行压缩')
  }
}

const handleFileChange = (file) => {
  if (!file.name.endsWith('.zip')) {
    ElMessage.error('只能上传 ZIP 格式的文件')
    fileList.value = []
    return
  }
  fileList.value = [file]
  if (!form.modelName) {
    form.modelName = file.name.replace('.zip', '')
  }
}

const handleClear = () => {
  fileList.value = []
  uploadRef.value?.clearFiles()
  form.modelName = ''
}

const handleGenerate = async () => {
  if (fileList.value.length === 0) {
    ElMessage.warning('请先上传 ZIP 文件')
    return
  }

  if (!form.modelName.trim()) {
    ElMessage.warning('请输入模型名称')
    return
  }

  generating.value = true

  try {
    const formData = new FormData()
    formData.append('file', fileList.value[0].raw)
    formData.append('modelName', form.modelName)
    formData.append('iterations', form.iterations)
    formData.append('resolution', form.resolution)

    const res = await generate3DGS(formData)

    ElMessage.success('生成任务已提交')
    currentTask.value = {
      taskId: res.data.taskId,
      modelName: form.modelName,
      status: 'processing',
      progress: 0,
      log: '',
      autoOptimize: form.autoOptimize,
      optimizeConfig: form.autoOptimize ? {
        algorithm: form.algorithm,
        compressionLevel: form.compressionLevel,
        qualityFirst: form.qualityFirst
      } : null
    }

    startStatusCheck(res.data.taskId)
    handleClear()
  } catch (error) {
    ElMessage.error(error.message || '提交失败')
  } finally {
    generating.value = false
  }
}

const startStatusCheck = (taskId) => {
  if (statusCheckInterval) {
    clearInterval(statusCheckInterval)
  }

  statusCheckInterval = setInterval(async () => {
    try {
      const res = await checkGenerateStatus(taskId)
      currentTask.value = {
        ...currentTask.value,
        status: res.data.status,
        progress: res.data.progress || 0,
        log: res.data.log || ''
      }

      if (res.data.status === 'completed' || res.data.status === 'failed') {
        clearInterval(statusCheckInterval)
        statusCheckInterval = null
        loadModels()

        if (res.data.status === 'completed') {
          ElMessage.success('模型生成完成')

          // 如果启用了自动优化，则自动开始优化
          if (currentTask.value.autoOptimize && res.data.modelId) {
            startAutoOptimize(res.data.modelId)
          }
        } else {
          ElMessage.error('模型生成失败')
        }
      }
    } catch (error) {
      console.error('检查状态失败:', error)
    }
  }, 3000)
}

const startAutoOptimize = async (modelId) => {
  try {
    ElMessage.info('开始自动优化模型...')

    const { optimizeModel: apiOptimizeModel, checkOptimizeStatus } = await import('@/api/3dgs')

    const optimizeConfig = currentTask.value.optimizeConfig
    const res = await apiOptimizeModel({
      modelId: modelId,
      algorithm: optimizeConfig.algorithm,
      compressionLevel: optimizeConfig.compressionLevel,
      qualityFirst: optimizeConfig.qualityFirst
    })

    ElMessage.success('优化任务已提交')

    // 更新当前任务为优化状态
    currentTask.value = {
      ...currentTask.value,
      taskId: res.data.taskId,
      status: 'optimizing',
      progress: 0,
      log: currentTask.value.log + '\n\n========== 开始自动优化 ==========\n',
      originalModelId: modelId
    }

    // 开始检查优化状态
    startOptimizeStatusCheck(res.data.taskId)
  } catch (error) {
    ElMessage.error('自动优化启动失败: ' + (error.message || '未知错误'))
  }
}

const startOptimizeStatusCheck = (taskId) => {
  const checkInterval = setInterval(async () => {
    try {
      const { checkOptimizeStatus } = await import('@/api/3dgs')
      const res = await checkOptimizeStatus(taskId)

      currentTask.value = {
        ...currentTask.value,
        status: res.data.status === 'completed' ? 'completed' : 'optimizing',
        progress: res.data.progress || 0,
        log: currentTask.value.log + (res.data.log || '').split('\n').slice(-5).join('\n')
      }

      if (res.data.status === 'completed' || res.data.status === 'failed') {
        clearInterval(checkInterval)
        loadModels()

        if (res.data.status === 'completed') {
          ElMessage.success('模型优化完成！')
          if (res.data.comparison) {
            ElMessage.success(
              `压缩率: ${res.data.comparison.compressionRatio}%, ` +
              `质量分数: ${res.data.comparison.psnr} dB`
            )
          }
        } else {
          ElMessage.error('模型优化失败')
        }
      }
    } catch (error) {
      console.error('检查优化状态失败:', error)
    }
  }, 3000)
}

const loadModels = async () => {
  loadingModels.value = true
  try {
    const res = await getModelList()
    models.value = res.data || []
  } catch (error) {
    ElMessage.error('加载模型列表失败')
  } finally {
    loadingModels.value = false
  }
}

const viewModel = (model) => {
  router.push({ path: '/viewer', query: { modelId: model.id } })
}

const optimizeModel = (model) => {
  router.push({ path: '/optimize', query: { modelId: model.id, modelName: model.name } })
}

const deleteModel = async (model) => {
  try {
    await ElMessageBox.confirm(`确定要删除模型 "${model.name}" 吗？`, '警告', {
      confirmButtonText: '确定',
      cancelButtonText: '取消',
      type: 'warning'
    })

    await apiDeleteModel(model.id)
    ElMessage.success('删除成功')
    loadModels()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败')
    }
  }
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
})

onUnmounted(() => {
  if (statusCheckInterval) {
    clearInterval(statusCheckInterval)
  }
})
</script>

<style scoped>
.generate-container {
  max-width: 1200px;
  margin: 0 auto;
}

.upload-card, .task-card, .models-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
}

.upload-zone {
  margin-bottom: 20px;
}

.upload-icon {
  font-size: 67px;
  color: #c0c4cc;
  margin-bottom: 16px;
}

.upload-text {
  font-size: 14px;
  color: #606266;
}

.upload-text em {
  color: #409eff;
  font-style: normal;
}

.upload-tip {
  color: #909399;
  font-size: 12px;
  margin-top: 7px;
}

.config-section {
  margin-top: 20px;
}

.button-group {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-top: 20px;
}

.task-info {
  padding: 10px 0;
}

.task-log {
  margin-top: 20px;
}

.task-log pre {
  font-family: 'Courier New', monospace;
  font-size: 12px;
  line-height: 1.5;
  color: #333;
  background: #f5f7fa;
  padding: 10px;
  border-radius: 4px;
  margin: 0;
}
</style>
