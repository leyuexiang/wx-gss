<template>
  <div class="generate-container">
    <el-row :gutter="20">
      <!-- 左侧：上传和配置 -->
      <el-col :span="14">
        <el-card class="upload-card">
          <template #header>
            <div class="card-header">
              <el-icon><UploadFilled /></el-icon>
              <span>数据上传</span>
            </div>
          </template>

          <el-upload
            ref="uploadRef"
            class="upload-zone"
            drag
            :auto-upload="false"
            :on-change="handleFileChange"
            :limit="1"
            accept=".zip,.mp4,.avi,.mov,.mkv,.webm"
            :file-list="fileList"
          >
            <el-icon class="upload-icon"><FolderOpened /></el-icon>
            <div class="upload-text">拖拽文件到此处，或<em>点击选择</em></div>
            <template #tip>
              <div class="upload-tip">
                <div>• 支持 ZIP 压缩包（包含图像序列）</div>
                <div>• 支持视频文件（MP4/AVI/MOV/MKV/WebM）</div>
                <div>• 视频会自动抽帧处理</div>
                <div>• 建议 50+ 张图像或 30 秒以上视频</div>
              </div>
            </template>
          </el-upload>

          <div v-if="fileList.length > 0" class="file-info">
            <el-alert :type="isVideoFile ? 'warning' : 'success'" :closable="false">
              <template #title>
                <div style="display: flex; align-items: center; justify-content: space-between">
                  <div>
                    <span v-if="isVideoFile">
                      <el-icon><VideoCamera /></el-icon>
                      视频文件: {{ fileList[0].name }}
                    </span>
                    <span v-else>
                      <el-icon><Folder /></el-icon>
                      已选择文件: {{ fileList[0].name }}
                    </span>
                  </div>
                  <el-button size="small" text @click="handleClear">更换文件</el-button>
                </div>
              </template>
            </el-alert>

            <!-- 视频抽帧配置 -->
            <el-card class="frame-extraction-card" v-if="isVideoFile" shadow="never">
              <template #header>
                <div style="display: flex; align-items: center; gap: 8px">
                  <el-icon><VideoCameraFilled /></el-icon>
                  <span>视频抽帧配置</span>
                </div>
              </template>

              <el-form :model="videoConfig" label-width="140px" label-position="left" size="default">
                <el-form-item label="抽帧模式">
                  <el-radio-group v-model="videoConfig.extractionMode">
                    <el-radio-button value="fps">
                      <div>按帧率抽取</div>
                    </el-radio-button>
                    <el-radio-button value="interval">
                      <div>按时间间隔</div>
                    </el-radio-button>
                    <el-radio-button value="count">
                      <div>指定总帧数</div>
                    </el-radio-button>
                  </el-radio-group>
                </el-form-item>

                <el-form-item label="抽帧参数" v-if="videoConfig.extractionMode === 'fps'">
                  <el-slider
                    v-model="videoConfig.fps"
                    :min="1"
                    :max="30"
                    :marks="{ 1: '1', 5: '5', 10: '10', 30: '30' }"
                    show-input
                    :input-size="'small'"
                  />
                  <div class="param-hint">
                    每秒抽取帧数，推荐 3-10 FPS
                  </div>
                </el-form-item>

                <el-form-item label="时间间隔" v-if="videoConfig.extractionMode === 'interval'">
                  <el-input-number
                    v-model="videoConfig.interval"
                    :min="0.1"
                    :max="5"
                    :step="0.1"
                    :precision="1"
                  />
                  <span style="margin-left: 8px">秒</span>
                  <div class="param-hint">
                    每隔多少秒抽取一帧，推荐 0.2-0.5 秒
                  </div>
                </el-form-item>

                <el-form-item label="总帧数" v-if="videoConfig.extractionMode === 'count'">
                  <el-input-number
                    v-model="videoConfig.totalFrames"
                    :min="20"
                    :max="500"
                    :step="10"
                  />
                  <div class="param-hint">
                    从视频中均匀抽取指定数量的帧，推荐 50-150 帧
                  </div>
                </el-form-item>

                <el-form-item label="起始时间">
                  <el-input-number
                    v-model="videoConfig.startTime"
                    :min="0"
                    :step="1"
                    :precision="1"
                  />
                  <span style="margin-left: 8px">秒</span>
                  <div class="param-hint">
                    从视频的第几秒开始抽帧，跳过片头等无用部分
                  </div>
                </el-form-item>

                <el-form-item label="结束时间">
                  <el-input-number
                    v-model="videoConfig.endTime"
                    :min="0"
                    :step="1"
                    :precision="1"
                    placeholder="留空则到视频结尾"
                  />
                  <span style="margin-left: 8px">秒</span>
                  <div class="param-hint">
                    到视频的第几秒结束，0 表示处理到结尾
                  </div>
                </el-form-item>

                <el-form-item label="图像质量">
                  <el-radio-group v-model="videoConfig.quality">
                    <el-radio-button value="low">低 (快速)</el-radio-button>
                    <el-radio-button value="medium">中 (推荐)</el-radio-button>
                    <el-radio-button value="high">高 (最佳)</el-radio-button>
                  </el-radio-group>
                  <div class="param-hint">
                    抽取帧的 JPEG 质量，影响后续训练效果
                  </div>
                </el-form-item>

                <el-form-item label="去除模糊帧">
                  <el-switch v-model="videoConfig.removeBlurry" />
                  <span style="margin-left: 10px; color: var(--text-muted)">
                    自动检测并移除模糊的帧
                  </span>
                </el-form-item>

                <el-form-item label="去除重复帧">
                  <el-switch v-model="videoConfig.removeDuplicates" />
                  <span style="margin-left: 10px; color: var(--text-muted)">
                    移除相似度过高的连续帧
                  </span>
                </el-form-item>

                <el-divider />

                <el-alert type="info" :closable="false" show-icon>
                  <template #title>
                    <div style="font-size: 13px">
                      <div><strong>预估抽帧结果</strong></div>
                      <div style="margin-top: 8px; color: var(--text-secondary)">
                        <span v-if="videoConfig.extractionMode === 'fps'">
                          按 {{ videoConfig.fps }} FPS 抽取
                        </span>
                        <span v-else-if="videoConfig.extractionMode === 'interval'">
                          每 {{ videoConfig.interval }} 秒抽取一帧
                        </span>
                        <span v-else>
                          均匀抽取 {{ videoConfig.totalFrames }} 帧
                        </span>
                      </div>
                      <div style="margin-top: 4px; color: var(--text-muted); font-size: 12px">
                        建议最终帧数在 50-150 之间，视频长度不同需调整参数
                      </div>
                    </div>
                  </template>
                </el-alert>
              </el-form>
            </el-card>
          </div>
        </el-card>

        <el-card class="config-card" v-if="fileList.length > 0">
          <template #header>
            <div class="card-header">
              <el-icon><Setting /></el-icon>
              <span>训练配置</span>
            </div>
          </template>

          <el-tabs v-model="activeTab" class="config-tabs">
            <!-- 基础配置 -->
            <el-tab-pane label="基础配置" name="basic">
              <el-form :model="form" label-width="140px" label-position="left">
                <el-form-item label="模型名称" required>
                  <el-input v-model="form.modelName" placeholder="为模型命名" clearable />
                </el-form-item>

                <el-form-item label="训练迭代次数">
                  <el-slider
                    v-model="form.iterations"
                    :min="1000"
                    :max="30000"
                    :step="1000"
                    :marks="iterationMarks"
                    show-input
                    :input-size="'small'"
                  />
                  <div class="param-hint">
                    推荐 7000-15000 次，迭代次数越多质量越好但耗时更长
                  </div>
                </el-form-item>

                <el-form-item label="图像分辨率">
                  <el-radio-group v-model="form.resolution">
                    <el-radio-button :label="1">
                      <span>原始分辨率</span>
                      <el-tag size="small" type="success" style="margin-left: 8px">最高质量</el-tag>
                    </el-radio-button>
                    <el-radio-button :label="2">
                      <span>1/2 分辨率</span>
                      <el-tag size="small" type="warning" style="margin-left: 8px">推荐</el-tag>
                    </el-radio-button>
                    <el-radio-button :label="4">1/4 分辨率</el-radio-button>
                    <el-radio-button :label="8">1/8 分辨率</el-radio-button>
                  </el-radio-group>
                  <div class="param-hint">
                    较低分辨率可加快训练速度，推荐从 1/2 开始
                  </div>
                </el-form-item>

                <el-form-item label="场景类型">
                  <el-select v-model="form.sceneType" placeholder="选择场景类型">
                    <el-option label="室外场景" value="outdoor">
                      <div>
                        <div><strong>室外场景</strong></div>
                        <div style="font-size: 12px; color: #909399">建筑、街景、自然景观等</div>
                      </div>
                    </el-option>
                    <el-option label="室内场景" value="indoor">
                      <div>
                        <div><strong>室内场景</strong></div>
                        <div style="font-size: 12px; color: #909399">房间、展厅等封闭空间</div>
                      </div>
                    </el-option>
                    <el-option label="物体/人物" value="object">
                      <div>
                        <div><strong>物体/人物</strong></div>
                        <div style="font-size: 12px; color: #909399">单个物体或人物的 360° 拍摄</div>
                      </div>
                    </el-option>
                  </el-select>
                </el-form-item>
              </el-form>
            </el-tab-pane>

            <!-- 高级配置 -->
            <el-tab-pane label="高级配置" name="advanced">
              <el-form :model="form" label-width="160px" label-position="left">
                <el-divider content-position="left">
                  <el-icon><Tools /></el-icon>
                  <span style="margin-left: 5px">COLMAP 参数</span>
                </el-divider>

                <el-form-item label="特征提取器">
                  <el-radio-group v-model="form.colmap.featureExtractor">
                    <el-radio label="sift">SIFT (标准)</el-radio>
                    <el-radio label="sift-gpu">SIFT-GPU (快速)</el-radio>
                  </el-radio-group>
                  <div class="param-hint">
                    GPU 模式需要 CUDA 支持，速度提升约 4-6 倍
                  </div>
                </el-form-item>

                <el-form-item label="匹配模式">
                  <el-select v-model="form.colmap.matcher">
                    <el-option label="Sequential (顺序)" value="sequential" />
                    <el-option label="Exhaustive (穷举)" value="exhaustive" />
                    <el-option label="Vocab Tree (词汇树)" value="vocab_tree" />
                  </el-select>
                  <div class="param-hint">
                    顺序匹配适合视频序列，穷举适合无序照片
                  </div>
                </el-form-item>

                <el-divider content-position="left">
                  <el-icon><Cpu /></el-icon>
                  <span style="margin-left: 5px">3DGS 训练参数</span>
                </el-divider>

                <el-form-item label="位置学习率">
                  <el-input-number v-model="form.advanced.positionLR" :min="0.00001" :max="0.001" :step="0.00001" :precision="5" />
                  <div class="param-hint">
                    默认 0.00016，控制高斯点位置更新速度
                  </div>
                </el-form-item>

                <el-form-item label="密集化间隔">
                  <el-input-number v-model="form.advanced.densifyInterval" :min="100" :max="500" :step="100" />
                  <div class="param-hint">
                    默认 100 次迭代，控制高斯点增加频率
                  </div>
                </el-form-item>

                <el-form-item label="不透明度重置">
                  <el-input-number v-model="form.advanced.opacityReset" :min="1000" :max="5000" :step="1000" />
                  <div class="param-hint">
                    默认 3000 次迭代重置不透明度
                  </div>
                </el-form-item>

                <el-form-item label="SH 阶数">
                  <el-slider v-model="form.advanced.shDegree" :min="0" :max="3" :marks="{ 0: '0', 1: '1', 2: '2', 3: '3' }" show-stops />
                  <div class="param-hint">
                    球谐函数阶数，越高视角依赖效果越好，推荐 3
                  </div>
                </el-form-item>

                <el-divider content-position="left">
                  <el-icon><MagicStick /></el-icon>
                  <span style="margin-left: 5px">质量优化 & 去噪</span>
                </el-divider>

                <el-form-item label="启用去噪">
                  <el-switch v-model="form.advanced.enableDenoising" />
                  <span style="margin-left: 10px; color: #909399">
                    去除训练过程中的噪点和杂点
                  </span>
                </el-form-item>

                <template v-if="form.advanced.enableDenoising">
                  <el-form-item label="去噪强度">
                    <el-slider
                      v-model="form.advanced.denoiseStrength"
                      :min="0"
                      :max="100"
                      :marks="{ 0: '弱', 50: '中', 100: '强' }"
                      show-stops
                    />
                    <div class="param-hint">
                      强度越高，去噪效果越明显，但可能丢失细节
                    </div>
                  </el-form-item>

                  <el-form-item label="不透明度阈值">
                    <el-input-number
                      v-model="form.advanced.opacityThreshold"
                      :min="0.001"
                      :max="0.1"
                      :step="0.001"
                      :precision="3"
                    />
                    <div class="param-hint">
                      默认 0.005，低于此值的高斯点将被剔除（去除杂点）
                    </div>
                  </el-form-item>

                  <el-form-item label="尺寸过滤">
                    <el-switch v-model="form.advanced.sizeFiltering" />
                    <span style="margin-left: 10px; color: #909399">
                      过滤过大或过小的高斯点
                    </span>
                  </el-form-item>

                  <el-form-item label="最小尺寸阈值" v-if="form.advanced.sizeFiltering">
                    <el-input-number
                      v-model="form.advanced.minSizeThreshold"
                      :min="0.0001"
                      :max="0.01"
                      :step="0.0001"
                      :precision="4"
                    />
                    <div class="param-hint">
                      默认 0.0002，小于此值的微小高斯点将被移除
                    </div>
                  </el-form-item>

                  <el-form-item label="最大尺寸阈值" v-if="form.advanced.sizeFiltering">
                    <el-input-number
                      v-model="form.advanced.maxSizeThreshold"
                      :min="0.1"
                      :max="10"
                      :step="0.1"
                      :precision="1"
                    />
                    <div class="param-hint">
                      默认 2.0，大于此值的过大高斯点将被分裂或移除
                    </div>
                  </el-form-item>
                </template>

                <el-form-item label="梯度裁剪">
                  <el-switch v-model="form.advanced.gradientClipping" />
                  <span style="margin-left: 10px; color: #909399">
                    防止训练过程中的梯度爆炸
                  </span>
                </el-form-item>

                <el-form-item label="平滑正则化" v-if="form.advanced.gradientClipping">
                  <el-slider
                    v-model="form.advanced.smoothnessWeight"
                    :min="0"
                    :max="1"
                    :step="0.01"
                    :marks="{ 0: '0', 0.5: '0.5', 1: '1' }"
                    show-input
                    :input-size="'small'"
                  />
                  <div class="param-hint">
                    增加平滑约束，减少视觉噪声，默认 0.1
                  </div>
                </el-form-item>

                <el-form-item label="深度正则化">
                  <el-switch v-model="form.advanced.depthRegularization" />
                  <span style="margin-left: 10px; color: #909399">
                    改善深度一致性，减少漂浮物
                  </span>
                </el-form-item>

                <el-form-item label="抗锯齿">
                  <el-switch v-model="form.advanced.antiAliasing" />
                  <span style="margin-left: 10px; color: #909399">
                    开启抗锯齿，提升边缘质量
                  </span>
                </el-form-item>
              </el-form>
            </el-tab-pane>

            <!-- 优化配置 -->
            <el-tab-pane label="压缩优化" name="optimization">
              <el-form :model="form" label-width="140px" label-position="left">
                <el-form-item label="自动优化">
                  <el-switch v-model="form.autoOptimize" @change="handleAutoOptimizeChange" />
                  <span style="margin-left: 10px; color: #909399">
                    训练完成后自动进行模型压缩
                  </span>
                </el-form-item>

                <template v-if="form.autoOptimize">
                  <el-divider />

                  <el-form-item label="优化算法">
                    <el-radio-group v-model="form.optimize.algorithm" class="algorithm-group">
                      <el-radio label="lightgaussian" border>
                        <div class="algorithm-option">
                          <div class="algorithm-name">
                            <span>LightGaussian</span>
                            <el-tag size="small" type="success">推荐</el-tag>
                          </div>
                          <div class="algorithm-desc">基于剪枝，压缩率高，质量损失小</div>
                          <div class="algorithm-stats">
                            <el-tag size="small" effect="plain">压缩率: 70-80%</el-tag>
                            <el-tag size="small" effect="plain">速度: 中等</el-tag>
                          </div>
                        </div>
                      </el-radio>

                      <el-radio label="mini-splatting" border>
                        <div class="algorithm-option">
                          <div class="algorithm-name">
                            <span>Mini-Splatting</span>
                            <el-tag size="small" type="warning">快速</el-tag>
                          </div>
                          <div class="algorithm-desc">快速压缩，适合实时渲染应用</div>
                          <div class="algorithm-stats">
                            <el-tag size="small" effect="plain">压缩率: 60-70%</el-tag>
                            <el-tag size="small" effect="plain">速度: 快</el-tag>
                          </div>
                        </div>
                      </el-radio>

                      <el-radio label="compact3d" border>
                        <div class="algorithm-option">
                          <div class="algorithm-name">
                            <span>Compact3D</span>
                            <el-tag size="small" type="info">高质量</el-tag>
                          </div>
                          <div class="algorithm-desc">保持高质量，压缩时间较长</div>
                          <div class="algorithm-stats">
                            <el-tag size="small" effect="plain">压缩率: 75-85%</el-tag>
                            <el-tag size="small" effect="plain">速度: 慢</el-tag>
                          </div>
                        </div>
                      </el-radio>
                    </el-radio-group>
                  </el-form-item>

                  <el-form-item label="压缩级别">
                    <el-radio-group v-model="form.optimize.compressionLevel" class="compression-level">
                      <el-radio-button label="low">
                        <div>轻度压缩</div>
                        <div style="font-size: 11px; opacity: 0.8">保留 50-70%</div>
                      </el-radio-button>
                      <el-radio-button label="medium">
                        <div>中度压缩</div>
                        <div style="font-size: 11px; opacity: 0.8">保留 30-50%</div>
                      </el-radio-button>
                      <el-radio-button label="high">
                        <div>重度压缩</div>
                        <div style="font-size: 11px; opacity: 0.8">保留 10-30%</div>
                      </el-radio-button>
                    </el-radio-group>
                  </el-form-item>

                  <el-form-item label="质量优先">
                    <el-switch v-model="form.optimize.qualityFirst" />
                    <span style="margin-left: 10px; color: #909399">
                      优先保证渲染质量，可能降低压缩率
                    </span>
                  </el-form-item>
                </template>
              </el-form>
            </el-tab-pane>
          </el-tabs>
        </el-card>
      </el-col>

      <!-- 右侧：预览和操作 -->
      <el-col :span="10">
        <el-card class="preview-card">
          <template #header>
            <div class="card-header">
              <el-icon><DataAnalysis /></el-icon>
              <span>配置预览</span>
            </div>
          </template>

          <div v-if="fileList.length === 0" class="empty-preview">
            <el-empty description="请先上传数据文件" />
          </div>

          <div v-else class="preview-content">
            <el-descriptions :column="1" border size="small">
              <el-descriptions-item label="文件名">
                {{ fileList[0].name }}
              </el-descriptions-item>
              <el-descriptions-item label="模型名称">
                {{ form.modelName || '未设置' }}
              </el-descriptions-item>
              <el-descriptions-item label="训练迭代">
                {{ form.iterations }} 次
              </el-descriptions-item>
              <el-descriptions-item label="图像分辨率">
                {{ resolutionText }}
              </el-descriptions-item>
              <el-descriptions-item label="场景类型">
                {{ sceneTypeText }}
              </el-descriptions-item>
              <el-descriptions-item label="自动优化">
                <el-tag :type="form.autoOptimize ? 'success' : 'info'" size="small">
                  {{ form.autoOptimize ? '已启用' : '未启用' }}
                </el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="优化算法" v-if="form.autoOptimize">
                {{ algorithmText }}
              </el-descriptions-item>
            </el-descriptions>

            <el-divider />

            <div class="estimate-info">
              <el-alert type="info" :closable="false">
                <template #title>
                  <div style="font-size: 13px">
                    <div><el-icon><Clock /></el-icon> 预计耗时</div>
                    <div style="margin-top: 5px; font-size: 20px; font-weight: bold">
                      {{ estimatedTime }}
                    </div>
                  </div>
                </template>
              </el-alert>
            </div>

            <div class="action-buttons">
              <el-button
                type="primary"
                size="large"
                @click="handleGenerate"
                :loading="generating"
                :disabled="!canGenerate"
                style="width: 100%"
              >
                <el-icon v-if="!generating"><VideoPlay /></el-icon>
                {{ generating ? '生成中...' : '开始生成' }}
              </el-button>
            </div>
          </div>
        </el-card>

        <!-- 任务进度 -->
        <el-card class="task-card" v-if="currentTask">
          <template #header>
            <div class="card-header">
              <el-icon><Loading /></el-icon>
              <span>任务进度</span>
            </div>
          </template>

          <div class="task-info">
            <div class="task-status">
              <el-tag :type="getStatusType(currentTask.status)" size="large">
                {{ getStatusText(currentTask.status) }}
              </el-tag>
              <span style="margin-left: 10px; color: #909399">
                {{ currentTask.modelName }}
              </span>
            </div>

            <div class="task-progress">
              <el-progress
                :percentage="currentTask.progress"
                :status="currentTask.status === 'failed' ? 'exception' : undefined"
                :stroke-width="20"
              >
                <template #default="{ percentage }">
                  <span style="font-size: 14px; font-weight: bold">{{ percentage }}%</span>
                </template>
              </el-progress>
            </div>

            <div class="task-log" v-if="currentTask.log">
              <el-scrollbar height="200px">
                <pre>{{ currentTask.log }}</pre>
              </el-scrollbar>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <!-- 模型列表 -->
    <el-card class="models-card" style="margin-top: 20px">
      <template #header>
        <div class="card-header">
          <div>
            <el-icon><Files /></el-icon>
            <span>已生成的模型</span>
          </div>
          <el-button type="primary" size="small" @click="loadModels" :icon="Refresh">刷新</el-button>
        </div>
      </template>

      <el-table :data="models" style="width: 100%" v-loading="loadingModels">
        <el-table-column prop="name" label="模型名称" min-width="200" />
        <el-table-column prop="createdAt" label="创建时间" width="180" />
        <el-table-column label="迭代次数" width="100">
          <template #default="scope">
            {{ scope.row.iterations || '-' }}
          </template>
        </el-table-column>
        <el-table-column label="点云数量" width="120">
          <template #default="scope">
            {{ formatNumber(scope.row.pointCount) }}
          </template>
        </el-table-column>
        <el-table-column prop="size" label="文件大小" width="120">
          <template #default="scope">
            {{ formatFileSize(scope.row.size) }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="100">
          <template #default="scope">
            <el-tag :type="scope.row.status === 'completed' ? 'success' : 'info'" size="small">
              {{ scope.row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="220" fixed="right">
          <template #default="scope">
            <el-button size="small" @click="viewModel(scope.row)" :icon="View">查看</el-button>
            <el-button size="small" type="warning" @click="optimizeModel(scope.row)" :icon="Operation">优化</el-button>
            <el-button size="small" type="danger" @click="deleteModel(scope.row)" :icon="Delete">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { generate3DGS, checkGenerateStatus, getModelList, deleteModel as apiDeleteModel } from '@/api/3dgs'
import {
  Refresh, View, Operation, Delete, Clock, Loading,
  UploadFilled, FolderOpened, Setting, Tools, Cpu, DataAnalysis, Files, VideoPlay, MagicStick,
  VideoCamera, VideoCameraFilled, Folder
} from '@element-plus/icons-vue'

const router = useRouter()
const uploadRef = ref(null)
const fileList = ref([])
const generating = ref(false)
const loadingModels = ref(false)
const currentTask = ref(null)
const models = ref([])
const activeTab = ref('basic')
const isVideoFile = ref(false)
let statusCheckInterval = null

const iterationMarks = {
  1000: '1K',
  7000: '7K',
  15000: '15K',
  30000: '30K'
}

const form = reactive({
  modelName: '',
  iterations: 7000,
  resolution: 2,
  sceneType: 'outdoor',
  colmap: {
    featureExtractor: 'sift-gpu',
    matcher: 'sequential'
  },
  advanced: {
    positionLR: 0.00016,
    densifyInterval: 100,
    opacityReset: 3000,
    shDegree: 3,
    // 去噪和质量优化
    enableDenoising: true,
    denoiseStrength: 50,
    opacityThreshold: 0.005,
    sizeFiltering: true,
    minSizeThreshold: 0.0002,
    maxSizeThreshold: 2.0,
    gradientClipping: true,
    smoothnessWeight: 0.1,
    depthRegularization: true,
    antiAliasing: true
  },
  autoOptimize: false,
  optimize: {
    algorithm: 'lightgaussian',
    compressionLevel: 'medium',
    qualityFirst: true
  }
})

const videoConfig = reactive({
  extractionMode: 'fps',
  fps: 5,
  interval: 0.3,
  totalFrames: 100,
  startTime: 0,
  endTime: 0,
  quality: 'medium',
  removeBlurry: true,
  removeDuplicates: true
})

const resolutionText = computed(() => {
  const map = {
    1: '原始分辨率',
    2: '1/2 分辨率',
    4: '1/4 分辨率',
    8: '1/8 分辨率'
  }
  return map[form.resolution] || '-'
})

const sceneTypeText = computed(() => {
  const map = {
    'outdoor': '室外场景',
    'indoor': '室内场景',
    'object': '物体/人物'
  }
  return map[form.sceneType] || '-'
})

const algorithmText = computed(() => {
  const map = {
    'lightgaussian': 'LightGaussian',
    'mini-splatting': 'Mini-Splatting',
    'compact3d': 'Compact3D'
  }
  return map[form.optimize.algorithm] || '-'
})

const estimatedTime = computed(() => {
  const baseTime = form.iterations / 1000 * 3 // 基础时间（分钟）
  const resolutionFactor = 1 / form.resolution // 分辨率影响
  const optimizeTime = form.autoOptimize ? 5 : 0 // 优化时间

  const totalMinutes = Math.ceil(baseTime * resolutionFactor + optimizeTime)

  if (totalMinutes < 60) {
    return `${totalMinutes} 分钟`
  } else {
    const hours = Math.floor(totalMinutes / 60)
    const minutes = totalMinutes % 60
    return `${hours} 小时 ${minutes} 分钟`
  }
})

const canGenerate = computed(() => {
  return fileList.value.length > 0 && form.modelName.trim().length > 0
})

const handleFileChange = (file) => {
  const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
  const fileName = file.name.toLowerCase()

  const isVideo = videoExtensions.some(ext => fileName.endsWith(ext))
  const isZip = fileName.endsWith('.zip')

  if (!isVideo && !isZip) {
    ElMessage.error('只支持 ZIP 或视频文件（MP4/AVI/MOV/MKV/WebM）')
    fileList.value = []
    isVideoFile.value = false
    return
  }

  fileList.value = [file]
  isVideoFile.value = isVideo

  if (!form.modelName) {
    const nameWithoutExt = file.name.replace(/\.(zip|mp4|avi|mov|mkv|webm)$/i, '')
    form.modelName = nameWithoutExt
  }

  if (isVideo) {
    ElMessage.info({
      message: '已检测到视频文件，将自动进行抽帧处理',
      duration: 3000
    })
  }
}

const handleClear = () => {
  fileList.value = []
  uploadRef.value?.clearFiles()
}

const handleAutoOptimizeChange = (value) => {
  if (value) {
    ElMessage.info('已启用自动优化，模型生成完成后将自动进行压缩')
  }
}

const handleGenerate = async () => {
  if (!canGenerate.value) {
    ElMessage.warning('请完善配置信息')
    return
  }

  generating.value = true

  try {
    const formData = new FormData()
    formData.append('file', fileList.value[0].raw)
    formData.append('modelName', form.modelName)
    formData.append('iterations', form.iterations)
    formData.append('resolution', form.resolution)
    formData.append('sceneType', form.sceneType)

    // 如果是视频文件，添加抽帧配置
    if (isVideoFile.value) {
      formData.append('isVideo', 'true')
      formData.append('videoConfig', JSON.stringify(videoConfig))
    }

    formData.append('config', JSON.stringify({
      colmap: form.colmap,
      advanced: form.advanced
    }))

    const res = await generate3DGS(formData)

    ElMessage.success('生成任务已提交')
    currentTask.value = {
      taskId: res.data.taskId,
      modelName: form.modelName,
      status: 'processing',
      progress: 0,
      log: '',
      autoOptimize: form.autoOptimize,
      optimizeConfig: form.autoOptimize ? form.optimize : null
    }

    startStatusCheck(res.data.taskId)
    activeTab.value = 'basic'
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

    currentTask.value = {
      ...currentTask.value,
      taskId: res.data.taskId,
      status: 'optimizing',
      progress: 0,
      log: currentTask.value.log + '\n\n========== 开始自动优化 ==========\n',
      originalModelId: modelId
    }

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
    'optimizing': 'warning',
    'completed': 'success',
    'failed': 'danger'
  }
  return map[status] || 'info'
}

const getStatusText = (status) => {
  const map = {
    'processing': '生成中',
    'optimizing': '优化中',
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

const formatNumber = (num) => {
  if (!num) return '-'
  return num.toLocaleString()
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
  max-width: 1800px;
  margin: 0 auto;
}

.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 600;
  font-size: 16px;
  color: var(--text-primary);
}

.card-header i {
  color: var(--accent-primary);
  font-size: 20px;
}

/* 上传区域增强 */
.upload-card {
  position: relative;
  overflow: hidden;
}

.upload-card::before {
  content: '';
  position: absolute;
  top: -2px;
  left: -2px;
  right: -2px;
  bottom: -2px;
  background: var(--accent-gradient);
  opacity: 0;
  border-radius: 16px;
  transition: opacity 0.3s ease;
  z-index: -1;
}

.upload-card:hover::before {
  opacity: 0.1;
}

.upload-zone {
  margin-bottom: 0;
}

.upload-icon {
  font-size: 67px;
  color: var(--accent-primary);
  margin-bottom: 16px;
  filter: drop-shadow(0 0 20px rgba(0, 217, 255, 0.3));
  animation: float 3s ease-in-out infinite;
}

@keyframes float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-10px); }
}

.upload-text {
  font-size: 16px;
  color: var(--text-secondary);
  font-weight: 500;
}

.upload-text em {
  color: var(--accent-primary);
  font-style: normal;
  font-weight: 600;
}

.upload-tip {
  color: var(--text-muted);
  font-size: 13px;
  margin-top: 12px;
  line-height: 1.8;
}

.upload-tip div {
  display: flex;
  align-items: center;
  gap: 8px;
}

.upload-tip div::before {
  content: '→';
  color: var(--accent-primary);
  font-weight: bold;
}

.file-info {
  margin-top: 20px;
}

/* 配置卡片 */
.config-card {
  margin-top: 20px;
  background: linear-gradient(135deg, rgba(26, 31, 58, 0.6) 0%, rgba(18, 22, 41, 0.6) 100%) !important;
}

.config-tabs {
  margin-top: -10px;
}

.config-tabs :deep(.el-tabs__header) {
  margin: 0;
  padding: 0 20px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 12px 12px 0 0;
}

.config-tabs :deep(.el-tabs__item) {
  padding: 0 30px;
  height: 50px;
  line-height: 50px;
  font-weight: 500;
  transition: all 0.3s ease;
}

.config-tabs :deep(.el-tabs__item:hover) {
  color: var(--accent-primary);
}

.param-hint {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 8px;
  line-height: 1.5;
  padding: 8px 12px;
  background: rgba(0, 217, 255, 0.05);
  border-left: 2px solid var(--accent-primary);
  border-radius: 4px;
}

/* 算法选择增强 */
.algorithm-group {
  display: flex;
  flex-direction: column;
  gap: 16px;
  width: 100%;
}

.algorithm-group :deep(.el-radio) {
  margin-right: 0;
  width: 100%;
}

.algorithm-group :deep(.el-radio.is-bordered) {
  padding: 20px;
  height: auto;
  border-radius: 12px;
  border: 2px solid var(--border-color);
  background: rgba(255, 255, 255, 0.02);
  transition: all 0.3s ease;
}

.algorithm-group :deep(.el-radio.is-bordered:hover) {
  border-color: var(--accent-primary);
  background: rgba(0, 217, 255, 0.05);
  transform: translateX(5px);
  box-shadow: -5px 0 15px rgba(0, 217, 255, 0.2);
}

.algorithm-group :deep(.el-radio.is-bordered.is-checked) {
  border-color: var(--accent-primary);
  background: rgba(0, 217, 255, 0.08);
  box-shadow: 0 0 20px rgba(0, 217, 255, 0.2);
}

.algorithm-option {
  width: 100%;
}

.algorithm-name {
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 600;
  font-size: 15px;
  margin-bottom: 8px;
  color: var(--text-primary);
}

.algorithm-desc {
  font-size: 13px;
  color: var(--text-secondary);
  margin-bottom: 10px;
  line-height: 1.5;
}

.algorithm-stats {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}

.compression-level :deep(.el-radio-button__inner) {
  padding: 16px 24px;
  height: auto;
  border-radius: 8px;
}

/* 预览卡片增强 */
.preview-card {
  position: sticky;
  top: 100px;
  background: linear-gradient(135deg, rgba(123, 47, 247, 0.1) 0%, rgba(0, 217, 255, 0.1) 100%),
              rgba(26, 31, 58, 0.6) !important;
  border: 2px solid var(--border-color) !important;
}

.preview-card::after {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: linear-gradient(
    45deg,
    transparent,
    rgba(0, 217, 255, 0.1),
    transparent
  );
  transform: rotate(45deg);
  animation: shine 3s infinite;
}

@keyframes shine {
  0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
  100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
}

.empty-preview {
  padding: 60px 0;
  text-align: center;
}

.preview-content {
  padding: 10px 0;
  position: relative;
  z-index: 1;
}

.preview-content :deep(.el-descriptions) {
  margin-bottom: 20px;
}

.estimate-info {
  margin: 24px 0;
}

.estimate-info :deep(.el-alert) {
  background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(123, 47, 247, 0.1) 100%) !important;
  border: 1px solid rgba(0, 217, 255, 0.3) !important;
  border-radius: 12px;
  padding: 20px;
}

.estimate-info :deep(.el-alert__title) {
  color: var(--text-primary) !important;
}

.action-buttons {
  margin-top: 24px;
}

.action-buttons :deep(.el-button--large) {
  height: 54px;
  font-size: 16px;
  font-weight: 600;
  letter-spacing: 0.5px;
  border-radius: 12px;
}

/* 任务进度卡片 */
.task-card {
  margin-top: 20px;
  border: 2px solid rgba(255, 193, 7, 0.3) !important;
  animation: pulse-border 2s ease-in-out infinite;
}

@keyframes pulse-border {
  0%, 100% { border-color: rgba(255, 193, 7, 0.3); }
  50% { border-color: rgba(255, 193, 7, 0.6); }
}

.task-info {
  padding: 10px 0;
}

.task-status {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 20px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 8px;
  border-left: 3px solid var(--accent-primary);
}

.task-status :deep(.el-tag--large) {
  font-size: 14px;
  padding: 8px 16px;
  font-weight: 600;
}

.task-progress {
  margin-bottom: 20px;
}

.task-progress :deep(.el-progress-bar__outer) {
  height: 24px !important;
  border-radius: 12px;
  overflow: hidden;
}

.task-progress :deep(.el-progress-bar__inner) {
  border-radius: 12px;
  background: var(--accent-gradient) !important;
  animation: progressShine 2s linear infinite;
}

@keyframes progressShine {
  0% { background-position: -200% 0; }
  100% { background-position: 200% 0; }
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
  font-family: 'Courier New', 'Consolas', monospace;
  font-size: 12px;
  line-height: 1.6;
  color: #00ff88;
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.task-log :deep(.el-scrollbar__thumb) {
  background: rgba(0, 217, 255, 0.3) !important;
}

/* 模型列表 */
.models-card {
  background: linear-gradient(135deg, rgba(26, 31, 58, 0.6) 0%, rgba(18, 22, 41, 0.6) 100%) !important;
}

.models-card :deep(.el-table) {
  background: transparent !important;
}

.models-card :deep(.el-table__body tr:hover) {
  background: rgba(0, 217, 255, 0.05) !important;
  cursor: pointer;
}

.models-card :deep(.el-button) {
  transition: all 0.3s ease;
}

.models-card :deep(.el-button:hover) {
  transform: translateY(-2px);
}

/* 响应式 */
@media (max-width: 1400px) {
  .generate-container :deep(.el-col-10) {
    flex: 0 0 100%;
    max-width: 100%;
  }

  .generate-container :deep(.el-col-14) {
    flex: 0 0 100%;
    max-width: 100%;
  }

  .preview-card {
    position: static;
    margin-top: 20px;
  }
}

/* 滚动条样式 */
:deep(.el-scrollbar__thumb) {
  background: rgba(0, 217, 255, 0.3) !important;
  border-radius: 10px;
}

:deep(.el-scrollbar__thumb:hover) {
  background: rgba(0, 217, 255, 0.5) !important;
}

/* 表单标签增强 */
:deep(.el-form-item__label) {
  color: var(--text-secondary) !important;
  font-weight: 500;
}

/* Divider 增强 */
:deep(.el-divider) {
  border-color: var(--border-color) !important;
}

:deep(.el-divider__text) {
  background: transparent !important;
  color: var(--text-primary) !important;
  font-weight: 600;
  padding: 0 16px;
}

/* 输入框增强 */
:deep(.el-input-number) {
  width: 100%;
}

:deep(.el-input-number .el-input__wrapper) {
  background: rgba(255, 255, 255, 0.05) !important;
}

/* 滑块增强 */
:deep(.el-slider__marks-text) {
  color: var(--text-muted) !important;
  font-size: 11px;
  margin-top: 8px;
}

:deep(.el-slider__button) {
  width: 16px;
  height: 16px;
  border: 3px solid var(--accent-primary) !important;
  box-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
}

:deep(.el-slider__button:hover) {
  transform: scale(1.2);
}

/* 视频抽帧卡片 */
.frame-extraction-card {
  margin-top: 16px;
  border: 2px solid rgba(123, 47, 247, 0.3) !important;
  background: linear-gradient(135deg, rgba(123, 47, 247, 0.05) 0%, rgba(0, 217, 255, 0.05) 100%) !important;
}

.frame-extraction-card :deep(.el-card__header) {
  background: rgba(123, 47, 247, 0.1) !important;
  border-bottom-color: rgba(123, 47, 247, 0.3) !important;
}
</style>
