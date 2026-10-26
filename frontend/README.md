# 3DGS 模型管理前端

基于 Vue3 + Element Plus 的 3D Gaussian Splatting 模型管理平台。

## 功能特性

### 1. 模型生成
- 拖拽上传 ZIP 格式的图像序列
- 支持自定义训练参数（迭代次数、分辨率）
- 实时显示生成进度和日志
- 模型列表管理

### 2. 模型压缩优化
- 支持多种优化算法：
  - **LightGaussian**: 基于剪枝的轻量化压缩
  - **Mini-Splatting**: 快速压缩算法
  - **Compact3D**: 高质量压缩方案
- 可配置压缩级别（轻度/中度/重度）
- 优化前后对比（文件大小、压缩率、质量分数）
- 优化历史记录

### 3. 模型可视化
- 3DGS 模型实时渲染
- 原始模型与优化模型对比查看
- 交互控制（旋转、平移、缩放）
- 渲染质量调节
- 截图功能

## 技术栈

- **Vue 3**: 渐进式 JavaScript 框架
- **Vite**: 下一代前端构建工具
- **Element Plus**: Vue 3 UI 组件库
- **Vue Router**: 官方路由管理器
- **Axios**: HTTP 客户端
- **Pinia**: Vue 状态管理

## 项目结构

```
3dgs-frontend/
├── src/
│   ├── api/              # API 接口
│   │   └── 3dgs.js
│   ├── router/           # 路由配置
│   │   └── index.js
│   ├── utils/            # 工具函数
│   │   └── request.js
│   ├── views/            # 页面组件
│   │   ├── Generate.vue  # 模型生成页面
│   │   ├── Optimize.vue  # 压缩优化页面
│   │   └── Viewer.vue    # 模型查看页面
│   ├── App.vue           # 根组件
│   └── main.js           # 入口文件
├── index.html
├── vite.config.js        # Vite 配置
└── package.json
```

## 快速开始

### 安装依赖

```bash
cd 3dgs-frontend
npm install
```

### 开发模式

```bash
npm run dev
```

访问 http://localhost:3000

### 构建生产版本

```bash
npm run build
```

### 预览生产构建

```bash
npm run preview
```

## API 接口说明

### 后端接口需求

后端服务需要实现以下 API 接口：

#### 1. 模型生成

**POST** `/api/3dgs/generate`
- 请求: `multipart/form-data`
  - `file`: ZIP 文件
  - `modelName`: 模型名称
  - `iterations`: 训练迭代次数
  - `resolution`: 图像分辨率
- 响应: `{ code: 200, data: { taskId: string } }`

**GET** `/api/3dgs/status/:taskId`
- 响应: `{ code: 200, data: { status, progress, log } }`

#### 2. 模型列表

**GET** `/api/3dgs/models`
- 响应: `{ code: 200, data: [ { id, name, size, status, createdAt } ] }`

**GET** `/api/3dgs/model/:modelId`
- 响应: `{ code: 200, data: { id, name, size, pointCount, ... } }`

**DELETE** `/api/3dgs/model/:modelId`
- 响应: `{ code: 200 }`

#### 3. 模型优化

**POST** `/api/3dgs/optimize`
- 请求: `{ modelId, algorithm, compressionLevel, qualityFirst }`
- 响应: `{ code: 200, data: { taskId } }`

**GET** `/api/3dgs/optimize/status/:taskId`
- 响应: `{ code: 200, data: { status, progress, log, comparison } }`

#### 4. 文件访问

**GET** `/api/3dgs/file/:modelId`
- 返回 3DGS 模型文件（.ply 或其他格式）

**GET** `/api/3dgs/download/:taskId`
- 下载优化后的模型

### ZIP 文件验证

后端需要验证上传的 ZIP 文件：
- 包含有效的图像文件（.jpg, .png 等）
- 图像数量符合要求（建议至少 20 张）
- 文件大小限制
- 可选：检查 EXIF 信息、图像质量等

## 3DGS 查看器集成

当前 Viewer 页面提供了完整的 UI 框架，但需要集成实际的 3DGS 渲染库。推荐方案：

### 方案 1: @antimatter15/splat
```bash
npm install @antimatter15/splat
```

### 方案 2: gaussian-splat-3d
```bash
npm install gaussian-splat-3d
```

### 方案 3: 自定义 WebGL 渲染器
参考开源实现：
- https://github.com/antimatter15/splat
- https://github.com/mkkellogg/GaussianSplats3D

## 压缩优化算法

推荐的开源压缩算法：

1. **LightGaussian**
   - GitHub: https://github.com/VITA-Group/LightGaussian
   - 特点: 轻量化、高压缩率

2. **Mini-Splatting**
   - GitHub: https://github.com/fatPeter/Mini-Splatting
   - 特点: 快速压缩、实时渲染

3. **Compact3D**
   - GitHub: https://github.com/UCDvision/compact3d
   - 特点: 保持高质量

## 代理配置

开发模式下，所有 `/api` 请求会被代理到 `http://localhost:8000`。

修改代理地址可编辑 `vite.config.js`:

```js
server: {
  proxy: {
    '/api': {
      target: 'http://your-backend-url',
      changeOrigin: true
    }
  }
}
```

## 浏览器支持

- Chrome >= 90
- Firefox >= 88
- Safari >= 14
- Edge >= 90

需要 WebGL 2.0 支持以运行 3DGS 查看器。

## 许可证

MIT
