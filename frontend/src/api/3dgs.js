import request from '@/utils/request'

export const generate3DGS = (formData) => {
  return request({
    url: '/3dgs/generate',
    method: 'post',
    data: formData,
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

export const checkGenerateStatus = (taskId) => {
  return request({
    url: `/3dgs/status/${taskId}`,
    method: 'get'
  })
}

export const getModelList = () => {
  return request({
    url: '/3dgs/models',
    method: 'get'
  })
}

export const optimizeModel = (data) => {
  return request({
    url: '/3dgs/optimize',
    method: 'post',
    data
  })
}

export const checkOptimizeStatus = (taskId) => {
  return request({
    url: `/3dgs/optimize/status/${taskId}`,
    method: 'get'
  })
}

export const getModelInfo = (modelId) => {
  return request({
    url: `/3dgs/model/${modelId}`,
    method: 'get'
  })
}

export const deleteModel = (modelId) => {
  return request({
    url: `/3dgs/model/${modelId}`,
    method: 'delete'
  })
}
