#!/bin/bash
# 高质量训练参数配置

API_URL="http://192.168.10.148:12524"
API_KEY="${API_KEY:?请先设置 API_KEY 环境变量}"

echo "==> 提交高质量训练任务..."
curl -X POST "$API_URL/train" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@$1" \
  -F "sfm_backend=mast3r" \
  -F "iterations=5000" \
  -F "downsample=1" \
  -F "cap_max=800000"

echo ""
echo "优化点："
echo "  - iterations=5000 (从3000增加，更好收敛)"
echo "  - downsample=1 (不降采样，保留原始分辨率)"
echo "  - cap_max=800000 (增加高斯数量上限)"
echo ""
echo "注意：需要更多显存(预计4-5GB)和训练时间(预计4-5分钟)"
