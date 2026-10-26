#!/bin/bash
# 5张照片3DGS训练脚本

set -e

API_URL="http://192.168.10.148:12524"
API_KEY="${API_KEY:?请先设置 API_KEY 环境变量}"

if [ $# -ne 1 ]; then
    echo "用法: $0 <photos.zip>"
    echo "示例: $0 /path/to/photos.zip"
    exit 1
fi

PHOTOS_ZIP="$1"

if [ ! -f "$PHOTOS_ZIP" ]; then
    echo "错误: 文件不存在: $PHOTOS_ZIP"
    exit 1
fi

echo "==> 1. 提交训练任务..."
RESPONSE=$(curl -s -X POST "$API_URL/train" \
  -H "X-API-Key: $API_KEY" \
  -F "file=@$PHOTOS_ZIP" \
  -F "sfm_backend=mast3r" \
  -F "iterations=3000" \
  -F "downsample=2" \
  -F "cap_max=500000")

TASK_ID=$(echo "$RESPONSE" | grep -o '"task_id":"[^"]*"' | cut -d'"' -f4)

if [ -z "$TASK_ID" ]; then
    echo "错误: 任务提交失败"
    echo "$RESPONSE"
    exit 1
fi

echo "任务ID: $TASK_ID"
echo ""

echo "==> 2. 等待训练完成..."
while true; do
    STATUS_JSON=$(curl -s -H "X-API-Key: $API_KEY" "$API_URL/train/$TASK_ID")
    STATUS=$(echo "$STATUS_JSON" | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
    PROGRESS=$(echo "$STATUS_JSON" | grep -o '"progress":[0-9.]*' | cut -d':' -f2)
    MESSAGE=$(echo "$STATUS_JSON" | grep -o '"message":"[^"]*"' | cut -d'"' -f4)

    if [ "$STATUS" = "done" ]; then
        echo ""
        echo "✓ 训练完成！"
        break
    elif [ "$STATUS" = "failed" ]; then
        ERROR=$(echo "$STATUS_JSON" | grep -o '"error":"[^"]*"' | cut -d'"' -f4)
        echo ""
        echo "✗ 训练失败: $ERROR"
        exit 1
    fi

    PROGRESS_PCT=$(echo "$PROGRESS * 100" | bc 2>/dev/null || echo "0")
    printf "\r进度: %.1f%% | %s | %s          " "$PROGRESS_PCT" "$STATUS" "$MESSAGE"
    sleep 3
done

echo ""
echo "==> 3. 下载结果..."
OUTPUT_FILE="3dgs_${TASK_ID}.zip"
curl -H "X-API-Key: $API_KEY" \
  -o "$OUTPUT_FILE" \
  "$API_URL/train/$TASK_ID/result"

echo "✓ 结果已保存到: $OUTPUT_FILE"
echo ""

# 解压查看内容
echo "==> 4. 结果文件:"
unzip -l "$OUTPUT_FILE"

echo ""
echo "完成！可以使用以下工具查看模型:"
echo "  - SuperSplat: https://playcanvas.com/supersplat/editor"
echo "  - Luma AI: https://lumalabs.ai/interactive-scenes"
echo "  - Three.js 查看器: https://github.com/mkkellogg/GaussianSplats3D"
