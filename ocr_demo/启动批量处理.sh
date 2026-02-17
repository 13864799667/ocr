#!/bin/bash

# 批量文件夹OCR处理启动脚本
# 用法: ./启动批量处理.sh <大文件夹路径>

echo "=========================================="
echo "  批量文件夹OCR处理脚本"
echo "=========================================="

# 检查参数
if [ -z "$1" ]; then
    echo ""
    echo "用法: ./启动批量处理.sh <大文件夹路径>"
    echo ""
    echo "示例:"
    echo "  ./启动批量处理.sh /home/lys/ocr_demo/女-166"
    echo ""
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 切换到脚本目录
cd "$SCRIPT_DIR"

# 检查必要文件
if [ ! -f "batch_folder_process.py" ]; then
    echo "错误: batch_folder_process.py 不存在"
    exit 1
fi

if [ ! -f "split_image.py" ]; then
    echo "错误: split_image.py 不存在"
    exit 1
fi

# 检查输入目录
if [ ! -d "$1" ]; then
    echo "错误: 输入目录不存在: $1"
    exit 1
fi

echo ""
echo "输入目录: $1"
echo "工作目录: $SCRIPT_DIR"
echo ""

# 运行批量处理脚本
python batch_folder_process.py "$1" --debug

echo ""
echo "处理完成！"
echo "查看日志: $SCRIPT_DIR/batch_process.log"
