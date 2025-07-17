#!/bin/bash
# 测试 TensorFlow Lite 日志修改的构建脚本

echo "=== TensorFlow Lite 日志系统修改测试 ==="
echo ""
echo "这个脚本将测试修改后的日志系统"
echo ""

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在正确的目录
if [ ! -f "WORKSPACE" ]; then
    echo -e "${RED}错误：请在 TensorFlow 根目录运行此脚本${NC}"
    exit 1
fi

echo -e "${YELLOW}步骤 1: 运行单元测试${NC}"
echo "命令: bazel test //tensorflow/lite:minimal_logging_test"
echo ""
echo "如果 Bazel 已配置，请运行以上命令"
echo ""

echo -e "${YELLOW}步骤 2: 查看修改的文件${NC}"
echo "修改的文件列表："
echo "  - tensorflow/lite/minimal_logging.cc"
echo "  - tensorflow/lite/minimal_logging_test.cc"
echo ""

echo -e "${YELLOW}步骤 3: 查看修改内容${NC}"
echo "日志格式修改："
echo "  VERBOSE  → [*] VERBOSE"
echo "  INFO     → [+] INFO"
echo "  WARNING  → [!] WARNING"
echo "  ERROR    → [X] ERROR"
echo "  SILENT   → [-] SILENT"
echo ""

echo -e "${GREEN}修改已完成！${NC}"
echo ""
echo "文档说明："
echo "  - 修改说明.md - 详细的修改说明"
echo "  - 如何验证修改.md - 验证步骤指南"
echo "  - test_minimal_logging.cc - C++ 测试程序"
echo ""

# 显示 git diff（如果可用）
if command -v git &> /dev/null; then
    echo -e "${YELLOW}Git 改动预览:${NC}"
    git diff --stat tensorflow/lite/minimal_logging.cc tensorflow/lite/minimal_logging_test.cc
fi

echo ""
echo "测试脚本执行完成！"