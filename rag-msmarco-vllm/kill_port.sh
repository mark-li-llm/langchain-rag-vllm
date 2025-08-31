#!/bin/bash
# 简易端口清理脚本
# 用法: ./kill_port.sh [端口号]
# 默认清理8000端口

PORT=${1:-8000}

echo "🔥 正在清理端口 $PORT..."

# 查找并杀死占用端口的进程
PIDS=$(lsof -ti:$PORT 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "✅ 端口 $PORT 没有被占用"
else
    echo "📋 发现进程占用端口 $PORT，正在清理..."
    echo "$PIDS" | xargs kill -9 2>/dev/null
    sleep 1
    
    # 验证是否清理成功
    if lsof -ti:$PORT >/dev/null 2>&1; then
        echo "❌ 端口 $PORT 仍被占用，尝试更强力清理..."
        sudo lsof -ti:$PORT | xargs sudo kill -9 2>/dev/null
        sleep 1
    fi
    
    if lsof -ti:$PORT >/dev/null 2>&1; then
        echo "😅 端口 $PORT 仍被占用，建议重启系统或使用其他端口"
        echo "建议命令: python -m uvicorn app.api:app --host 0.0.0.0 --port $((PORT + 1))"
    else
        echo "🎉 端口 $PORT 已成功释放!"
    fi
fi

echo ""
echo "🚀 现在可以启动应用了:"
echo "conda activate rag"
echo "python -m uvicorn app.api:app --host 0.0.0.0 --port $PORT"
