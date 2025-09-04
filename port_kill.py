#!/usr/bin/env python3
"""强制清理占用8000端口的所有进程"""

import os
import signal
import socket
import subprocess
import time


def force_kill_port_8000():
    print("🔍 强力查找并清理占用8000端口的所有进程...")

    # 方法1: 直接查找监听8000端口的进程
    try:
        # 在Linux系统中查找监听端口的进程
        result = subprocess.run(
            [
                "python",
                "-c",
                """
import os
import psutil
import signal

killed_count = 0
for proc in psutil.process_iter():
    try:
        connections = proc.connections(kind="inet")
        for conn in connections:
            if conn.laddr.port == 8000:
                print(f"发现占用8000端口的进程: PID={proc.pid}, 名称={proc.name()}")
                proc.kill()
                killed_count += 1
                print(f"  -> 已强制终止 PID {proc.pid}")
    except:
        pass

if killed_count == 0:
    print("未找到直接占用8000端口的进程")
else:
    print(f"共终止了 {killed_count} 个进程")
        """,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.stdout:
            print(result.stdout)

    except Exception as e:
        print(f"方法1失败: {e}")

    # 方法2: 查找所有相关Python进程
    print("\n🔍 查找所有Python uvicorn进程...")
    try:
        result = subprocess.run(
            [
                "python",
                "-c",
                """
import psutil
import os
import signal

killed_count = 0
for proc in psutil.process_iter():
    try:
        if "python" in proc.name().lower():
            cmdline = " ".join(proc.cmdline())
            if "uvicorn" in cmdline and ("8000" in cmdline or "app.api" in cmdline):
                print(f"发现相关进程: PID={proc.pid}")
                print(f"  命令行: {cmdline[:80]}...")
                proc.kill()
                killed_count += 1
                print(f"  -> 已强制终止 PID {proc.pid}")
    except:
        pass

print(f"\\n共终止了 {killed_count} 个相关进程")
        """,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.stdout:
            print(result.stdout)

    except Exception as e:
        print(f"方法2失败: {e}")

    # 等待进程完全退出
    print("\n⏳ 等待进程完全退出...")
    time.sleep(3)

    # 最终验证
    print("\n✅ 验证端口状态...")
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 8000))
            print("🎉 端口8000现在完全可用！")
            return True
        except OSError:
            if attempt < max_attempts - 1:
                print(f"端口仍被占用，等待中... ({attempt + 1}/{max_attempts})")
                time.sleep(2)
            else:
                print("❌ 端口8000仍被占用，可能需要重启容器")
                return False


if __name__ == "__main__":
    success = force_kill_port_8000()
    if success:
        print("\n🚀 现在可以启动服务器了:")
        print("python -m uvicorn app.api:app --host 0.0.0.0 --port 8000")
    else:
        print("\n⚠️  如果问题持续，建议使用其他端口:")
        print("python -m uvicorn app.api:app --host 0.0.0.0 --port 8001")

