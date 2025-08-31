#!/usr/bin/env python3
"""
简易端口清理脚本
用法: python kill_port.py [端口号]
默认清理8000端口
"""

import subprocess
import sys
import time


def kill_port(port=8000):
    """强制关闭指定端口上的所有进程"""
    print(f"🔥 正在清理端口 {port}...")
    
    try:
        # 查找占用端口的进程
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            print(f"📋 发现 {len(pids)} 个进程占用端口 {port}")
            
            # 强制杀死所有进程
            for pid in pids:
                if pid:
                    try:
                        subprocess.run(["kill", "-9", pid], check=True)
                        print(f"✅ 已杀死进程 {pid}")
                    except subprocess.CalledProcessError:
                        print(f"❌ 无法杀死进程 {pid}")
        else:
            print(f"✅ 端口 {port} 没有被占用")
            return True
        
        # 等待并验证
        time.sleep(1)
        
        # 验证端口是否释放
        verify_result = subprocess.run(
            ["lsof", "-ti", f":{port}"], 
            capture_output=True, 
            text=True
        )
        
        if verify_result.returncode == 0 and verify_result.stdout.strip():
            print(f"❌ 端口 {port} 仍被占用")
            return False
        else:
            print(f"🎉 端口 {port} 已成功释放!")
            return True
            
    except Exception as e:
        print(f"❌ 清理端口时出错: {e}")
        return False

def main():
    # 获取端口号参数
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("❌ 端口号必须是数字")
            sys.exit(1)
    else:
        port = 8000
    
    # 执行清理
    success = kill_port(port)
    
    if success:
        print(f"\n🚀 现在可以使用端口 {port} 了!")
        print(f"启动命令: python -m uvicorn app.api:app --host 0.0.0.0 --port {port}")
    else:
        print(f"\n😅 建议使用其他端口:")
        print(f"python -m uvicorn app.api:app --host 0.0.0.0 --port {port + 1}")

if __name__ == "__main__":
    main()
