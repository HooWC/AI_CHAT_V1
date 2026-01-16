import subprocess
import sys

def install_packages():
    packages = [
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "sentencepiece",
        "protobuf",
        "accelerate",  # 加速推理
        "bitsandbytes"  # 量化支持，减少内存使用
    ]
    
    for package in packages:
        print(f"正在安装 {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("所有包安装完成！")

if __name__ == "__main__":
    install_packages()