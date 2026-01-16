try:
    from transformers import pipeline
    import torch
    print("✅ transformers 安装成功")
    print(f"✅ torch 版本: {torch.__version__}")
    print(f"✅ CUDA 是否可用: {torch.cuda.is_available()}")
except Exception as e:
    print(f"❌ 安装失败: {e}")