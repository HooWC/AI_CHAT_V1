# a2_fixed.py
from transformers import AutoTokenizer, AutoModel
import torch

class ChineseChatbot:
    def __init__(self, model_name="THUDM/chatglm3-6b"):
        """
        ChatGLM3-6B是清华大学的开源中文对话模型
        专门为中文对话优化，效果非常好
        """
        print(f"正在加载 {model_name}...")
        print("⚠️ 注意：首次下载需要较长时间（约12GB）")
        
        # 修复：使用 dtype 替代 torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # 根据硬件自动选择数据类型
        if torch.cuda.is_available():
            print("✅ 检测到CUDA，使用GPU加速")
            dtype = torch.float16  # 半精度节省显存
        else:
            print("⚠️ 使用CPU模式（较慢）")
            dtype = torch.float32
        
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",  # 自动选择GPU/CPU
            dtype=dtype,  # 修复：使用 dtype
            low_cpu_mem_usage=True  # 减少CPU内存使用
        ).eval()
        
        self.history = []
        print("✅ 模型加载完成！")
        print("💡 提示：这是专门的中文对话模型，支持多轮对话")
    
    def chat(self, user_input, max_length=4096):
        # 使用ChatGLM的内置对话接口
        response, self.history = self.model.chat(
            self.tokenizer,
            user_input,
            history=self.history,
            max_length=max_length,
            temperature=0.7
        )
        
        # 保持历史长度
        if len(self.history) > 20:
            self.history = self.history[-20:]
        
        return response
    
    def clear_history(self):
        self.history = []
        return "对话历史已清空"

def main():
    print("=" * 60)
    print("🤖 中文对话机器人 - ChatGLM3-6B")
    print("=" * 60)
    print("特点：")
    print("• 清华大学开发，专门为中文优化")
    print("• 支持上下文理解（多轮对话）")
    print("• 代码、数学、推理能力强")
    print("• 完全免费开源")
    print("-" * 60)
    
    try:
        bot = ChineseChatbot()
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        print("\n💡 建议：")
        print("1. 如果不想下载12GB大模型，请按 Ctrl+C 中断")
        print("2. 使用下面的轻量级替代方案")
        return
    
    print("\n💬 开始对话（输入 '退出' 结束，'clear' 清空历史）")
    
    while True:
        try:
            user_input = input("\n👤 你: ").strip()
        except KeyboardInterrupt:
            print("\n🤖 AI: 再见！")
            break
        
        if user_input.lower() in ['退出', 'exit', 'quit']:
            print("🤖 AI: 再见！期待下次聊天！")
            break
        elif user_input.lower() in ['clear', '清空', '清除']:
            result = bot.clear_history()
            print(f"🤖 AI: {result}")
            continue
        
        try:
            response = bot.chat(user_input)
            print(f"🤖 AI: {response}")
        except Exception as e:
            print(f"❌ 生成回复时出错: {e}")

if __name__ == "__main__":
    main()