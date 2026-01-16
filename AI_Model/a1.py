# blender_chat.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class BlenderChatbot:
    def __init__(self, model_name="facebook/blenderbot-400M-distill"):
        """
        BlenderBot是Facebook专门为对话设计的模型
        效果比普通语言模型好很多
        """
        print(f"正在加载 {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # 对话历史
        self.history = []
        
        print("✅ 模型加载完成！")
        print("💡 提示：这是英文对话模型，擅长闲聊")
    
    def chat(self, user_input):
        # 构建对话上下文
        if self.history:
            # BlenderBot使用特殊的格式
            history_text = " ".join([f"{speaker}: {text}" for speaker, text in self.history[-4:]])
            input_text = f"{history_text} Human: {user_input} Person:"
        else:
            input_text = f"Human: {user_input} Person:"
        
        # 编码输入
        inputs = self.tokenizer([input_text], return_tensors="pt", truncation=True)
        
        # 生成回复
        reply_ids = self.model.generate(
            **inputs,
            max_length=100,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
        
        response = self.tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
        
        # 提取回复
        if "Person:" in response:
            response = response.split("Person:")[-1].strip()
        
        # 更新历史
        self.history.append(("Human", user_input))
        self.history.append(("Person", response))
        
        # 保持历史长度
        if len(self.history) > 10:
            self.history = self.history[-10:]
        
        return response

def main():
    print("=" * 60)
    print("🤖 Facebook BlenderBot 对话机器人")
    print("=" * 60)
    print("特点：")
    print("• Facebook专门为对话训练")
    print("• 能进行有意义的对话")
    print("• 擅长闲聊和日常对话")
    print("• 英文模型，但效果极佳")
    print("-" * 60)
    
    bot = BlenderChatbot()
    
    print("\n💬 开始对话（输入 '退出' 或 'clear' 清空历史）")
    
    while True:
        user_input = input("\n👤 你: ").strip()
        
        if user_input.lower() == '退出':
            print("🤖 AI: 再见！")
            break
        elif user_input.lower() == 'clear':
            bot.history = []
            print("🤖 AI: 对话历史已清空")
            continue
        
        response = bot.chat(user_input)
        print(f"🤖 AI: {response}")

if __name__ == "__main__":
    main()