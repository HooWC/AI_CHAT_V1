from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ChineseChatbot:
    def __init__(self):
        # 使用阿里巴巴的 Qwen2.5 0.5B 指令微调版
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"正在加载中文模型 {self.model_name}...")
        print("模型大小：约 950MB，初次加载可能需要一点时间...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto" # 自动检测 GPU 或 CPU
        )
        
        # 存储对话历史
        self.messages = [
            {"role": "system", "content": "你是一个乐于助人的中文 AI 助手。"}
        ]
        
        print("✅ 中文模型加载完成！")

    def chat(self, user_input):
        # 1. 将用户输入添加到历史记录
        self.messages.append({"role": "user", "content": user_input})
        
        # 2. 使用模板处理对话格式
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 3. 编码输入
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # 4. 生成回复
        # 注意：这里解决了你之前遇到的 attention_mask 和 pad_token 问题
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1
        )
        
        # 5. 提取新生成的 token
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        # 6. 解码并保存到历史记录
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        self.messages.append({"role": "assistant", "content": response})
        
        return response

    def clear_history(self):
        self.messages = [{"role": "system", "content": "你是一个乐于助人的中文 AI 助手。"}]
        return "对话历史已清空"

def main():
    print("=" * 60)
    print("🤖 免费中文对话机器人 - Qwen2.5-0.5B")
    print("=" * 60)
    print("✓ 完全免费 | ✓ 支持中文 | ✓ 本地运行")
    print("-" * 60)
    
    bot = ChineseChatbot()
    
    print("\n💬 开始对话（输入 '退出' 结束，'clear' 清空历史）")
    
    while True:
        user_input = input("\n👤 你: ").strip()
        
        if user_input.lower() in ['退出', 'quit', 'exit']:
            print("🤖 AI: 再见！祝你开心每一天！ 👋")
            break
        elif user_input.lower() == 'clear':
            result = bot.clear_history()
            print(f"🤖 AI: {result}")
            continue
        
        if not user_input:
            continue
        
        try:
            response = bot.chat(user_input)
            print(f"🤖 AI: {response}")
        except Exception as e:
            print(f"❌ 出错啦: {e}")

if __name__ == "__main__":
    main()