import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

class SuperChatbot:
    def __init__(self):
        # 如果内存允许，建议换成 "Qwen/Qwen2.5-1.5B-Instruct" 效果好非常多
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"正在加载引擎: {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # 默认模式
        self.mode = "assistant"
        self.messages = []
        self.reset_history()

    def reset_history(self):
        """重置对话，根据模式设定不同的系统提示词"""
        if self.mode == "novel":
            prompt = "你是一位精通各种风格的小说家。请根据用户的要求创作情节丰富、描写生动、逻辑自洽的小说。"
        else:
            prompt = "你是一个通晓百科、乐于助人的中文 AI 助手。"
        
        self.messages = [{"role": "system", "content": prompt}]

    def chat(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        
        # 构建输入
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # 实例化流式器，让文字在 CMD 里一个一个蹦出来
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        print("🤖 AI: ", end="", flush=True)
        
        # 生成参数优化
        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=1024, # 允许生成更长的文本
            do_sample=True,      # 必须开启采样才能写小说
            temperature=0.85,    # 提高随机性，避免由于太死板导致的词穷
            top_p=0.9,
            repetition_penalty=1.1 # 稍微加大惩罚，防止写小说时反复重复一段话
        )

        # 启动生成
        generated_ids = self.model.generate(**generation_kwargs)
        
        # 获取回复内容存入历史
        response = self.tokenizer.decode(generated_ids[0][model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        self.messages.append({"role": "assistant", "content": response})

def main():
    bot = SuperChatbot()
    print("\n[指令说明]: 输入 'novel' 进入小说模式 | 'chat' 回到问答模式 | 'clear' 重置")
    
    while True:
        mode_str = f"[{'写作' if bot.mode=='novel' else '助手'}]"
        user_input = input(f"\n👤 {mode_str} 你: ").strip()
        
        if not user_input: continue
        
        if user_input.lower() == 'exit': break
        if user_input.lower() == 'novel':
            bot.mode = "novel"
            bot.reset_history()
            print("✨ 已切换到小说创作模式！请输入你的小说开头或设定。")
            continue
        if user_input.lower() == 'chat':
            bot.mode = "assistant"
            bot.reset_history()
            print("💡 已回到普通问答模式。")
            continue
        if user_input.lower() == 'clear':
            bot.reset_history()
            print("🧹 记忆已清空。")
            continue

        bot.chat(user_input)

if __name__ == "__main__":
    main()