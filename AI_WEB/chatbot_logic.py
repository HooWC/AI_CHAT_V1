import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

class SuperChatbot:
    def __init__(self):
        # 降级到 0.5B，这是目前能跑的最轻量且有智商的版本
        self.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        
        print(f"🚀 正在启动轻量版引擎 (Qwen2.5-0.5B)...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # 强制 CPU 运行，且关闭所有不必要的加载项
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map={"": "cpu"} 
        )
        print("✅ 引擎启动成功！现在系统应该非常流畅。")

    def chat_stream(self, user_input, history):
        # 系统提示词稍微加强，弥补模型参数小的不足
        messages = [{"role": "system", "content": "你是一个简明扼要、专业的 AI 助手。"}]
        # 0.5B 记不住太长的东西，只保留最近 2 轮对话
        messages.extend(history[-4:]) 
        messages.append({"role": "user", "content": user_input})

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt")

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=300, # 缩短单次回复长度，进一步提升速度
            do_sample=True,
            temperature=0.7,
            top_p=0.8
        )

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text