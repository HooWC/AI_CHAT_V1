import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SuperChatbot:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype="auto", device_map="auto"
        )
        self.gen_kwargs = {"max_new_tokens": 512, "do_sample": True, "temperature": 0.7, "top_p": 0.9}

    def chat(self, user_input, history):
        # history 是前端传回来的数组
        messages = [{"role": "system", "content": "你是一个有用的 AI 助手。"}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_input})

        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        if self.device == "cuda":
            torch.cuda.empty_cache()

        generated_ids = self.model.generate(**model_inputs, **self.gen_kwargs)
        response = self.tokenizer.decode(generated_ids[0][model_inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        return response