import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig

class FastNovelWriter:
    def __init__(self):
        # 依然使用 1.5B 效果较好，如果追求极致速度可以换回 0.5B
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct" 
        print(f"🚀 正在以加速模式加载引擎: {self.model_name}...")

        # 1. 配置 4-bit 量化，这是提速的关键
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 # 如果显卡不支持bf16，改为torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # 2. 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config, # 应用量化
            device_map="auto",             # 自动分配显存/内存
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # 3. 开启推理加速（针对支持的算子）
        # self.model = self.model.to_bettertransformer() # 可选，视环境而定

        self.system_prompt = (
            "你是一位顶级的网文大神，擅长细腻的心理描写、环境渲染和慢节奏的情节铺陈。\n"
            "【规则】：严禁跳过剧情，禁止做总结性陈述，每一章必须包含大量的细节描写，节奏要慢。"
        )
        self.messages = []

    def write_long_chapter(self, prompt, target_length=1500):
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.messages.append({"role": "user", "content": f"请开始创作小说：{prompt}。注意：细节要丰富，不要急于完结。"})
        
        full_story = ""
        print(f"\n⚡ 高速模式开启，目标字数：{target_length}...")

        while len(full_story) < target_length:
            # 使用 apply_chat_template 构建输入
            text = self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # 使用流式输出，边写边看就不会觉得慢了
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            # 生成配置优化
            with torch.no_grad(): # 禁用梯度计算，省内存提速
                generated_ids = self.model.generate(
                    **model_inputs,
                    streamer=streamer,
                    max_new_tokens=512, # 减小单次生成长度，保持推理高效
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    use_cache=True # 务必开启缓存，这是提速核心
                )
            
            response_ids = generated_ids[0][model_inputs.input_ids.shape[-1]:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            full_story += response_text
            self.messages.append({"role": "assistant", "content": response_text})
            
            if len(full_story) < target_length:
                self.messages.append({"role": "user", "content": "请紧接上文，继续详细描写情节。"})
            else:
                break
        
        return full_story

def main():
    writer = FastNovelWriter()
    while True:
        user_topic = input("\n👤 输入主题: ").strip()
        if user_topic.lower() == 'exit': break
        writer.write_long_chapter(user_topic, target_length=1500)

if __name__ == "__main__":
    main()