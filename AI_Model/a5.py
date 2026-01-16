import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

class NovelProWriter:
    def __init__(self):
        # 建议至少使用 1.5B 模型，0.5B 的逻辑链太短，很难写长文不跑题
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct" 
        print(f"正在加载专业创作引擎: {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # 核心：极其详细的系统设定，强迫 AI 放弃“总结式”写法，改用“描写式”写法
        self.system_prompt = (
            "你是一位顶级的网文大神，擅长细腻的心理描写、环境渲染和慢节奏的情节铺陈。\n"
            "【规则】：\n"
            "1. 严禁跳过剧情，禁止做总结性陈述（如“他们经历了一场激战”是错误的，必须写具体的动作和对话）。\n"
            "2. 每一章必须包含大量的环境细节描写和人物内心活动。\n"
            "3. 节奏要慢，语言要优美且有感染力。\n"
            "4. 如果故事没写完，请在结尾留下伏笔。"
        )
        self.messages = []

    def write_long_chapter(self, prompt, target_length=1500):
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.messages.append({"role": "user", "content": f"请开始创作小说：{prompt}。注意：请先写第一部分，细节要丰富，不要急于完结。"})
        
        full_story = ""
        current_step = 1
        
        print(f"\n🚀 开始创作长篇章节，目标字数：{target_length}...")

        while len(full_story) < target_length:
            print(f"\n--- 正在创作第 {current_step} 段 (当前总字数: {len(full_story)}) ---")
            
            # 构建输入
            text = self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            # 生成这一段
            generated_ids = self.model.generate(
                **model_inputs,
                streamer=streamer,
                max_new_tokens=800, # 每次生成的中段长度
                do_sample=True,
                temperature=0.9,     # 略高一点增加文采
                top_p=0.95,
                repetition_penalty=1.15
            )
            
            response_ids = generated_ids[0][model_inputs.input_ids.shape[-1]:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # 拼接到全文
            full_story += response_text
            self.messages.append({"role": "assistant", "content": response_text})
            
            # 检查字数，如果不够，自动追加指令
            if len(full_story) < target_length:
                self.messages.append({"role": "user", "content": "请继续紧接上文描写，保持细节丰富，不要跳跃剧情，继续写。"})
                current_step += 1
            else:
                break
        
        print(f"\n✅ 章节创作完成！总字数：{len(full_story)}")
        return full_story

def main():
    writer = NovelProWriter()
    while True:
        user_topic = input("\n👤 输入小说主题或开头: ").strip()
        if user_topic.lower() == 'exit': break
        
        # 设定目标字数为 1500
        chapter_content = writer.write_long_chapter(user_topic, target_length=1500)
        
        save_yn = input("\n💾 是否保存到文件？(y/n): ")
        if save_yn.lower() == 'y':
            with open("novel_chapter.txt", "w", encoding="utf-8") as f:
                f.write(chapter_content)
            print("📁 已保存至 novel_chapter.txt")

if __name__ == "__main__":
    main()