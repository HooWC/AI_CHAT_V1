import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

class HSGEngineJSON:
    """
    使用 JSON 格式的结构化知识库
    优点：更容易管理和更新，支持分类和优先级
    """
    
    def __init__(self):
        self.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        
        # 读取 JSON 知识库
        self.knowledge_file = "knowledge/knowledge_base.json"
        self.knowledge_data = {}
        
        if os.path.exists(self.knowledge_file):
            with open(self.knowledge_file, "r", encoding="utf-8") as f:
                self.knowledge_data = json.load(f)
            print(f"✅ 成功载入 JSON 知识库")
        else:
            print(f"⚠️ 未找到 {self.knowledge_file}，将创建示例文件")
            self._create_sample_json()
        
        # 构建知识库文本
        self.combined_knowledge = self._build_knowledge_text()
        
        # 加载模型
        print(f"🚀 正在启动 HSG JSON 智能引擎...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map={"": "cpu"}
        )
        
        # 系统提示词
        self.system_prompt = (
            "你现在是马来西亚 Hong Seng Group (HSG) 的专属 IT 助手。\n"
            "【绝对指令】回答必须完全基于以下提供的【公司本地知识库】内容。\n"
            "1. 身份：你是马来西亚的丰成集团，绝不是香港公司，也不是通义千问。\n"
            "2. 语气：简短、直接、专业，不要废话。\n\n"
            f"--- 公司知识库 ---\n{self.combined_knowledge}\n--- 结束 ---"
        )
    
    def _create_sample_json(self):
        """创建示例 JSON 文件"""
        sample_data = {
            "company_info": {
                "name": "Hong Seng Group",
                "chinese_name": "丰成集团",
                "country": "Malaysia",
                "location": "Bandar Baru Bangi, Selangor, Malaysia",
                "description": "马来西亚综合性大集团"
            },
            "it_support": {
                "documents_path": "\\\\192.1.1.30:2828\\IT_Documents",
                "itsm_server": "https://192.1.1.30:2828",
                "node_api": "https://192.1.1.30:2828",
                "support_note": "如有技术问题，请访问上述地址获取帮助"
            },
            "faq": [
                {
                    "category": "IT",
                    "question": "IT部门的文档存放在哪里？",
                    "answer": "IT部门的文档存储在 \\\\192.1.1.30:2828\\IT_Documents",
                    "priority": "high"
                },
                {
                    "category": "ITSM",
                    "question": "ITSM系统地址是什么？",
                    "answer": "ITSM系统地址：https://192.1.1.30:2828",
                    "priority": "high"
                },
                {
                    "category": "Company",
                    "question": "Hong Seng Group在哪里？",
                    "answer": "Hong Seng Group总部位于马来西亚（Malaysia），不是香港公司",
                    "priority": "high"
                }
            ]
        }
        
        os.makedirs("knowledge", exist_ok=True)
        with open(self.knowledge_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=4)
        
        self.knowledge_data = sample_data
        print(f"✅ 已创建示例 JSON 知识库文件")
    
    def _build_knowledge_text(self):
        """将 JSON 数据转换为文本格式"""
        text_parts = []
        
        # 公司信息
        if "company_info" in self.knowledge_data:
            text_parts.append("【公司基本信息】")
            for key, value in self.knowledge_data["company_info"].items():
                text_parts.append(f"{key}: {value}")
        
        # IT 支持信息
        if "it_support" in self.knowledge_data:
            text_parts.append("\n【IT支持信息】")
            for key, value in self.knowledge_data["it_support"].items():
                text_parts.append(f"{key}: {value}")
        
        # FAQ
        if "faq" in self.knowledge_data:
            text_parts.append("\n【常见问题解答】")
            for item in self.knowledge_data["faq"]:
                priority = item.get("priority", "normal")
                category = item.get("category", "General")
                text_parts.append(f"\n[{category}] [{priority.upper()}] Q: {item['question']}")
                text_parts.append(f"A: {item['answer']}")
        
        return "\n".join(text_parts)
    
    def search_faq(self, user_input):
        """在FAQ中搜索相关问题"""
        if "faq" not in self.knowledge_data:
            return None
        
        user_lower = user_input.lower()
        best_match = None
        
        for item in self.knowledge_data["faq"]:
            question = item["question"].lower()
            # 简单的关键词匹配
            if any(word in user_lower for word in question.split()):
                if best_match is None or item.get("priority") == "high":
                    best_match = item
        
        return best_match
    
    def chat_stream(self, user_input, history):
        # 先尝试在FAQ中查找
        faq_match = self.search_faq(user_input)
        extra_context = ""
        if faq_match:
            extra_context = f"\n\n【最相关的FAQ】\nQ: {faq_match['question']}\nA: {faq_match['answer']}"
        
        # 身份强化
        extra_remind = ""
        identity_keywords = ["你是谁", "哪里的", "什么公司", "who are you", "where"]
        if any(k in user_input.lower() for k in identity_keywords):
            extra_remind = "\n(特别提醒：请再次确认你是马来西亚的 HSG，不是香港的。)"
        
        # 构造消息
        messages = [{"role": "system", "content": self.system_prompt + extra_context + extra_remind}]
        
        for h in history[-6:]:
            messages.append(h)
        
        messages.append({"role": "user", "content": user_input})
        
        # 编码输入
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generate_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.8,
            repetition_penalty=1.2
        )
        
        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()
        
        for new_text in streamer:
            yield new_text
