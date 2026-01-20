import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sentence_transformers import SentenceTransformer
import numpy as np
from threading import Thread


class HSGEngineRAG:
    """
    使用 RAG（检索增强生成）+ 向量数据库
    优点：语义搜索，更智能的上下文理解
    
    安装依赖：
    pip install sentence-transformers
    """
    
    def __init__(self):
        self.model_id = "Qwen/Qwen2.5-0.5B-Instruct"

        print(f"🚀 正在启动 HSG RAG 智能引擎...")

        # 加载嵌入模型（用于语义搜索）
        print("📊 加载嵌入模型...")
        try:
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ 嵌入模型加载成功")
        except Exception as e:
            print(f"⚠️ 嵌入模型加载失败: {e}")
            print("请运行: pip install sentence-transformers")
            self.embed_model = None
        
        # 读取知识库（统一从 knowledge_base.json + .txt 构建 chunks）
        self.knowledge_chunks = []
        self.chunk_embeddings = None
        self._load_knowledge_base()
        
        # 加载对话模型
        print("🤖 加载对话模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float32,
            device_map={"": "cpu"},
        )

        # 从 JSON 中抽取关键信息，用于系统提示
        kb_dir = "knowledge"
        kb_json_path = os.path.join(kb_dir, "knowledge_base.json")
        kb_json = None
        if os.path.exists(kb_json_path):
            try:
                with open(kb_json_path, "r", encoding="utf-8") as f:
                    kb_json = json.load(f)
            except Exception as e:
                print(f"⚠️ RAG 引擎加载 knowledge_base.json 失败: {e}")

        company_name = (
            kb_json.get("company_info", {}).get("name", "Hong Seng Group") if kb_json else "Hong Seng Group"
        )
        company_cn_name = (
            kb_json.get("company_info", {}).get("chinese_name", "丰成集团")
            if kb_json
            else "丰成集团"
        )
        important_note = (
            kb_json.get("company_info", {}).get("important_note", "我们是马来西亚的公司，不是香港的公司")
            if kb_json
            else "我们是马来西亚的公司，不是香港的公司"
        )

        # 基础系统提示词
        self.base_system_prompt = (
            f"你现在是马来西亚 {company_name} ({company_cn_name}) 的专属 IT 助手。\n"
            "【绝对指令】回答必须完全基于提供的上下文内容（包括本地知识库检索结果）。\n"
            f"1. 身份：{important_note}。\n"
            "2. 语气：简短、直接、专业。\n"
        )

        print("✅ RAG 引擎启动完成！")
    
    def _load_knowledge_base(self):
        """加载并向量化知识库：优先使用 knowledge_base.json，再用 .txt 补充"""
        kb_dir = "knowledge"

        if not os.path.exists(kb_dir):
            os.makedirs(kb_dir)
            print("⚠️ 知识库文件夹不存在，已创建")
            return

        # 1) 从 JSON 构建 chunks（主数据源）
        kb_json_path = os.path.join(kb_dir, "knowledge_base.json")
        if os.path.exists(kb_json_path):
            try:
                with open(kb_json_path, "r", encoding="utf-8") as f:
                    kb_json = json.load(f)

                # 公司信息 / IT 支持 / FAQ / quick_links 展开为多个 chunk
                self._add_json_chunks(kb_json)
                print("✅ RAG 已从 knowledge_base.json 构建语义检索知识库。")
            except Exception as e:
                print(f"⚠️ 加载 knowledge_base.json 失败，退回到 TXT 模式: {e}")

        # 2) 用 TXT 作为补充
        txt_files = [f for f in os.listdir(kb_dir) if f.endswith(".txt")]
        for filename in txt_files:
            filepath = os.path.join(kb_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()

                if not content:
                    continue

                # 将内容分成小块（chunk）
                chunks = self._split_into_chunks(content, chunk_size=200)

                for chunk in chunks:
                    self.knowledge_chunks.append(
                        {
                            "source": filename,
                            "text": chunk,
                        }
                    )

        print(f"✅ 已载入补充 TXT 文档 {len(txt_files)} 个，总知识块数量: {len(self.knowledge_chunks)}")

        # 生成嵌入向量
        if self.embed_model and self.knowledge_chunks:
            print("🔄 正在生成向量嵌入...")
            texts = [chunk["text"] for chunk in self.knowledge_chunks]
            self.chunk_embeddings = self.embed_model.encode(texts, convert_to_numpy=True)
            print("✅ 向量嵌入生成完成")
    
    def _split_into_chunks(self, text, chunk_size=200):
        """将长文本分割成小块"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks if chunks else [text]

    def _add_json_chunks(self, kb_json: dict):
        """把 knowledge_base.json 各部分展开成适合检索的文本块"""
        # 公司信息
        company = kb_json.get("company_info", {})
        if company:
            text = (
                f"公司名称: {company.get('name', '')}（{company.get('chinese_name', '')}）\n"
                f"国家: {company.get('country', '')}\n"
                f"地址: {company.get('full_address', '')}\n"
                f"简介: {company.get('description', '')}\n"
                f"重要说明: {company.get('important_note', '')}"
            )
            self.knowledge_chunks.append({"source": "knowledge_base.json:company_info", "text": text})

        # IT 支持
        it = kb_json.get("it_support", {})
        if it:
            text = (
                f"IT 文档路径: {it.get('documents_path', '')}\n"
                f"文档说明: {it.get('documents_description', '')}\n"
                f"ITSM 地址: {it.get('itsm_server', '')}\n"
                f"Node API 地址: {it.get('node_api', '')}\n"
                f"支持说明: {it.get('support_note', '')}\n"
                f"联系方式: {it.get('contact', '')}"
            )
            self.knowledge_chunks.append({"source": "knowledge_base.json:it_support", "text": text})

        # 部门
        departments = kb_json.get("departments", {})
        for key, dep in departments.items():
            text = (
                f"部门标识: {key}\n"
                f"名称: {dep.get('name', '')}\n"
                f"职责: {dep.get('role', '')}\n"
                f"服务: {', '.join(dep.get('services', []))}"
            )
            self.knowledge_chunks.append({"source": f"knowledge_base.json:departments:{key}", "text": text})

        # FAQ
        faq_list = kb_json.get("faq", [])
        for item in faq_list:
            text = (
                f"FAQ 问题: {item.get('question', '')}\n"
                f"回答: {item.get('answer', '')}\n"
                f"分类: {item.get('category', '')}\n"
                f"重要级别: {item.get('priority', '')}\n"
                f"关键词: {', '.join(item.get('keywords', []))}"
            )
            self.knowledge_chunks.append({"source": "knowledge_base.json:faq", "text": text})

        # 快速链接
        quick_links = kb_json.get("quick_links", {})
        if quick_links:
            text_lines = [f"{k}: {v}" for k, v in quick_links.items()]
            text = "快速链接:\n" + "\n".join(text_lines)
            self.knowledge_chunks.append({"source": "knowledge_base.json:quick_links", "text": text})
    
    def _semantic_search(self, query, top_k=3):
        """语义搜索最相关的知识块"""
        if not self.embed_model or self.chunk_embeddings is None or len(self.knowledge_chunks) == 0:
            return []
        
        # 对查询进行编码
        query_embedding = self.embed_model.encode([query], convert_to_numpy=True)[0]
        
        # 计算余弦相似度
        similarities = np.dot(self.chunk_embeddings, query_embedding)
        
        # 获取 top_k 个最相似的
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # 相似度阈值
                results.append({
                    "text": self.knowledge_chunks[idx]["text"],
                    "source": self.knowledge_chunks[idx]["source"],
                    "score": float(similarities[idx])
                })
        
        return results
    
    def chat_stream(self, user_input, history):
        # 使用语义搜索找到最相关的内容
        relevant_chunks = self._semantic_search(user_input, top_k=3)
        
        # 构建上下文
        context = ""
        if relevant_chunks:
            context = "\n\n【相关知识库内容】\n"
            for i, chunk in enumerate(relevant_chunks, 1):
                context += f"\n[来源: {chunk['source']}]\n{chunk['text']}\n"
        
        # 身份强化
        extra_remind = ""
        identity_keywords = ["你是谁", "哪里的", "什么公司", "who are you", "where"]
        if any(k in user_input.lower() for k in identity_keywords):
            extra_remind = "\n(特别提醒：你是马来西亚的 HSG，不是香港的。)"
        
        # 构造完整的系统提示词
        full_system_prompt = self.base_system_prompt + context + extra_remind
        
        # 构造消息
        messages = [{"role": "system", "content": full_system_prompt}]
        
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
