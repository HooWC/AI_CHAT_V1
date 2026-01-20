import torch
import os
import sqlite3
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from datetime import datetime

class HSGEngineDB:
    """
    使用 SQLite 数据库存储知识库
    优点：支持动态更新，可以集成现有系统，支持复杂查询
    """
    
    def __init__(self, db_path="knowledge/knowledge.db"):
        self.model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        self.db_path = db_path
        
        print(f"🚀 正在启动 HSG 数据库智能引擎...")
        
        # 初始化数据库
        self._init_database()
        
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map={"": "cpu"}
        )
        
        # 系统提示词
        self.base_system_prompt = (
            "你现在是马来西亚 Hong Seng Group (HSG) 的专属 IT 助手。\n"
            "【绝对指令】回答必须完全基于提供的知识库内容。\n"
            "1. 身份：你是马来西亚的丰成集团，绝不是香港公司。\n"
            "2. 语气：简短、直接、专业。\n"
        )
        
        print("✅ 数据库引擎启动完成！")
    
    def _init_database(self):
        """初始化数据库结构"""
        os.makedirs(os.path.dirname(self.db_path) if os.path.dirname(self.db_path) else ".", exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建知识库表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                title TEXT,
                content TEXT,
                keywords TEXT,
                priority INTEGER DEFAULT 5,
                created_at TEXT,
                updated_at TEXT
            )
        ''')
        
        # 创建FAQ表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faq (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                question TEXT,
                answer TEXT,
                priority INTEGER DEFAULT 5,
                view_count INTEGER DEFAULT 0,
                created_at TEXT
            )
        ''')
        
        # 检查是否有数据，如果没有就插入示例数据
        cursor.execute("SELECT COUNT(*) FROM knowledge_base")
        if cursor.fetchone()[0] == 0:
            self._insert_sample_data(cursor)
        
        conn.commit()
        conn.close()
        
        print(f"✅ 数据库初始化完成：{self.db_path}")
    
    def _insert_sample_data(self, cursor):
        """插入示例数据"""
        now = datetime.now().isoformat()
        
        # 公司信息
        cursor.execute('''
            INSERT INTO knowledge_base (category, title, content, keywords, priority, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            "Company",
            "Hong Seng Group 基本信息",
            "Hong Seng Group（丰成集团）是一家总部位于马来西亚（Malaysia）的综合性大集团。地址：Lot 53, Jalan 1/5, Seksyen 1, 43650 Bandar Baru Bangi, Selangor, Malaysia. 我们不是香港的公司。",
            "公司,马来西亚,地址,Malaysia",
            10,
            now,
            now
        ))
        
        # IT支持信息
        cursor.execute('''
            INSERT INTO knowledge_base (category, title, content, keywords, priority, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            "IT",
            "IT文档和系统地址",
            "IT部门的文档存储在 \\\\192.1.1.30:2828\\IT_Documents。ITSM系统和Node API都在 https://192.1.1.30:2828",
            "IT,文档,ITSM,服务器,地址",
            10,
            now,
            now
        ))
        
        # FAQ
        cursor.execute('''
            INSERT INTO faq (category, question, answer, priority, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            "IT",
            "IT部门的文档存放在哪里？",
            "IT部门的文档存储在 \\\\192.1.1.30:2828\\IT_Documents",
            10,
            now
        ))
        
        cursor.execute('''
            INSERT INTO faq (category, question, answer, priority, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            "ITSM",
            "ITSM系统的地址是什么？",
            "ITSM系统地址是 https://192.1.1.30:2828",
            10,
            now
        ))
        
        print("✅ 已插入示例数据")
    
    def _search_knowledge(self, query):
        """在数据库中搜索相关知识"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 简单的关键词搜索
        keywords = query.split()
        results = []
        
        for keyword in keywords:
            # 搜索知识库
            cursor.execute('''
                SELECT category, title, content, priority 
                FROM knowledge_base 
                WHERE content LIKE ? OR keywords LIKE ? OR title LIKE ?
                ORDER BY priority DESC
                LIMIT 3
            ''', (f"%{keyword}%", f"%{keyword}%", f"%{keyword}%"))
            
            for row in cursor.fetchall():
                results.append({
                    "type": "knowledge",
                    "category": row[0],
                    "title": row[1],
                    "content": row[2],
                    "priority": row[3]
                })
            
            # 搜索FAQ
            cursor.execute('''
                SELECT category, question, answer, priority 
                FROM faq 
                WHERE question LIKE ? OR answer LIKE ?
                ORDER BY priority DESC, view_count DESC
                LIMIT 2
            ''', (f"%{keyword}%", f"%{keyword}%"))
            
            for row in cursor.fetchall():
                results.append({
                    "type": "faq",
                    "category": row[0],
                    "question": row[1],
                    "answer": row[2],
                    "priority": row[3]
                })
        
        conn.close()
        
        # 去重并按优先级排序
        unique_results = {r["content"] if r["type"] == "knowledge" else r["answer"]: r for r in results}
        sorted_results = sorted(unique_results.values(), key=lambda x: x["priority"], reverse=True)
        
        return sorted_results[:3]
    
    def chat_stream(self, user_input, history):
        # 在数据库中搜索相关内容
        relevant_info = self._search_knowledge(user_input)
        
        # 构建上下文
        context = ""
        if relevant_info:
            context = "\n\n【相关知识库内容】\n"
            for item in relevant_info:
                if item["type"] == "knowledge":
                    context += f"\n[{item['category']}] {item['title']}\n{item['content']}\n"
                else:  # FAQ
                    context += f"\n[FAQ - {item['category']}] Q: {item['question']}\nA: {item['answer']}\n"
        
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
    
    def add_knowledge(self, category, title, content, keywords, priority=5):
        """动态添加知识（API接口可以调用）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO knowledge_base (category, title, content, keywords, priority, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (category, title, content, keywords, priority, now, now))
        
        conn.commit()
        conn.close()
        print(f"✅ 已添加新知识：{title}")
    
    def add_faq(self, category, question, answer, priority=5):
        """动态添加FAQ"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO faq (category, question, answer, priority, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (category, question, answer, priority, now))
        
        conn.commit()
        conn.close()
        print(f"✅ 已添加新FAQ：{question}")
