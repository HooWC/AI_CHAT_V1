import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread


class HSGEngine:
    def __init__(self):
        self.model_id = "Qwen/Qwen2.5-0.5B-Instruct"

        # --- 使用统一的 JSON 知识库 ---
        self.kb_dir = "knowledge"
        self.kb_json_path = os.path.join(self.kb_dir, "knowledge_base.json")
        self.combined_knowledge = ""

        if not os.path.exists(self.kb_dir):
            os.makedirs(self.kb_dir)
            print(f"⚠️ 文件夹 {self.kb_dir} 已创建，请放入知识库文件。")

        # 1) 先加载 JSON 知识库（主数据源）
        kb_json = None
        if os.path.exists(self.kb_json_path):
            try:
                with open(self.kb_json_path, "r", encoding="utf-8") as f:
                    kb_json = json.load(f)
                print("✅ 已加载 knowledge/knowledge_base.json 作为主知识库。")
            except Exception as e:
                print(f"⚠️ 加载 knowledge_base.json 失败: {e}")
        else:
            print("⚠️ 未找到 knowledge_base.json，将仅使用 .txt 文档（如果有）。")

        if kb_json:
            self.combined_knowledge += self._format_json_knowledge(kb_json)

        # 2) 兼容：继续把所有 TXT 当作补充知识库（可选）
        txt_files = [f for f in os.listdir(self.kb_dir) if f.endswith(".txt")]
        if txt_files:
            for filename in txt_files:
                with open(os.path.join(self.kb_dir, filename), "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if not content:
                        continue
                    self.combined_knowledge += f"\n[参考文档: {filename}]\n{content}\n"
            print(f"✅ 额外载入 {len(txt_files)} 个 .txt 公司知识文档作为补充。")
        else:
            if not kb_json:
                print("⚠️ 警告：知识库内没有 JSON 或 TXT 文件，AI 将失去公司背景数据。")

        # 加载模型和分词器
        print("🚀 正在启动 HSG 深度智能引擎...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float32,
            device_map={"": "cpu"},  # 强制使用 CPU
        )

        # 从 JSON 中尽量取关键信息，保证和知识库信息一致
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
        it_docs_path = (
            kb_json.get("it_support", {}).get("documents_path", "\\\\192.1.1.30:2828\\IT_Documents")
            if kb_json
            else "\\\\192.1.1.30:2828\\IT_Documents"
        )
        itsm_url = (
            kb_json.get("it_support", {}).get("itsm_server", "http://hongseng.ddns.net/itsm 或 http://rs1/itsm")
            if kb_json
            else "https://192.1.1.30:2828"
        )
        node_api_url = (
            kb_json.get("it_support", {}).get("node_api", itsm_url)
            if kb_json
            else "https://192.1.1.30:2828"
        )

        # --- 系统提示词：完全基于本地 JSON / TXT 知识库 ---
        self.system_prompt = (
            f"你现在是马来西亚 {company_name} ({company_cn_name}) 的专属 IT 助手。\n"
            "【绝对指令】回答必须优先基于以下提供的【公司本地知识库】内容，如无相关内容再做合理推理。\n"
            f"1. 身份：{important_note}。\n"
            f"2. IT 文件位置：必须回答 {it_docs_path}。\n"
            f"3. ITSM 和 Node API 统一地址：{itsm_url}（Node API 默认同址：{node_api_url}）。\n"
            "4. 语气：简短、直接、专业，不要废话。\n\n"
            f"--- 公司本地知识库开始 ---\n{self.combined_knowledge}\n--- 结束 ---"
        )

    def _format_json_knowledge(self, kb_json: dict) -> str:
        """把 knowledge_base.json 结构化地展开成一段长文本，方便大模型读取。"""
        parts = []

        company = kb_json.get("company_info", {})
        if company:
            parts.append("[公司信息]")
            parts.append(
                f"名称: {company.get('name', '')}（{company.get('chinese_name', '')}）\n"
                f"国家: {company.get('country', '')}\n"
                f"地址: {company.get('full_address', '')}\n"
                f"简介: {company.get('description', '')}\n"
                f"重要说明: {company.get('important_note', '')}"
            )

        it = kb_json.get("it_support", {})
        if it:
            parts.append("\n[IT 支持信息]")
            parts.append(
                f"IT 文档路径: {it.get('documents_path', '')}\n"
                f"文档说明: {it.get('documents_description', '')}\n"
                f"ITSM 地址: {it.get('itsm_server', '')}\n"
                f"Node API 地址: {it.get('node_api', '')}\n"
                f"支持说明: {it.get('support_note', '')}\n"
                f"联系方式: {it.get('contact', '')}"
            )

        departments = kb_json.get("departments", {})
        if departments:
            parts.append("\n[部门信息]")
            for key, dep in departments.items():
                parts.append(
                    f"- 部门标识: {key}\n"
                    f"  名称: {dep.get('name', '')}\n"
                    f"  职责: {dep.get('role', '')}\n"
                    f"  服务: {', '.join(dep.get('services', []))}"
                )

        faq_list = kb_json.get("faq", [])
        if faq_list:
            parts.append("\n[常见问题 FAQ]")
            for item in faq_list:
                parts.append(
                    f"Q: {item.get('question', '')}\n"
                    f"A: {item.get('answer', '')}\n"
                    f"分类: {item.get('category', '')}, 重要级别: {item.get('priority', '')}\n"
                    f"关键词: {', '.join(item.get('keywords', []))}"
                )

        quick_links = kb_json.get("quick_links", {})
        if quick_links:
            parts.append("\n[快速链接]")
            for k, v in quick_links.items():
                parts.append(f"{k}: {v}")

        return "\n".join(parts)

    def chat_stream(self, user_input, history):
        # 实时拦截：如果用户问身份，我们在 Prompt 里加码
        extra_remind = ""
        identity_keywords = ["你是谁", "哪里的", "什么公司", "who are you", "where"]
        if any(k in user_input.lower() for k in identity_keywords):
            extra_remind = "\n(特别提醒：请再次确认你是马来西亚的 HSG，不是香港的。)"

        # 构造对话消息
        messages = [{"role": "system", "content": self.system_prompt + extra_remind}]
        
        # 限制历史记忆（保留最近几轮对话）
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
        
        # 生成参数
        generate_kwargs = dict(
            input_ids=input_ids,
            streamer=streamer,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,      # 调到最低，保证回答极其精准
            top_p=0.8,
            repetition_penalty=1.2
        )

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text