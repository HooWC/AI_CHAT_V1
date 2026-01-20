# 🤖 Hong Seng Group AI 训练方式指南

## 当前使用的方法：TXT 文件知识库

您目前使用的是最简单的方法：在 `knowledge/` 文件夹中放置 `.txt` 文件。

---

## 🚀 其他可用的AI训练方式

### 1. 📋 JSON 格式结构化知识库（推荐）

**优点：**
- 结构化数据，易于管理
- 支持分类、标签、优先级
- 更精确的信息检索

**实现方式：**
创建 `knowledge/knowledge_base.json`：

```json
{
  "company_info": {
    "name": "Hong Seng Group",
    "location": "Malaysia",
    "address": "Lot 53, Jalan 1/5, Seksyen 1, 43650 Bandar Baru Bangi, Selangor, Malaysia"
  },
  "it_support": {
    "documents_location": "\\\\192.1.1.30:2828",
    "itsm_url": "https://192.1.1.30:2828",
    "contact_email": "it@hongseng.com"
  },
  "faq": [
    {
      "question": "IT文档在哪里？",
      "answer": "所有IT文档存储在 \\\\192.1.1.30:2828",
      "category": "IT",
      "priority": "high"
    }
  ]
}
```

### 2. 🗄️ 数据库集成（适合大量数据）

**优点：**
- 支持海量数据
- 动态更新，无需重启
- 可以集成现有系统

**实现方式：**
- **SQLite**：轻量级，适合中小型数据
- **MySQL/PostgreSQL**：适合企业级应用
- 可以存储对话历史、知识库、用户反馈

### 3. 🔍 向量数据库 + RAG（检索增强生成）【最强大】

**优点：**
- 语义搜索，而不是关键词匹配
- 支持大量文档
- 更智能的上下文理解

**推荐工具：**
- **FAISS**（Facebook AI）：免费，高性能
- **ChromaDB**：简单易用
- **Pinecone**：云服务（付费）

**工作原理：**
1. 将文档转换为向量（embeddings）
2. 用户提问时，搜索最相关的文档片段
3. 将相关内容注入到 AI 提示词中

### 4. 📄 多格式文档支持

**支持的格式：**
- PDF 文件（使用 PyPDF2 或 pdfplumber）
- Word 文档（使用 python-docx）
- Excel 表格（使用 pandas）
- CSV 文件
- Markdown 文件

### 5. 🌐 网站爬虫 + 实时数据

**优点：**
- 自动更新公司网站信息
- 可以抓取内部系统数据

**实现方式：**
- 使用 BeautifulSoup 或 Scrapy
- 定时任务自动更新

### 6. 🎯 Fine-tuning（微调模型）【最高级】

**优点：**
- 完全定制化的AI
- 最准确的回答

**缺点：**
- 需要大量计算资源
- 需要专业知识
- 成本较高

---

## 💡 我为您推荐的方案

### 方案A：JSON + 向量数据库（最佳平衡）

适合您的场景，结合了结构化数据和智能检索。

### 方案B：继续使用 TXT + 增强版本

如果想保持简单，可以改进现有的 TXT 系统：
- 添加自动索引
- 支持更多文件格式
- 添加搜索排序

---

## 📦 实现文件已包含

我已经为您创建了以下文件：

1. **hsg_engine_json.py** - JSON知识库版本
2. **hsg_engine_rag.py** - RAG向量数据库版本（推荐）
3. **hsg_engine_database.py** - SQLite数据库版本

您可以根据需要选择使用！

---

## 🔧 如何切换训练方式

在 `app.py` 中修改导入：

```python
# 方式1：使用原始TXT（当前）
from hsg_engine import HSGEngine

# 方式2：使用JSON知识库
from hsg_engine_json import HSGEngineJSON as HSGEngine

# 方式3：使用RAG向量数据库
from hsg_engine_rag import HSGEngineRAG as HSGEngine

# 方式4：使用数据库
from hsg_engine_database import HSGEngineDB as HSGEngine
```

---

## 📞 需要帮助？

如果您想实现某个特定的方案，请告诉我，我可以帮您配置！
