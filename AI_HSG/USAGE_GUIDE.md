# 🎓 Hong Seng Group AI 助手 - 使用指南

## 📖 目录

1. [快速开始](#快速开始)
2. [AI训练方式对比](#ai训练方式对比)
3. [如何添加公司知识](#如何添加公司知识)
4. [如何切换训练方式](#如何切换训练方式)
5. [界面自定义](#界面自定义)
6. [常见问题](#常见问题)

---

## 🚀 快速开始

### 第一步：安装依赖

```bash
# 安装基础依赖
pip install flask transformers torch accelerate

# 如果想使用RAG向量数据库（推荐）
pip install sentence-transformers
```

### 第二步：启动服务

```bash
python app.py
```

### 第三步：访问

在浏览器中打开：http://localhost:5000

---

## 🎯 AI训练方式对比

### 对比表格

| 方式 | 优点 | 缺点 | 适用场景 | 难度 |
|------|------|------|----------|------|
| **TXT文件** | 简单易用 | 搜索不智能 | 小型知识库 | ⭐ |
| **JSON格式** | 结构化，易管理 | 需要JSON格式 | 中型知识库 | ⭐⭐ |
| **RAG向量库** | 语义搜索，最智能 | 需要额外依赖 | 大型知识库 | ⭐⭐⭐ |
| **SQLite数据库** | 动态更新，可扩展 | 需要数据库知识 | 企业级应用 | ⭐⭐⭐ |

### 推荐方案

- **小公司（<100个文档）**：使用 **TXT文件** 或 **JSON格式**
- **中型公司（100-1000个文档）**：使用 **RAG向量数据库**⭐ 推荐
- **大型企业（>1000个文档）**：使用 **数据库** + **RAG**

---

## 📚 如何添加公司知识

### 方法1：TXT文件（最简单）

1. 在 `knowledge/` 文件夹中创建 `.txt` 文件
2. 直接写入内容：

```text
【IT支持信息】
IT部门文档位置：\\192.1.1.30:2828\IT_Documents
ITSM系统：https://192.1.1.30:2828

如有技术问题，请访问ITSM系统提交工单。
```

3. 保存后重启程序

### 方法2：JSON格式（推荐）

1. 编辑 `knowledge/knowledge_base.json`
2. 添加新的FAQ或知识：

```json
{
    "faq": [
        {
            "category": "新分类",
            "question": "你的问题？",
            "answer": "答案内容",
            "priority": "high",
            "keywords": ["关键词1", "关键词2"]
        }
    ]
}
```

3. 在 `app.py` 中切换到JSON引擎：

```python
from hsg_engine_json import HSGEngineJSON as HSGEngine
```

### 方法3：使用数据库（动态更新）

```python
# 使用Python代码添加知识
from hsg_engine_database import HSGEngineDB

engine = HSGEngineDB()

# 添加知识
engine.add_knowledge(
    category="IT",
    title="新系统上线通知",
    content="我们的新系统已在 2026年1月 上线...",
    keywords="系统,上线,通知",
    priority=10
)

# 添加FAQ
engine.add_faq(
    category="HR",
    question="如何申请年假？",
    answer="请登录HR系统提交申请...",
    priority=8
)
```

---

## 🔄 如何切换训练方式

### 步骤1：选择引擎

编辑 `app.py` 文件，修改第4行：

```python
# 选项1：TXT文件（当前默认）
from hsg_engine import HSGEngine

# 选项2：JSON格式
from hsg_engine_json import HSGEngineJSON as HSGEngine

# 选项3：RAG向量数据库（推荐）
from hsg_engine_rag import HSGEngineRAG as HSGEngine

# 选项4：SQLite数据库
from hsg_engine_database import HSGEngineDB as HSGEngine
```

### 步骤2：安装依赖

```bash
# RAG需要额外安装
pip install sentence-transformers
```

### 步骤3：重启服务

```bash
python app.py
```

---

## 🎨 界面自定义

### 修改配色方案

编辑 `static/style.css`，修改CSS变量：

```css
:root {
    --accent-blue: #00d4ff;     /* 主色调 */
    --accent-purple: #a78bfa;   /* 辅助色 */
    --main-bg: #0a0e27;         /* 背景色 */
}
```

### 修改公司Logo

在 `templates/index.html` 中找到：

```html
<div class="header-title">🏢 HONG SENG GROUP AI 助手</div>
```

替换为：

```html
<div class="header-title">
    <img src="/static/logo.png" alt="Logo" style="height: 30px;">
    HONG SENG GROUP AI 助手
</div>
```

### 修改欢迎消息

在 `templates/index.html` 中找到 `.welcome-message` 部分并修改内容。

---

## ❓ 常见问题

### Q1：AI回答不准确怎么办？

**答：** 
1. 检查 `knowledge/` 文件夹中的内容是否完整
2. 增加更多具体的信息和示例
3. 考虑切换到 **RAG向量数据库** 引擎
4. 在知识库中添加更多相关关键词

### Q2：如何让AI记住更多公司信息？

**答：**
1. 在 `knowledge/` 中添加更多文件
2. 使用 **JSON格式** 添加优先级高的FAQ
3. 使用 **数据库方式** 进行结构化管理

### Q3：能否集成到公司现有系统？

**答：** 可以！
- Flask提供了REST API接口
- 可以通过 `/chat` 端点进行集成
- 支持SSE（Server-Sent Events）流式响应

### Q4：如何备份聊天记录？

**答：**
聊天记录保存在 `chat_history.json` 文件中，定期备份此文件即可。

### Q5：能否支持多语言？

**答：**
当前模型主要支持中文和英文。如需其他语言，可以：
1. 更换支持多语言的模型
2. 在知识库中添加多语言内容

### Q6：消耗多少计算资源？

**答：**
- **CPU模式**：2-4GB RAM，适合办公电脑
- **GPU模式**：速度更快，但需要NVIDIA显卡
- 可在 `hsg_engine.py` 中修改 `device_map`

---

## 🔧 高级配置

### 修改AI回答长度

在各个 `hsg_engine_*.py` 文件中修改：

```python
generate_kwargs = dict(
    max_new_tokens=512,  # 改为 1024 获得更长回答
    temperature=0.1,     # 0.1=保守，0.7=创意
)
```

### 添加用户认证

在 `app.py` 中添加 Flask-Login：

```python
from flask_login import LoginManager, login_required

@app.route('/')
@login_required
def index():
    return render_template('index.html')
```

### 集成数据库

修改 `app.py`，将 `chat_history.json` 替换为数据库存储。

---

## 📞 技术支持

如有问题，请联系 IT 部门：
- **文档位置**：\\192.1.1.30:2828\IT_Documents
- **ITSM 系统**：https://192.1.1.30:2828

---

## 📝 更新知识库的最佳实践

1. **定期更新**：每周检查一次知识库内容
2. **分类管理**：按部门/主题分类
3. **优先级标记**：重要信息设置 `priority: high`
4. **添加关键词**：帮助AI更好地检索
5. **测试验证**：添加新知识后测试AI回答

---

**Happy Coding! 🚀**

*Hong Seng Group IT Department*
