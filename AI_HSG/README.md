# 🏢 Hong Seng Group AI 助手

一个为香港生集团（马来西亚）定制的智能AI聊天助手系统。

## ✨ 特性

- 💬 实时流式对话体验
- 📚 多会话管理
- 🎨 现代化美观界面
- 🔒 本地知识库，数据安全
- 🚀 多种AI训练方式可选

## 🎯 快速开始

### 1. 安装依赖

```bash
# 基础版本（TXT知识库）
pip install flask transformers torch accelerate

# 或者安装完整版（包含所有功能）
pip install -r requirements_extended.txt
```

### 2. 运行程序

```bash
python app.py
```

### 3. 访问

打开浏览器访问：http://localhost:5000

## 📂 项目结构

```
AI_HSG/
├── app.py                      # Flask 主程序
├── hsg_engine.py               # AI引擎（TXT知识库）
├── hsg_engine_json.py          # JSON知识库版本
├── hsg_engine_rag.py           # RAG向量数据库版本（推荐）
├── hsg_engine_database.py      # SQLite数据库版本
├── templates/
│   └── index.html              # 前端页面
├── static/
│   └── style.css               # 样式文件
├── knowledge/                  # 知识库文件夹
│   ├── about_us.txt
│   ├── contact.txt
│   ├── identity.txt
│   ├── it_support.txt
│   └── products.txt
└── chat_history.json           # 聊天历史
```

## 🔧 AI训练方式选择

### 方式 1：TXT 文件知识库（当前默认）

**优点：** 简单易用，直接编辑文本文件
**使用：** 在 `knowledge/` 文件夹中添加 `.txt` 文件

```python
# app.py
from hsg_engine import HSGEngine
```

### 方式 2：JSON 结构化知识库

**优点：** 结构化数据，支持分类和优先级
**使用：** 创建 `knowledge/knowledge_base.json`

```python
# app.py
from hsg_engine_json import HSGEngineJSON as HSGEngine
```

### 方式 3：RAG 向量数据库（推荐⭐）

**优点：** 语义搜索，最智能的检索
**依赖：** `pip install sentence-transformers`

```python
# app.py
from hsg_engine_rag import HSGEngineRAG as HSGEngine
```

### 方式 4：SQLite 数据库

**优点：** 支持动态更新，可以集成API
**使用：** 自动创建数据库

```python
# app.py
from hsg_engine_database import HSGEngineDB as HSGEngine
```

## 📚 详细文档

查看 [AI_TRAINING_METHODS.md](AI_TRAINING_METHODS.md) 了解更多训练方式的详细信息。

## 🎨 界面特性

- ✅ 现代化渐变色设计
- ✅ 流畅的动画效果
- ✅ 响应式布局
- ✅ 暗色主题
- ✅ 欢迎页面和快速操作
- ✅ 打字指示器
- ✅ 代码高亮支持（Markdown）

## 🔐 安全性

- 所有数据存储在本地
- 不连接外部服务器
- 知识库完全可控

## 📞 技术支持

如有问题，请联系 IT 部门：
- 文档位置：\\192.1.1.30:2828\IT_Documents
- ITSM 系统：https://192.1.1.30:2828

## 🚀 下一步

1. **添加公司知识库**：在 `knowledge/` 文件夹中添加更多 TXT 文件
2. **选择训练方式**：根据需求选择合适的 AI 引擎
3. **定制界面**：修改 `static/style.css` 自定义样式
4. **集成系统**：可以通过 API 集成到现有系统

## 📝 更新日志

### v2.0 (2026-01)
- ✨ 全新美化的界面设计
- ✨ 添加多种AI训练方式
- ✨ 欢迎页面和快速操作
- ✨ 改进的流式对话体验

### v1.0 (初始版本)
- 基础聊天功能
- TXT知识库支持
- 多会话管理

---

**Hong Seng Group IT Department**
*Powered by Qwen 2.5*
