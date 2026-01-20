# ⚡ 快速开始 - 5分钟上手

## 🎯 第一次使用？从这里开始！

### 步骤 1：查看新界面
```bash
python app.py
```
打开浏览器访问：http://localhost:5000

您会看到：
- ✨ 全新的渐变色界面
- 👋 欢迎页面
- 🎨 流畅的动画效果

---

### 步骤 2：尝试快速提问

点击欢迎页面的按钮：
- 📁 IT文档位置
- 🏢 公司介绍  
- 💬 IT支持

---

### 步骤 3：添加公司知识

**最简单的方法（推荐新手）：**

在 `knowledge/` 文件夹创建 `my_info.txt`：

```
【我的部门信息】
部门名称：人力资源部
部门负责人：张经理
联系方式：hr@hongseng.com

常见问题：
1. 如何申请年假？
答：请登录HR系统提交申请。

2. 考勤时间是几点？
答：上午9点到下午6点。
```

保存后重启程序，AI就会学到这些信息！

---

### 步骤 4：升级到更智能的方式（可选）

#### 选项A：使用JSON格式（更好管理）

1. 编辑 `knowledge/knowledge_base.json`
2. 修改 `app.py` 第4行：
```python
from hsg_engine_json import HSGEngineJSON as HSGEngine
```

#### 选项B：使用RAG智能检索（最强大）

1. 安装依赖：
```bash
pip install sentence-transformers
```

2. 修改 `app.py` 第4行：
```python
from hsg_engine_rag import HSGEngineRAG as HSGEngine
```

---

## 🎨 个性化设置（5分钟）

### 修改主色调

编辑 `static/style.css` 第3-4行：

```css
--accent-blue: #00d4ff;      /* 改成你喜欢的颜色 */
--accent-purple: #a78bfa;    /* 辅助颜色 */
```

### 添加Logo

1. 把Logo文件（`logo.png`）放到 `static/` 文件夹
2. 参考 `HOW_TO_ADD_LOGO.md` 详细说明

---

## 📊 选择AI训练方式

### 我该用哪种？

| 您的情况 | 推荐方式 | 文件 |
|---------|---------|------|
| 刚开始用，知识不多 | TXT文件 | `hsg_engine.py` ✅ 当前 |
| 需要分类管理 | JSON格式 | `hsg_engine_json.py` |
| 文档很多，想要智能搜索 | RAG向量库 | `hsg_engine_rag.py` ⭐ |
| 要集成其他系统 | 数据库 | `hsg_engine_database.py` |

### 切换方式

编辑 `app.py` 第4行：

```python
# 原始（当前）
from hsg_engine import HSGEngine

# 改成其他方式（选一个）
from hsg_engine_json import HSGEngineJSON as HSGEngine
from hsg_engine_rag import HSGEngineRAG as HSGEngine
from hsg_engine_database import HSGEngineDB as HSGEngine
```

---

## 🆘 常见问题

### Q: 程序启动很慢？
A: 第一次运行需要下载AI模型（约500MB），请耐心等待。

### Q: AI回答不对？
A: 检查 `knowledge/` 文件夹是否有相关信息。

### Q: 想让AI更聪明？
A: 
1. 在知识库中添加更多信息
2. 升级到RAG方式
3. 增加FAQ示例

### Q: 可以部署到服务器吗？
A: 可以！修改 `app.py` 最后一行：
```python
app.run(host='0.0.0.0', port=5000)
```

---

## 📚 更多帮助

- **完整教程** → `USAGE_GUIDE.md`
- **AI训练方式** → `AI_TRAINING_METHODS.md`
- **添加Logo** → `HOW_TO_ADD_LOGO.md`
- **升级说明** → `UPGRADE_SUMMARY.md`

---

## ✅ 快速检查

- [ ] 程序能正常启动
- [ ] 界面显示正常（有渐变色背景）
- [ ] 能发送消息并收到回复
- [ ] 欢迎页面显示正确
- [ ] 可以创建新对话
- [ ] 可以切换历史会话

全部打勾？恭喜您已成功上手！🎉

---

## 🚀 接下来做什么？

1. ✅ 添加更多公司知识到 `knowledge/`
2. ✅ 自定义界面颜色
3. ✅ 尝试不同的AI训练方式
4. ✅ 添加公司Logo
5. ✅ 分享给同事使用

---

**需要帮助？** 查看 `USAGE_GUIDE.md` 获取详细说明！

*Hong Seng Group IT Department*  
*让AI助手为您服务！🤖*
