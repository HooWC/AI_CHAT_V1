# 🎨 如何添加公司Logo和品牌元素

## 方法1：添加Logo图片

### 步骤1：准备Logo文件

将您的Logo文件（PNG、JPG或SVG格式）放入 `static/` 文件夹：

```
static/
  ├── style.css
  └── logo.png  ← 您的Logo
```

### 步骤2：在Header中显示Logo

编辑 `templates/index.html`，找到第26行：

**原代码：**
```html
<div class="header-title">🏢 HONG SENG GROUP AI 助手</div>
```

**修改为：**
```html
<div class="header-title">
    <img src="{{ url_for('static', filename='logo.png') }}" 
         alt="Hong Seng Group" 
         class="company-logo">
    HONG SENG GROUP AI 助手
</div>
```

### 步骤3：添加Logo样式

在 `static/style.css` 中添加：

```css
.company-logo {
    height: 32px;
    margin-right: 12px;
    vertical-align: middle;
    filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.3));
    transition: transform 0.3s ease;
}

.company-logo:hover {
    transform: scale(1.05);
}
```

---

## 方法2：使用SVG Logo（推荐）

SVG格式的Logo可以无损缩放且文件小。

### 创建内联SVG Logo

编辑 `templates/index.html`：

```html
<div class="header-title">
    <svg class="company-logo-svg" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
        <!-- 这里是您的SVG代码 -->
        <circle cx="50" cy="50" r="40" fill="url(#gradient)"/>
        <text x="50" y="55" text-anchor="middle" fill="white" font-size="20" font-weight="bold">HSG</text>
        <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#00d4ff;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#a78bfa;stop-opacity:1" />
            </linearGradient>
        </defs>
    </svg>
    HONG SENG GROUP AI 助手
</div>
```

CSS样式：

```css
.company-logo-svg {
    height: 36px;
    width: 36px;
    margin-right: 12px;
    vertical-align: middle;
    filter: drop-shadow(0 0 10px rgba(0, 212, 255, 0.4));
}
```

---

## 方法3：添加Favicon（浏览器标签图标）

### 步骤1：准备Favicon文件

将 `favicon.ico` 或 `favicon.png` 放入 `static/` 文件夹。

### 步骤2：在HTML中引用

编辑 `templates/index.html`，在 `<head>` 标签中添加：

```html
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hong Seng Group AI 助手</title>
    
    <!-- 添加Favicon -->
    <link rel="icon" type="image/png" href="{{ url_for('static', filename='favicon.png') }}">
    <link rel="shortcut icon" href="{{ url_for('static', filename='favicon.ico') }}">
    
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
```

---

## 方法4：自定义品牌配色

### 在CSS中定义公司品牌色

编辑 `static/style.css`，修改CSS变量：

```css
:root {
    /* 根据您的公司VI（视觉识别系统）修改这些颜色 */
    --brand-primary: #00d4ff;      /* 主品牌色 */
    --brand-secondary: #a78bfa;    /* 辅助品牌色 */
    --brand-accent: #ff6b6b;       /* 强调色 */
    
    /* 应用到界面元素 */
    --accent-blue: var(--brand-primary);
    --accent-purple: var(--brand-secondary);
}
```

---

## 方法5：添加公司口号或标语

### 在侧边栏底部添加

编辑 `templates/index.html`，找到侧边栏底部：

```html
<div class="sidebar-footer">
    <div class="company-info">
        <img src="{{ url_for('static', filename='logo.png') }}" 
             alt="HSG" 
             class="sidebar-logo">
        <p class="company-name">Hong Seng Group</p>
        <p class="company-tagline">Excellence in Everything</p>
        <p class="company-location">📍 Malaysia</p>
    </div>
</div>
```

### 添加样式

```css
.company-info {
    text-align: center;
    padding: 15px;
}

.sidebar-logo {
    height: 40px;
    margin-bottom: 10px;
    filter: brightness(0.9);
}

.company-name {
    font-weight: 600;
    color: var(--accent-blue);
    margin: 5px 0;
    font-size: 14px;
}

.company-tagline {
    font-size: 11px;
    color: #888;
    font-style: italic;
    margin: 3px 0;
}

.company-location {
    font-size: 12px;
    color: #999;
    margin-top: 5px;
}
```

---

## 方法6：欢迎页面品牌化

### 添加大Logo到欢迎页面

编辑 `templates/index.html`，修改欢迎消息部分：

```html
<div class="welcome-message">
    <img src="{{ url_for('static', filename='logo.png') }}" 
         alt="Hong Seng Group" 
         class="welcome-logo">
    <h2>欢迎使用 Hong Seng Group AI 助手</h2>
    <p class="welcome-subtitle">您的智能IT助手，随时为您服务</p>
    <div class="company-badge">
        <span class="badge-icon">🏢</span>
        <span class="badge-text">马来西亚 · 丰成集团</span>
    </div>
    <div class="quick-actions">
        <button class="quick-btn" onclick="quickAsk('IT部门的文档存放在哪里？')">
            📁 IT文档位置
        </button>
        <button class="quick-btn" onclick="quickAsk('介绍一下Hong Seng Group')">
            🏢 公司介绍
        </button>
        <button class="quick-btn" onclick="quickAsk('如何联系IT支持？')">
            💬 IT支持
        </button>
    </div>
</div>
```

### 样式

```css
.welcome-logo {
    height: 80px;
    margin-bottom: 20px;
    filter: drop-shadow(0 4px 20px rgba(0, 212, 255, 0.3));
    animation: logoFloat 3s ease-in-out infinite;
}

@keyframes logoFloat {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

.welcome-subtitle {
    color: #aaa;
    margin: 10px 0 20px 0;
    font-size: 15px;
}

.company-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0, 212, 255, 0.1);
    padding: 8px 20px;
    border-radius: 20px;
    border: 1px solid rgba(0, 212, 255, 0.3);
    margin-bottom: 30px;
    font-size: 14px;
    color: var(--accent-blue);
}

.badge-icon {
    font-size: 18px;
}
```

---

## 快速示例：完整品牌化方案

### 创建简单的文字Logo

如果暂时没有Logo图片，可以使用文字Logo：

```html
<div class="header-title">
    <div class="text-logo">
        <span class="logo-h">H</span>
        <span class="logo-s">S</span>
        <span class="logo-g">G</span>
    </div>
    HONG SENG GROUP AI 助手
</div>
```

```css
.text-logo {
    display: inline-flex;
    margin-right: 12px;
    gap: 2px;
}

.text-logo span {
    font-weight: 900;
    font-size: 20px;
    padding: 4px 8px;
    border-radius: 6px;
    text-shadow: 0 0 10px currentColor;
}

.logo-h {
    background: linear-gradient(135deg, #00d4ff, #0099cc);
    color: white;
}

.logo-s {
    background: linear-gradient(135deg, #0099cc, #a78bfa);
    color: white;
}

.logo-g {
    background: linear-gradient(135deg, #a78bfa, #8b5cf6);
    color: white;
}
```

---

## 📁 推荐的Logo尺寸

| 位置 | 推荐尺寸 | 格式 |
|------|---------|------|
| Header Logo | 32-40px 高度 | PNG/SVG |
| Sidebar Logo | 40-50px 高度 | PNG/SVG |
| Welcome Logo | 80-120px 高度 | PNG/SVG |
| Favicon | 16x16, 32x32 | ICO/PNG |

---

## 🎨 在线工具推荐

- **Logo设计**：Canva, Figma
- **Favicon生成**：https://favicon.io/
- **SVG优化**：https://jakearchibald.github.io/svgomg/
- **配色方案**：https://coolors.co/

---

## ✅ 完成检查清单

- [ ] 准备好Logo文件（PNG/SVG）
- [ ] 将Logo放入 `static/` 文件夹
- [ ] 在Header中添加Logo
- [ ] 添加Favicon
- [ ] 自定义品牌配色
- [ ] 在欢迎页面添加品牌元素
- [ ] 测试不同屏幕尺寸的显示效果

---

**需要帮助？** 请联系IT部门获取公司的官方Logo文件和品牌指南。

*Hong Seng Group IT Department*
