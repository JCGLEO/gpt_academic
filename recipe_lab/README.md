# Recipe R&D Assistant (Gemini 2.5 + Python)

这是一个可部署在云端、通过网页在 **Android / iOS / macOS / Windows** 访问的菜谱研发助手方案。

> 说明：Gemini 2.5 目前不支持你在本地“重新训练一个完整基座模型参数”。对于你的目标，工程上可行且效果最好的方式是：
> 1) 构建私有菜谱知识库；
> 2) 用 RAG（检索增强生成）+ 提示工程；
> 3) 通过 Gemini 2.5 做推理与创意研发。

## 架构

- `app/data_pipeline.py`：把料理书籍 TXT/MD/JSONL/PDF 文本切片并构建 FAISS 索引。
- `app/rag.py`：检索相关片段并组装上下文。
- `app/main.py`：FastAPI 服务（REST API）。
- `web/index.html`：最小前端页面（可直接封装为 PWA 或部署到静态托管）。

## 1) 安装

```bash
cd recipe_lab
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

配置 `.env`：

```env
GEMINI_API_KEY=your_google_ai_studio_key
GEMINI_MODEL=gemini-2.5-pro
EMBED_MODEL=models/text-embedding-004
DATA_DIR=./data/raw
INDEX_DIR=./data/index
TOP_K=5
```

## 2) 书籍资料如何“投喂”到模型（重点）

你可以把这套流程理解成“训练知识库”而不是“重训基座模型”。

### 2.1 先做数据清洗（推荐）

建议你把每本书先拆成结构化信息：

- 菜名
- 食材（主料/辅料/调料）
- 克重或体积
- 步骤
- 温度/火候
- 时间
- 失败点与纠偏
- 适配设备（炒锅/烤箱/空气炸锅）

### 2.2 文件放置规范

把资料放到 `data/raw/` 下，支持：

- `.txt`
- `.md`
- `.jsonl`
- `.pdf`（可检索文本 PDF）

建议目录示例：

```text
data/raw/
  chuan/
    mapo_tofu.md
  yue/
    steamed_fish.txt
  books/
    modern_chinese_cuisine.pdf
  recipes.jsonl
```

### 2.3 JSONL 推荐格式（最稳定）

```json
{"title":"宫保鸡丁","content":"食材:鸡腿肉300g... 步骤:... 失败点:...","tags":["川菜","家常","快手"]}
{"title":"清蒸鲈鱼","content":"食材:鲈鱼1条... 步骤:... 失败点:...","tags":["粤菜","蒸"]}
```

### 2.4 执行索引构建（即投喂）

```bash
python -m app.data_pipeline
```

执行完成后会生成：

- `data/index/faiss.index`
- `data/index/chunks.json`

这一步完成后，你的新知识就能在问答时被检索并注入 Gemini 2.5。

### 2.5 增量更新建议

每次新增书籍后，重新执行一次 `python -m app.data_pipeline`。

当数据量变大时，建议：

- 按菜系分索引（川菜索引/粤菜索引）
- 按场景分索引（健身餐/儿童餐/宴客）
- 用 `tags` 在应用层先过滤再检索

## 3) 启动 API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 4) 访问网页

直接打开 `web/index.html`，或将其部署到任意静态服务。

默认 API 地址：`http://localhost:8000`。

## 核心 API

### `POST /chat`

请求：

```json
{
  "query": "请基于川菜和粤菜，设计3道适合家庭空气炸锅的晚餐菜谱",
  "session_id": "demo-user"
}
```

返回：

```json
{
  "answer": "...",
  "contexts": ["知识片段1", "知识片段2"],
  "model": "gemini-2.5-pro"
}
```

### `GET /health`

健康检查。

## 部署建议（跨平台访问）

- 后端：Docker + 云主机/容器平台（Cloud Run、Fly.io、Railway、Render）。
- 前端：Vercel/Netlify/GitHub Pages。
- 手机访问：直接浏览器访问网页；若要“像 App”，可进一步做 PWA（manifest + service worker）。

## 后续增强

1. 增加 OCR 流程（扫描版书籍）。
2. 增加菜品成本、营养、过敏原标签。
3. 增加实验记录模块（A/B 配方对比）。
4. 增加用户反馈闭环（点赞/复做成功率）用于提示优化。
