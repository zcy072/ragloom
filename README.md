# RAGLoom - 企业 RAG 知识库与 Agent 问答系统

面向企业文档的检索增强生成（RAG）知识库。系统把 PDF、DOCX、Markdown、TXT 和图片解析、分块后建立 FTS5 + Chroma 混合索引，问答时经过混合召回、重排序和上下文恢复，以 SSE 流式输出回答，并给出可追溯的来源编号。

前后端分离，Docker Compose 一键启动，内置账户、角色与知识库级权限控制，并附带可复现的离线检索评测。

## 特性

- **混合检索**：Chroma 语义检索与 SQLite FTS5 关键词检索加权融合，补充精确词召回，跨路线去重后用 BGE Reranker 重排序。
- **多路线分块**：固定父子分块保证稳定召回，命中子块后恢复父块语境；结构化分块保留章节标题链和表格；视觉内容（图片描述、OCR 文字）形成补充证据。
- **多种解析方式**：PDF 可选本地文本提取、MinerU 云端解析或 PaddleOCR；图片优先走视觉模型，失败自动回退本地 OCR。
- **异步索引**：上传后创建持久化任务，由线程池 worker 并行处理。解析与嵌入阶段不持锁，只在准备和发布阶段短暂持有知识库锁，同名文件重复上传时旧任务自动作废。
- **可追溯问答**：SSE 流式回答、来源编号、会话记录与摘要记忆、追问改写、按文档类型/标签/生效日期筛选。可选的检索规划（function calling）与书面化补充查询。
- **权限与安全**：账户、角色、知识库级权限，每次请求实时校验；登录与注册限流；模型 API Key 加密后存入 SQLite，支持主密钥轮换。
- **模型可配置**：聊天模型接入任意 OpenAI 兼容服务，在界面中管理；嵌入与重排序使用 SiliconFlow 的 BGE 系列模型。

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 后端 | Python 3.10、FastAPI、Pydantic 2、SQLite、SSE |
| 检索 | Chroma、SQLite FTS5、jieba、BGE-M3、BGE Reranker v2 m3 |
| 文档处理 | pypdf、python-docx、PaddleOCR、MinerU（可选） |
| 前端 | Vue 3、Vite、Nginx |
| 工程化 | Docker Compose、锁定依赖、Ruff、Pyright、unittest、Playwright |

## 架构

```text
浏览器
  |
Nginx（frontend，8080） ---------> FastAPI（/api，8000）
                                       |
                                       +-- SQLite：账户、配置、索引任务、会话
                                       +-- storage/data：原始上传文件
                                       +-- Chroma + FTS5：混合检索索引
                                       +-- IndexWorker：线程池并行执行索引任务
                                       +-- PaddleOCR：本地 OCR
                                       +-- 外部嵌入、重排序、聊天与解析服务
```

**入库流程**：上传文件 -> 创建 `queued` 索引任务 -> 解析/OCR -> 分块 -> 计算嵌入 -> 写入 FTS5 和 Chroma -> 状态变为 `ready`。

**问答流程**：加载会话摘要与近期消息 -> 可选检索规划、追问改写或补充查询 -> FTS5 + Chroma 多路召回 -> 去重与重排序 -> 父块恢复与上下文预算 -> SSE 流式回答 -> 保存会话和实际引用的来源。

### 代码结构

```text
api.py, app_storage.py     应用入口与依赖装配
routers/                   HTTP 路由、鉴权依赖、响应模型
services/
  core/                    配置、数据库 schema、依赖容器、模型配置、密钥加密
  identity/                账户、角色、令牌、限流
  documents/               解析、OCR、MinerU、视觉内容
  indexing/                分块、持久化任务、worker、知识库锁
  retrieval/               FTS5、Chroma、混合召回、上下文恢复
  chat/                    查询规划、改写、重排、引用、会话、流式回答
  knowledge_bases/         知识库生命周期
frontend/                  Vue 3 前端、单元测试与 Playwright 端到端测试
evaluation/                离线检索评测
contracts/openapi.json     OpenAPI 快照，由测试校验
scripts/                   检查、冒烟、清理、打包脚本
tests/                     后端测试
```

依赖方向为 `documents -> indexing -> retrieval -> chat`，`core` 提供共享基础设施。

## 检索评测

使用公开数据集 [CRUD-RAG](https://github.com/IAAR-Shanghai/CRUD_RAG)（Apache-2.0）的 2 文档和 3 文档问答任务，语料经过真实的上传、解析、分块、入库链路。共 1223 篇文档（423 篇标准文档加 800 篇随机干扰文档），随机抽取 200 道题，种子固定为 42。指标按文档计算。

| 配置 | hit@5 | full@5 | MRR | 单题耗时 p50 |
| --- | --- | --- | --- | --- |
| 仅 FTS5 | 0.945 | 0.625 | 0.826 | 29 ms |
| 仅向量 | 0.975 | 0.770 | 0.927 | 195 ms |
| 混合（FTS5 + 向量） | 0.970 | 0.775 | 0.927 | 302 ms |
| 混合 + 重排序 | 0.980 | 0.760 | 0.941 | 582 ms |

`hit@5` 是前 5 个文档中至少含一篇标准文档的题目比例，`full@5` 是覆盖全部标准文档的题目比例（多文档题的关键指标），`MRR` 是首个标准文档排名倒数的平均值。重排序耗时包含远程模型调用。

结果说明：混合检索相对单一的关键词检索提升明显；重排序把正确文档排得更靠前（MRR 与 hit@5 最高），但在 full@10 上略低于未重排序的混合检索，因为重排只保留前 10 个候选。

评测范围为文本新闻检索，语料是抽样子集，适合比较各检索配置之间的相对差异。完整报告在 [`evaluation/results/crud_rag_report.json`](evaluation/results/crud_rag_report.json)，复现步骤见 [`evaluation/README.md`](evaluation/README.md)。

## 快速开始

前置条件：Docker（含 Docker Compose）、SiliconFlow API Key（嵌入与重排序）、一个 OpenAI 兼容的聊天模型 API Key。宿主机不需要安装 Python 或 Node。

```bash
git clone <仓库地址> && cd ragloom
mkdir -p storage
cp .env.example .env
# 在 .env 中填写 MODEL_API_KEY_ENCRYPTION_KEY，生成方式：openssl rand -hex 32
docker compose up -d --build
docker compose ps
```

`MODEL_API_KEY_ENCRYPTION_KEY` 留空时，在“模型管理”保存任何 API Key 都会失败，所以首次启动前先填好。其余变量都有默认值。首次构建会下载 PaddlePaddle、Chroma 等较大的依赖，需要几分钟。

打开 `http://localhost:8080`，前端会把 `/api` 代理到后端（`http://localhost:8000`）。全新数据库会创建 `admin` 账户、空的“通用知识库”和“超级管理员”角色，默认密码为 `admin888`，登录后请立即修改。

首次使用：

1. 使用 `admin` / `admin888` 登录，在“模型管理”配置 SiliconFlow API Key 和聊天模型。
2. 在“通用知识库”上传文档，或新建知识库后上传。
3. 等待文档状态从 `processing` 变为 `ready`。
4. 在“问答对话”选择知识库和聊天模型后提问。

### 构建网络

基础镜像默认从 Daocloud 镜像站拉取，pip 使用官方 PyPI，npm 使用 npmmirror。可在 `.env` 中覆盖：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `DOCKER_LIBRARY_PREFIX` | `docker.m.daocloud.io/library` | 设为 `docker.io/library` 使用官方源 |
| `PIP_INDEX_URL` | `https://pypi.org/simple` | Python 包索引 |
| `NPM_REGISTRY` | `https://registry.npmmirror.com/` | 前端 npm 源 |

PaddleOCR 的模型权重在首次使用 OCR 时下载，保存在 `pymodel_paddle_models` 卷中。

### 常用命令

```bash
docker compose ps                        # 服务状态
docker compose logs -f backend           # 后端日志，frontend 同理
docker compose up -d --build backend     # 改了后端代码：重建并重启
docker compose up -d backend             # 只改了 .env：重新创建容器
docker compose restart backend           # 重启（不会重新读取 .env）
docker compose down                      # 停止并移除容器，数据保留
```

不要给 `down` 加 `-v`，它会删除 OCR 模型卷，下次 OCR 需要重新下载权重。

重置全部业务数据（账户、知识库、文档、索引、会话都会丢失）：

```bash
docker compose down && rm -rf storage && mkdir -p storage && docker compose up -d
```

数据库 schema 版本不匹配导致后端拒绝启动时，也用这个办法。后端默认以 `1000:1000` 运行，仅当日志报告 `storage` 权限错误时，才在 `.env` 设置 `APP_UID`、`APP_GID` 为 `id -u`、`id -g` 的输出并重新构建后端。

## 配置

所有可调项都在 [`.env.example`](.env.example) 中有注释，常用的有：

| 变量 | 说明 |
| --- | --- |
| `MODEL_API_KEY_ENCRYPTION_KEY` | 模型 API Key 的加密主密钥，至少 32 个字符，必填 |
| `INDEX_WORKER_THREADS` | 同时处理的文档数，1~8，默认 3 |
| `OCR_MAX_CONCURRENCY` / `MINERU_MAX_CONCURRENCY` / `EMBEDDING_MAX_CONCURRENCY` / `VISION_GLOBAL_MAX_CONCURRENCY` | 各类外部资源的并发上限，接口限流严格时调低 |
| `RAG_IMAGE_ANALYSIS` | 图片分析策略：`auto`（视觉模型，失败回退 OCR）、`ocr`、`disabled` |
| `MINERU_API_TOKEN` | 仅在选择 MinerU 云端解析时需要 |
| `CHAT_RETRIEVAL_PLANNING` | 让聊天模型通过 function calling 决定是否检索，默认关闭 |
| `CHAT_QUERY_VARIANT` | 额外生成一条书面化补充查询，默认关闭 |

支持的文档类型：PDF、DOCX、Markdown、TXT、PNG、JPG、JPEG，单文件不超过 50 MiB。

## 数据去向

| 用途 | 默认实现 | 数据去向 |
| --- | --- | --- |
| 向量嵌入、重排序 | SiliconFlow BGE 模型 | 文档切片、查询和候选文本发送到 SiliconFlow |
| 聊天模型 | 模型管理中启用的兼容模型 | 问题、上下文和可选图片发送到对应服务商 |
| 云端解析 | MinerU（按文档选择） | 原始 PDF 或 DOCX 发送到 MinerU |
| 图片解析 | 视觉模型或 PaddleOCR | 视觉模型接收图片；本地 OCR 不发送文件 |

不希望文件外发时，PDF 选择本地文本或 PaddleOCR，图片分析设为 `ocr`。模型 API Key 加密后存入 SQLite，不要分享 `storage/`、`.env` 或真实密钥。备份时整体复制 `storage/`，并另外保管 `MODEL_API_KEY_ENCRYPTION_KEY`，只有数据库而没有该值时无法恢复已保存的 API Key。

## 开发与测试

```bash
./scripts/check.sh                                              # 完整检查：镜像构建、Ruff、Pyright、后端测试、前端检查、端到端
SMOKE_ADMIN_PASSWORD='<管理员密码>' ./scripts/smoke-test.sh     # 部署冒烟测试
./evaluation/run_docker.sh                                      # 检索评测，需要 SILICONFLOW_API_KEY
./scripts/clean-workspace.sh                                    # 清理缓存与构建产物
```

检查全部在容器内运行。后端约 30 个测试文件，覆盖鉴权、索引并发与任务恢复、检索与上下文恢复、文档解析、模型密钥加密、OpenAPI 契约等；前端有 Node 单元测试和 Playwright 端到端测试。`contracts/openapi.json` 是接口快照，修改接口后用 `scripts/openapi_snapshot.py` 更新。

## 许可证

[MIT](LICENSE)
