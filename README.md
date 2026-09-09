# VerdantFlare App Image

> **状态：草案（待审批）**  
> **创建时间**：2026-09-09  
> **产品归属**：VerdantFlare Station 执行与存储平面

VerdantFlare App Image 是部署在 VerdantFlare Station 上的 AI 图像资产制作与编辑应用套件。它通过单一的领域级 MCP（Model Context Protocol）服务收口上游商业接口，并采用**“本地脚本直测先行、核心逻辑模块化、生产统一镜像分发”**的工程范式。

系统**不依赖任何本地 CLI（不通过 Codex CLI 或 Gemini CLI）**，底层全面采用标准 HTTP API 中继，内置**双核 API 驱动**：

1. **Codex 图像引擎**：基于 `OPENAI_BASE_URL` + `OPENAI_API_KEY`，调用 OpenAI Responses API / Images Edits API，权威出图模型指定为 **`gpt-image-2`**；
2. **Gemini 图像引擎**：基于 `ANTHROPIC_BASE_URL` + `ANTHROPIC_AUTH_TOKEN`，调用 Anthropic Messages 协议中继接口（`/v1/messages`），权威出图模型指定为 **`gemini-3.1-flash-image`**，内建纯净图与多模态指令编辑支持。

---

## 1. 架构定位：告别套娃，单微服务收敛

生图与本地重型 GPU 推理（如 MiniMax H3、UVR5）不同，底层对接的是云端 API/中继。因此，本套件**彻底废弃多层中继微服务的过度设计，也不依赖本地 CLI 子进程**，全仓库只维护**一个服务、一个镜像**：`image-mcp-server`。

```mermaid
flowchart TD
    subgraph Client[业务编排与智能体客户端]
        Agent[Antigravity / Coding Agent]
        Studio[VerdantFlare Studio]
        Pipeline[Music-MV 编排工作流]
    end

    subgraph Service[唯一生产微服务: image-mcp-server (Port 8000)]
        direction TB
        Server[MCP Server 网关 & Bearer 鉴权]
        Artifacts[ArtifactStore 产物管理与哈希校验]

        subgraph Providers[内置双核 API 驱动模块]
            CodexP[providers/codex.py<br/>OpenAI Responses / gpt-image-2]
            GeminiP[providers/gemini.py<br/>Anthropic Messages / gemini-3.1-flash-image]
        end

        Server --> Artifacts
        Server --> Providers
    end

    subgraph External[云端 API 中继]
        Sub2OpenAI[OpenAI 中继: OPENAI_BASE_URL]
        Sub2Anthropic[Anthropic 中继: ANTHROPIC_BASE_URL]
    end

    subgraph Storage[持久化存储]
        PVC[hostpath PVC: /data/projects/{project_id}]
    end

    Client -->|标准 Streamable HTTP / MCP 协议| Server
    CodexP -->|HTTP POST /v1/responses| Sub2OpenAI
    GeminiP -->|HTTP POST /v1/messages| Sub2Anthropic
    Artifacts -->|原子落盘| PVC
```

---

## 2. 研发与测试工作流：本地脚本直测先行，再编 MCP 镜像

生图链路不依赖任何本地 CLI 工具，全链路基于标准 HTTP API 协议。为避免“线上改一行、打一次镜像、部署看日志”的高昂调试成本，代码严格实行**“核心驱动模块化、纯 Python 独立脚本本地直测，服务层直接复用”**的工程范式：

```text
verdantflare-app-image/
├── README.md
├── scripts/
│   ├── test_codex_image.py           # 本地独立调试脚本：直连 OPENAI_BASE_URL 验证 Responses/Edits 出图与 4K 回退
│   ├── test_gemini_image.py          # 本地独立调试脚本：直连 ANTHROPIC_BASE_URL 验证 gemini-3.1-flash-image 出图
│   └── acceptance-image-workflow.sh  # 集群端到端与 MCP 契约验收脚本
└── services/
    └── image-mcp-server/             # 唯一的生产发布工程与 Docker 镜像
        ├── Dockerfile
        ├── requirements.txt
        ├── tests/
        │   ├── test_artifacts.py     # 存储与 SHA-256 校验单测
        │   ├── test_tasks.py         # 任务状态流转单测
        │   ├── test_gemini_provider.py # GeminiProvider 双通道鉴权与数据解码单测
        │   └── test_server.py        # HTTP 与 MCP 接口单测
        └── src/
            ├── __init__.py
            ├── artifacts.py          # 不可变 ArtifactStore
            ├── tasks.py              # 异步任务管理
            ├── server.py             # Starlette + MCP SDK 接口暴露与 Bearer 鉴权
            └── providers/
                ├── __init__.py
                ├── codex.py          # 生产级 Codex 协议驱动（逻辑与本地测试脚本 100% 同构）
                └── gemini.py         # 生产级 Gemini 协议驱动（逻辑与本地测试脚本 100% 同构）
```

### 本地测试与发布闭环标准：

1. **第一步：本地直接运行 Python 脚本测试（零 CLI 依赖）**
   - 配置本地根目录 `.env`：
     ```bash
     OPENAI_BASE_URL="https://<openai-relay-endpoint>/v1"
     OPENAI_API_KEY="sk-..."
     ANTHROPIC_BASE_URL="https://<anthropic-relay-endpoint>"
     ANTHROPIC_AUTH_TOKEN="sk-..."
     ```
   - 测试 Codex 渠道：
     ```bash
     python3 scripts/test_codex_image.py "一张深圳国资云的科技风封面图" --size 2048x1152 --output /tmp/test-codex.png
     ```
   - 测试 Gemini 渠道：
     ```bash
     python3 scripts/test_gemini_image.py "未来赛博朋克城市的雨夜街道" --output /tmp/test-gemini.png
     ```
   - 确认图片生成完整、分辨率达标、无文本污染、格式正确。
2. **第二步：本地运行 MCP Server 单元测试**
   - 运行 `PYTHONPATH=services/image-mcp-server python3 -m unittest discover -s services/image-mcp-server/tests -p 'test_*.py'`，确认 Artifact 登记、路径隔离、Bearer 鉴权拦截与双通道鉴权逻辑全部通过。
3. **第三步：合并至 `release` 分支自动触发镜像构建**
   - 自动编译并推送唯一的生产镜像：`verdantflare-app:image-mcp-server-v0.1.0`。
4. **第四步：Kubernetes 滚动发布与端到端验收**
   - 部署至成都验证集群，运行 `acceptance-image-workflow.sh` 执行 MCP 级别冒烟。

---

## 3. Image MCP 工具契约 (v1)

`image-mcp-server` 暴露以下标准工具（入参严格规范，产物受控管理）：

### 3.1 `artifact.import`

导入外部已批准的素材，下载并校验 SHA-256 后转化为受控不可变 Artifact。

- **入参**：`project_id`、`source_url`（需在白名单内）、`filename`、`expected_sha256`。
- **输出**：内部登记的 `artifact_id`、媒体类型与下载相对路径。

### 3.2 `image.generate`

文本驱动原子生图，支持分镜、概念设计。

- **入参**：
  - `project_id` (string): 项目标识。
  - `idempotency_key` (string): 幂等唯一键（建议 `<unit_id>/<attempt_id>`）。
  - `engine` (string): 指定底层引擎（`"codex"` 或 `"gemini"`，默认 `"gemini"`）。
  - `model` (string, 可选): 指定具体模型（Codex 默认 `gpt-image-2`，Gemini 默认 `gemini-3.1-flash-image`）。
  - `prompt` (string): 提示词。
  - `aspect_ratio` (string): 画幅比例（`"16:9"`, `"9:16"`, `"1:1"`, `"4:3"`, `"3:4"`，默认 `"16:9"`）。
  - `resolution` (string): 分辨率（`"2k"` 或 `"4k"`，默认 `"2k"`）。
  - `quality` (string): 质量偏好（`"auto"`, `"standard"`, `"hd"`）。
- **输出**：`task_id`、初始状态与时间戳。

### 3.3 `image.edit`

以图生图 / 图像编辑（角色服装、微调风格、姿态迁移）。

- **入参**：`project_id`、`idempotency_key`、`engine`、`source_artifact_id`、`prompt`。
- **输出**：`task_id`。

### 3.4 `image.inpaint`

局部遮罩重绘（人脸微调、杂物擦除、局部服装替换）。

- **入参**：`project_id`、`idempotency_key`、`source_artifact_id`、`mask_artifact_id`、`prompt`。
- **输出**：`task_id`。

### 3.5 `image.status` & `image.result`

- **`image.status`**：异步查询当前进度。
- **`image.result`**：任务完成时获取结果，返回不可变 `artifact_id`、长宽尺寸、格式、SHA-256 以及下载接口 `/image/artifacts/{id}/content`。

### 3.6 `image.list`

分页与多条件检索任务列表，支持状态机审计。

- **入参**：
  - `project_id` (string, 可选): 按项目筛选。
  - `engine` (string, 可选): 按引擎筛选（`"codex"` 或 `"gemini"`）。
  - `status` (string, 可选): 按状态筛选（`"queued"`, `"running"`, `"completed"`, `"failed"`, `"canceled"`）。
  - `limit` (int, 默认 50): 分页大小。
  - `offset` (int, 默认 0): 分页偏移。
- **输出**：`tasks` 数组与 `total` 总数。

---

## 4. 排队机制与任务持久化设计

为了确保高并发下不压垮上游 API 中继（防 429），并保障服务在意外重启或 Pod 重建后任务不丢失，系统内置轻量自闭环的排队与持久化体系：

### 4.1 SQLite WAL 任务持久化
- **存储位置**：`${IMAGE_ARTIFACT_ROOT}/tasks.db`，落在持久化 PVC（`/data`）上，天然持久化。
- **并发与安全**：启用 SQLite WAL 模式（`PRAGMA journal_mode=WAL;`），单写多读无阻塞，毫秒级响应。
- **任务模型**：单表记录 `task_id`、`project_id`、`idempotency_key`、`engine`、`model`、`prompt_preview`、`status`、`duration_seconds`、`artifact_id`、`error` 与全量请求参数。
- **重启自愈**：服务启动生命周期自动执行 `recover_hanging_tasks()`，将此前因 Pod 异常退出遗留在 `running` 的未完结任务标为 `failed` 并附带说明，同时自动恢复重放 `queued` 待处理任务，杜绝状态死锁。

### 4.2 双核通道并发隔离与背压队列 (TaskQueueManager)
- **通道并发隔离**：使用独立的 `asyncio.Semaphore` 限制各引擎最大并发请求数：
  - `CODEX_MAX_CONCURRENCY`（默认 2 并发）
  - `GEMINI_MAX_CONCURRENCY`（默认 5 并发）
- **背压缓冲**：当瞬间提交大量分镜出图任务时，超出并发槽位的任务自动挂在内存 `asyncio.Queue` 与 SQLite `status='queued'` 中按 FIFO 顺序排队等待，避免触发中继限流封禁。

---

## 5. 内置可视化任务看板 (Web Dashboard)

服务原生内置只读监控看板与统计接口，方便研发与运维直观掌控实时排队、耗时分布与图片产物：

- **看板访问地址**：`GET /dashboard`（支持在 URL 追加 `?token=...` 或在页面弹窗填入 Token）。
- **页面功能**：
  1. **实时指标卡片**：排队中（Queued）、执行中（Running）、已完成（Completed）、已失败（Failed）、平均生成耗时（Avg Duration）。
  2. **任务列表表格**：展示任务 ID、引擎/模型、提示词摘要、画幅分辨率、实际耗时与状态彩色徽标。
  3. **实时轮询与过滤**：支持 3 秒自动轮询（可暂停），支持按项目 ID、引擎类型、执行状态组合筛选。
  4. **大图弹窗与一键下载**：点击“查看”直接弹出高保真图片预览，展示完整 Prompt、SHA-256 哈希与原图下载直链。
- **监控与列表 API**：
  - `GET /api/tasks`：返回符合过滤条件的分页任务列表 JSON。
  - `GET /api/tasks/stats`：返回聚合指标统计 JSON。

---

## 6. 环境变量与安全配置

生产环境所有凭据均由 Kubernetes Secret（`image-mcp-auth`）注入，严禁暴露在客户端或代码库：

| 环境变量                     | 默认值                                     | 用途                                      |
| :--------------------------- | :----------------------------------------- | :---------------------------------------- |
| `IMAGE_ARTIFACT_ROOT`        | `/data/projects`                           | 持久化存储根目录                          |
| `IMAGE_ASSET_IMPORT_ORIGINS` | 未设置                                     | 允许导入外部素材的 HTTPS 域名白名单       |
| `IMAGE_MCP_BEARER_TOKEN`     | 未设置                                     | 保护 MCP 接口与产物下载的全局密钥（必填） |
| `IMAGE_MCP_ALLOWED_HOSTS`    | `127.0.0.1:*`                              | DNS Rebinding 防护 Host 白名单            |
| `IMAGE_MCP_ALLOWED_ORIGINS`  | `http://127.0.0.1:*`                       | DNS Rebinding 防护 Origin 白名单          |
| `IMAGE_QUEUE_WORKERS`        | `8`                                        | 队列 Worker 协程消费池大小                |
| `CODEX_MAX_CONCURRENCY`      | `2`                                        | Codex 渠道最大并发限制                    |
| `GEMINI_MAX_CONCURRENCY`     | `5`                                        | Gemini 渠道最大并发限制                   |
| `OPENAI_BASE_URL`            | -                                          | Codex 生图中继 API 根地址（由环境或 Secret 注入）                 |
| `OPENAI_API_KEY`             | -                                          | Codex 生图 API 凭据（由 Secret 注入）     |
| `ANTHROPIC_BASE_URL`         | -                                          | Gemini 生图中继 API 根地址（由环境或 Secret 注入）                |
| `ANTHROPIC_AUTH_TOKEN`       | -                                          | Gemini 生图 API 凭据（由 Secret 注入）    |
| `HTTPS_PROXY`                | -                                          | 可选的企业出网代理                        |

---

## 7. 验证环境与客户端 MCP 接入

### 7.1 集群部署拓扑

- **命名空间**：`verdantflare-image`。
- **清单存储**：声明式配置统一归档于 `deploys/` 对应集群目录。
- **存储卷**：固定节点 `hostpath` PVC（`image-projects` -> `/data/volumes/image-projects`）。
- **服务入口**：通过网关 Ingress 暴露标准 MCP 端点：
  ```text
  POST ${IMAGE_MCP_PUBLIC_BASE_URL}
  # 网关端点形如：POST /image
  ```

### 7.2 客户端标准 MCP 接入配置

不依赖任何本地 CLI，所有智能体、工作流编排器与 IDE 扩展通过标准 Streamable HTTP / SSE 协议直连：

#### A. 声明式 MCP 客户端配置

在智能体或 IDE MCP 配置文件（如 `mcp_config.json`）中注册：

```json
{
  "mcpServers": {
    "verdantflare-image": {
      "serverUrl": "${IMAGE_MCP_PUBLIC_BASE_URL}",
      "headers": {
        "Authorization": "Bearer ${IMAGE_MCP_BEARER_TOKEN}"
      }
    }
  }
}
```

#### B. 业务工作流与 API 直连

上层业务套件（如 `verdantflare-music-mv`）可直接通过 HTTP POST 向 `${IMAGE_MCP_PUBLIC_BASE_URL}` 发送标准 MCP 工具调用（`image.generate`、`image.edit`、`image.inpaint`），并通过返回的 `/artifacts/{artifact_id}/content` 获取已落盘的受控图像资产。

客户端统一配合安装 `verdantflare-skills/skills/verdantflare-image` 即可获得完全受控的原子图像生产能力。
