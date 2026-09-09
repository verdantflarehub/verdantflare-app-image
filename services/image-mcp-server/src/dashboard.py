from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Any
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response

from .artifacts import ArtifactStore
from .tasks import TaskStore

STATIC_HTML_PATH = Path(__file__).parent / "static" / "index.html"

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>VerdantFlare Image Station - 任务监控看板</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      darkMode: 'class',
      theme: {
        extend: {
          colors: {
            brand: { 50: '#ecfdf5', 500: '#10b981', 600: '#059669', 900: '#064e3b' }
          }
        }
      }
    }
  </script>
  <style>
    @keyframes pulse-slow { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    .animate-pulse-slow { animation: pulse-slow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
    <!-- Header -->
    <header class="flex flex-col md:flex-row md:items-center md:justify-between pb-6 border-b border-slate-800 gap-4">
      <div>
        <div class="flex items-center gap-3">
          <span class="inline-flex items-center justify-center p-2 bg-emerald-500/10 text-emerald-400 rounded-lg border border-emerald-500/20">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
            </svg>
          </span>
          <div>
            <h1 class="text-xl font-bold tracking-tight text-white">VerdantFlare Image Station</h1>
            <p class="text-xs text-slate-400">异步任务排队与双核生图状态监控</p>
          </div>
        </div>
      </div>

      <div class="flex items-center gap-3">
        <div class="flex items-center gap-2 bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-300">
          <span id="pollIndicator" class="w-2 h-2 rounded-full bg-emerald-500 animate-ping"></span>
          <span id="pollText">实时轮询中</span>
        </div>
        <button id="togglePollBtn" onclick="togglePolling()" class="px-3 py-1.5 text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg border border-slate-700 transition">
          暂停刷新
        </button>
        <button onclick="fetchData()" class="px-3 py-1.5 text-xs bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-medium transition flex items-center gap-1.5">
          <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg>
          立即刷新
        </button>
      </div>
    </header>

    <!-- Stats Cards -->
    <div class="grid grid-cols-2 md:grid-cols-5 gap-4 my-6">
      <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4">
        <div class="text-xs font-medium text-slate-400">⏳ 排队中 (Queued)</div>
        <div class="text-2xl font-bold mt-1 text-amber-400" id="statQueued">0</div>
      </div>
      <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4">
        <div class="text-xs font-medium text-slate-400">⚡ 执行中 (Running)</div>
        <div class="text-2xl font-bold mt-1 text-sky-400" id="statRunning">0</div>
      </div>
      <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4">
        <div class="text-xs font-medium text-slate-400">✅ 已完成 (Completed)</div>
        <div class="text-2xl font-bold mt-1 text-emerald-400" id="statCompleted">0</div>
      </div>
      <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4">
        <div class="text-xs font-medium text-slate-400">❌ 已失败 (Failed)</div>
        <div class="text-2xl font-bold mt-1 text-rose-400" id="statFailed">0</div>
      </div>
      <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 col-span-2 md:col-span-1">
        <div class="text-xs font-medium text-slate-400">⏱️ 平均生成耗时</div>
        <div class="text-2xl font-bold mt-1 text-purple-400" id="statAvgDuration">0.0s</div>
      </div>
    </div>

    <!-- Filter Bar -->
    <div class="bg-slate-900/40 border border-slate-800/60 rounded-xl p-3.5 mb-6 flex flex-wrap items-center justify-between gap-3 text-xs">
      <div class="flex flex-wrap items-center gap-3">
        <div class="flex items-center gap-1.5">
          <span class="text-slate-400 font-medium">状态:</span>
          <select id="filterStatus" onchange="fetchData()" class="bg-slate-800 border border-slate-700 text-slate-200 rounded-lg px-2.5 py-1 focus:outline-none focus:border-emerald-500">
            <option value="">全部状态</option>
            <option value="queued">排队中</option>
            <option value="running">执行中</option>
            <option value="completed">已完成</option>
            <option value="failed">已失败</option>
          </select>
        </div>

        <div class="flex items-center gap-1.5">
          <span class="text-slate-400 font-medium">引擎:</span>
          <select id="filterEngine" onchange="fetchData()" class="bg-slate-800 border border-slate-700 text-slate-200 rounded-lg px-2.5 py-1 focus:outline-none focus:border-emerald-500">
            <option value="">全部引擎</option>
            <option value="gemini">Gemini</option>
            <option value="codex">Codex</option>
          </select>
        </div>

        <div class="flex items-center gap-1.5">
          <span class="text-slate-400 font-medium">项目:</span>
          <input type="text" id="filterProject" placeholder="项目 ID..." onkeyup="if(event.key==='Enter') fetchData()" class="bg-slate-800 border border-slate-700 text-slate-200 rounded-lg px-2.5 py-1 placeholder-slate-500 focus:outline-none focus:border-emerald-500 w-32">
        </div>
      </div>

      <div class="text-slate-500 text-xs" id="totalTasksCount">共 0 条任务记录</div>
    </div>

    <!-- Tasks Table -->
    <div class="bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs text-slate-300">
          <thead class="bg-slate-900/90 text-slate-400 uppercase font-semibold border-b border-slate-800 tracking-wider">
            <tr>
              <th class="px-4 py-3">任务 ID</th>
              <th class="px-4 py-3">引擎 / 模型</th>
              <th class="px-4 py-3">提示词摘要</th>
              <th class="px-4 py-3">状态</th>
              <th class="px-4 py-3">耗时</th>
              <th class="px-4 py-3">创建时间</th>
              <th class="px-4 py-3 text-right">操作</th>
            </tr>
          </thead>
          <tbody id="tasksBody" class="divide-y divide-slate-800/60">
            <tr>
              <td colspan="7" class="px-4 py-8 text-center text-slate-500">加载中...</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Detail Modal -->
  <div id="detailModal" class="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm hidden flex items-center justify-center p-4">
    <div class="bg-slate-900 border border-slate-800 rounded-2xl max-w-2xl w-full p-6 shadow-2xl relative max-h-[90vh] overflow-y-auto">
      <button onclick="closeModal()" class="absolute top-4 right-4 text-slate-400 hover:text-white p-1 rounded-lg hover:bg-slate-800 transition">
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
      </button>
      <h3 class="text-base font-bold text-white mb-4 flex items-center gap-2">
        <span>任务详情</span>
        <span id="modalTaskIdBadge" class="text-xs font-mono font-normal text-slate-400"></span>
      </h3>

      <div class="space-y-4 text-xs">
        <div id="modalImageContainer" class="rounded-xl overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center min-h-[220px] max-h-[380px]">
          <img id="modalImg" src="" alt="Output Preview" class="max-h-[380px] w-auto object-contain hidden" />
          <div id="modalImgPlaceholder" class="text-slate-500">暂无图片产物</div>
        </div>

        <div class="grid grid-cols-2 gap-3 bg-slate-950/60 p-3 rounded-xl border border-slate-800/80 font-mono text-[11px]">
          <div><span class="text-slate-500">引擎:</span> <span id="modalEngine" class="text-slate-200"></span></div>
          <div><span class="text-slate-500">模型:</span> <span id="modalModel" class="text-slate-200"></span></div>
          <div><span class="text-slate-500">耗时:</span> <span id="modalDuration" class="text-slate-200"></span></div>
          <div><span class="text-slate-500">状态:</span> <span id="modalStatus" class="text-slate-200"></span></div>
          <div class="col-span-2 truncate"><span class="text-slate-500">Artifact ID:</span> <span id="modalArtifactId" class="text-emerald-400"></span></div>
        </div>

        <div>
          <div class="text-slate-400 font-medium mb-1">提示词 (Prompt):</div>
          <div id="modalPrompt" class="bg-slate-950 p-3 rounded-lg border border-slate-800 text-slate-300 whitespace-pre-wrap max-h-28 overflow-y-auto"></div>
        </div>

        <div id="modalErrorBox" class="hidden">
          <div class="text-rose-400 font-medium mb-1">错误原因:</div>
          <div id="modalError" class="bg-rose-950/40 border border-rose-900/60 p-3 rounded-lg text-rose-300 font-mono"></div>
        </div>

        <div class="flex justify-end gap-2 pt-2">
          <a id="modalDownloadBtn" href="#" target="_blank" class="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white rounded-lg font-medium transition hidden">
            下载原图
          </a>
        </div>
      </div>
    </div>
  </div>

  <script>
    let pollInterval = null;
    let isPolling = true;

    function getAuthToken() {
      const urlParams = new URLSearchParams(window.location.search);
      return urlParams.get('token') || localStorage.getItem('image_mcp_token') || '';
    }

    function getHeaders() {
      const token = getAuthToken();
      return token ? { 'Authorization': `Bearer ${token}` } : {};
    }

    async function fetchData() {
      const status = document.getElementById('filterStatus').value;
      const engine = document.getElementById('filterEngine').value;
      const project = document.getElementById('filterProject').value.trim();

      const params = new URLSearchParams({ limit: '50', offset: '0' });
      if (status) params.append('status', status);
      if (engine) params.append('engine', engine);
      if (project) params.append('project_id', project);

      try {
        const [tasksResp, statsResp] = await Promise.all([
          fetch(`/api/tasks?${params.toString()}`, { headers: getHeaders() }),
          fetch('/api/tasks/stats', { headers: getHeaders() })
        ]);

        if (tasksResp.status === 401 || statsResp.status === 401) {
          const userToken = prompt('请输入访问鉴权 Token (IMAGE_MCP_BEARER_TOKEN):');
          if (userToken) {
            localStorage.setItem('image_mcp_token', userToken.trim());
            fetchData();
          }
          return;
        }

        const tasksData = await tasksResp.json();
        const statsData = await statsResp.json();

        updateStats(statsData);
        renderTasks(tasksData.tasks || []);
        document.getElementById('totalTasksCount').innerText = `共 ${tasksData.total || 0} 条任务记录`;
      } catch (err) {
        console.error('获取数据失败:', err);
      }
    }

    function updateStats(s) {
      document.getElementById('statQueued').innerText = s.queued || 0;
      document.getElementById('statRunning').innerText = s.running || 0;
      document.getElementById('statCompleted').innerText = s.completed || 0;
      document.getElementById('statFailed').innerText = s.failed || 0;
      document.getElementById('statAvgDuration').innerText = `${s.avg_duration || 0}s`;
    }

    function renderTasks(tasks) {
      const tbody = document.getElementById('tasksBody');
      if (!tasks.length) {
        tbody.innerHTML = '<tr><td colspan="7" class="px-4 py-8 text-center text-slate-500">暂无任务记录</td></tr>';
        return;
      }

      tbody.innerHTML = tasks.map(t => {
        const badgeColor = {
          'queued': 'bg-amber-500/10 text-amber-400 border-amber-500/20',
          'running': 'bg-sky-500/10 text-sky-400 border-sky-500/20 animate-pulse',
          'completed': 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
          'failed': 'bg-rose-500/10 text-rose-400 border-rose-500/20',
          'canceled': 'bg-slate-500/10 text-slate-400 border-slate-500/20',
        }[t.status] || 'bg-slate-800 text-slate-400';

        const statusLabel = {
          'queued': '⏳ 排队中',
          'running': '⚡ 执行中',
          'completed': '✅ 已完成',
          'failed': '❌ 失败',
          'canceled': '⏹️ 已取消'
        }[t.status] || t.status;

        const dur = t.duration_seconds > 0 ? `${t.duration_seconds.toFixed(1)}s` : '--';
        const dateStr = t.created_at ? new Date(t.created_at).toLocaleTimeString() : '--';
        const engineLabel = t.engine === 'gemini' ? 'Gemini' : 'Codex';
        const modelLabel = t.model || (t.engine === 'gemini' ? 'gemini-3.1-f' : 'gpt-image-2');

        return `
          <tr class="hover:bg-slate-800/40 transition">
            <td class="px-4 py-3 font-mono font-medium text-slate-300">${t.task_id}</td>
            <td class="px-4 py-3">
              <span class="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800 text-slate-300 border border-slate-700">
                ${engineLabel} · ${modelLabel}
              </span>
            </td>
            <td class="px-4 py-3 max-w-xs truncate text-slate-400" title="${t.prompt_preview || ''}">
              ${t.prompt_preview || '--'}
            </td>
            <td class="px-4 py-3">
              <span class="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium border ${badgeColor}">
                ${statusLabel}
              </span>
            </td>
            <td class="px-4 py-3 font-mono text-slate-400">${dur}</td>
            <td class="px-4 py-3 text-slate-400">${dateStr}</td>
            <td class="px-4 py-3 text-right">
              <button onclick='viewTask(${JSON.stringify(t)})' class="px-2.5 py-1 text-xs bg-slate-800 hover:bg-slate-700 text-slate-200 rounded border border-slate-700 transition">
                查看
              </button>
            </td>
          </tr>
        `;
      }).join('');
    }

    function viewTask(t) {
      document.getElementById('modalTaskIdBadge').innerText = t.task_id;
      document.getElementById('modalEngine').innerText = t.engine;
      document.getElementById('modalModel').innerText = t.model || '--';
      document.getElementById('modalDuration').innerText = t.duration_seconds ? `${t.duration_seconds.toFixed(2)}s` : '--';
      document.getElementById('modalStatus').innerText = t.status;
      document.getElementById('modalArtifactId').innerText = t.artifact_id || '未生成';
      document.getElementById('modalPrompt').innerText = (t.request_params && t.request_params.prompt) || t.prompt_preview || '--';

      const img = document.getElementById('modalImg');
      const placeholder = document.getElementById('modalImgPlaceholder');
      const dlBtn = document.getElementById('modalDownloadBtn');
      const errBox = document.getElementById('modalErrorBox');

      if (t.artifact_id) {
        const token = getAuthToken();
        const srcUrl = `/artifacts/${t.artifact_id}/content` + (token ? `?token=${token}` : '');
        img.src = srcUrl;
        img.classList.remove('hidden');
        placeholder.classList.add('hidden');
        dlBtn.href = srcUrl;
        dlBtn.classList.remove('hidden');
      } else {
        img.classList.add('hidden');
        placeholder.classList.remove('hidden');
        dlBtn.classList.add('hidden');
      }

      if (t.error) {
        document.getElementById('modalError').innerText = t.error;
        errBox.classList.remove('hidden');
      } else {
        errBox.classList.add('hidden');
      }

      document.getElementById('detailModal').classList.remove('hidden');
    }

    function closeModal() {
      document.getElementById('detailModal').classList.add('hidden');
    }

    function togglePolling() {
      isPolling = !isPolling;
      const indicator = document.getElementById('pollIndicator');
      const text = document.getElementById('pollText');
      const btn = document.getElementById('togglePollBtn');

      if (isPolling) {
        indicator.className = 'w-2 h-2 rounded-full bg-emerald-500 animate-ping';
        text.innerText = '实时轮询中';
        btn.innerText = '暂停刷新';
        startPolling();
      } else {
        indicator.className = 'w-2 h-2 rounded-full bg-slate-500';
        text.innerText = '已暂停';
        btn.innerText = '开启轮询';
        clearInterval(pollInterval);
      }
    }

    function startPolling() {
      clearInterval(pollInterval);
      pollInterval = setInterval(fetchData, 3000);
    }

    document.addEventListener('DOMContentLoaded', () => {
      fetchData();
      startPolling();
    });
  </script>
</body>
</html>
"""


def check_auth(request: Request) -> bool:
    token = os.environ.get("IMAGE_MCP_BEARER_TOKEN", "").strip()
    if not token:
        return True

    # 1. Check Header
    auth_header = request.headers.get("authorization", "")
    if auth_header and hmac.compare_digest(auth_header, f"Bearer {token}"):
        return True

    # 2. Check Query Param
    query_token = request.query_params.get("token", "")
    if query_token and hmac.compare_digest(query_token, token):
        return True

    # 3. Check X-MCP-Token Header
    x_token = request.headers.get("x-mcp-token", "")
    if x_token and hmac.compare_digest(x_token, token):
        return True

    return False


async def dashboard_page(request: Request) -> Response:
    if STATIC_HTML_PATH.exists():
        content = STATIC_HTML_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content)
    return HTMLResponse(DASHBOARD_HTML)


async def api_list_tasks(request: Request) -> Response:
    if not check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    tasks_store = request.app.state.tasks
    project_id = request.query_params.get("project_id")
    engine = request.query_params.get("engine")
    status = request.query_params.get("status")
    limit = int(request.query_params.get("limit", "50"))
    offset = int(request.query_params.get("offset", "0"))

    records, total = tasks_store.list_tasks(
        project_id=project_id,
        engine=engine,
        status=status,
        limit=limit,
        offset=offset,
    )
    return JSONResponse({
        "tasks": [r.to_dict() for r in records],
        "total": total,
        "limit": limit,
        "offset": offset,
    })


async def api_task_stats(request: Request) -> Response:
    if not check_auth(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)

    tasks_store = request.app.state.tasks
    stats = tasks_store.get_stats()
    return JSONResponse(stats)

