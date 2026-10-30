<template>
  <div class="flex-1 overflow-y-auto p-6 bg-[#f6f9f9] w-full text-slate-800">
    <div class="mb-6 flex items-end justify-between">
      <div><p class="text-[11px] font-semibold text-[#649092]">运维</p><h2 class="mt-1 text-lg font-semibold text-[#1e3437]">运行监控</h2></div>
      <div class="flex items-center gap-3">
        <label class="flex items-center gap-1.5 text-xs text-[#819497]">
          <span>统计范围</span>
          <select v-model.number="days" aria-label="统计范围" class="h-8 rounded-md border border-gray-200 bg-white px-2 text-xs text-gray-700" @change="loadOverview">
            <option :value="1">最近 1 天</option>
            <option :value="7">最近 7 天</option>
            <option :value="30">最近 30 天</option>
          </select>
        </label>
        <button type="button" class="flex h-8 items-center gap-1.5 rounded-md border border-gray-200 bg-white px-3 text-xs text-gray-600 hover:bg-gray-50" @click="refreshAll">
          <RefreshCw class="h-3.5 w-3.5" aria-hidden="true" /> 刷新
        </button>
      </div>
    </div>

    <div class="mb-4 flex flex-wrap items-center gap-2 text-xs" aria-label="依赖状态">
      <span class="text-[#819497]">依赖状态</span>
      <span
        v-for="dependency in dependencies"
        :key="dependency.name"
        :class="dependency.ok ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-600'"
        :title="dependency.detail"
        class="rounded px-2 py-1"
      >
        {{ dependencyLabel(dependency.name) }} {{ dependency.ok ? '正常' : '异常' }} · {{ formatMs(dependency.latency_ms) }}
        <span v-if="!dependency.ok" class="break-all">（{{ dependency.detail }}）</span>
      </span>
      <span v-if="!dependencies.length" class="text-gray-400">检查中…</span>
    </div>

    <div class="grid grid-cols-2 xl:grid-cols-4 gap-4 mb-4">
      <div v-for="card in cards" :key="card.title" class="bg-white border border-[#e1eaea] rounded-lg p-5 shadow-[0_1px_2px_rgba(18,53,57,0.04)]">
        <div class="text-xl font-semibold text-[#233b3e]">{{ card.value }}</div>
        <div class="text-xs text-[#819295] mt-1">{{ card.title }}</div>
        <div v-if="card.hint" class="text-[11px] text-gray-400 mt-1">{{ card.hint }}</div>
      </div>
    </div>
    <p v-if="overview.truncated" class="mb-4 text-xs text-amber-600">所选范围内记录过多，统计只覆盖最近 20000 条。</p>

    <div class="grid grid-cols-1 xl:grid-cols-2 gap-4 mb-4">
      <div class="bg-white border border-[#e1eaea] rounded-lg p-5 shadow-[0_1px_2px_rgba(18,53,57,0.04)]">
        <h3 class="font-semibold text-sm mb-3">按天问答数与总耗时 P95</h3>
        <p v-if="!overview.daily.length" class="text-xs text-gray-400">暂无记录</p>
        <ul v-else class="space-y-1.5" aria-label="按天问答趋势">
          <li v-for="point in overview.daily" :key="point.date" class="flex items-center gap-2 text-xs">
            <span class="w-20 text-gray-500 tabular-nums">{{ point.date }}</span>
            <div class="flex-1 h-2 rounded bg-gray-100">
              <div class="h-2 rounded bg-blue-500" :style="{ width: `${Math.round((point.runs / maxDailyRuns) * 100)}%` }"></div>
            </div>
            <span class="w-24 text-right tabular-nums text-gray-600">{{ point.runs }} 次<span v-if="point.errors" class="text-red-500">（失败 {{ point.errors }}）</span></span>
            <span class="w-16 text-right tabular-nums text-gray-400">{{ formatMs(point.p95_total_ms) }}</span>
          </li>
        </ul>
      </div>

      <div class="bg-white border border-[#e1eaea] rounded-lg p-5 shadow-[0_1px_2px_rgba(18,53,57,0.04)]">
        <h3 class="font-semibold text-sm mb-3">状态分布</h3>
        <ul class="space-y-1.5 text-xs">
          <li v-for="status in statusOptions" :key="status.value" class="flex items-center justify-between">
            <span class="text-gray-500">{{ status.label }}</span>
            <span class="tabular-nums text-gray-800">{{ overview.status_counts[status.value] || 0 }}</span>
          </li>
        </ul>
      </div>
    </div>

    <div class="bg-white border border-[#e1eaea] rounded-lg shadow-[0_1px_2px_rgba(18,53,57,0.04)] overflow-hidden mb-4">
      <div class="px-5 py-4 border-b border-[#e5eded] flex flex-wrap items-center justify-between gap-3">
        <h3 class="font-semibold text-sm">问答记录</h3>
        <div class="flex flex-wrap items-center gap-2 text-xs">
          <select v-model="filters.status" aria-label="状态过滤" class="h-8 rounded-md border border-gray-200 bg-white px-2" @change="applyFilters">
            <option value="">全部状态</option>
            <option v-for="status in statusOptions" :key="status.value" :value="status.value">{{ status.label }}</option>
          </select>
          <input v-model.trim="filters.kb" aria-label="知识库过滤" placeholder="知识库名称" maxlength="80" class="h-8 w-32 rounded-md border border-gray-200 px-2" @change="applyFilters" />
          <select v-model="filters.slowMs" aria-label="慢请求过滤" class="h-8 rounded-md border border-gray-200 bg-white px-2" @change="applyFilters">
            <option value="">不限耗时</option>
            <option value="3000">总耗时 ≥ 3 秒</option>
            <option value="10000">总耗时 ≥ 10 秒</option>
            <option value="30000">总耗时 ≥ 30 秒</option>
          </select>
          <select v-model="filters.feedback" aria-label="用户评价过滤" class="h-8 rounded-md border border-gray-200 bg-white px-2" @change="applyFilters">
            <option value="">不限评价</option>
            <option value="-1">👎 没用</option>
            <option value="1">👍 有用</option>
          </select>
          <button type="button" class="h-8 rounded-md border border-gray-200 bg-white px-3 text-gray-600 hover:bg-gray-50" title="把评为没用的问答导出为 evaluation/ 的题目格式" @click="exportDownvoted">导出 👎 为评测题</button>
        </div>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs text-gray-600">
          <thead class="bg-[#f6f9f9] text-[#6c8083] font-medium">
            <tr>
              <th class="px-4 py-3 font-medium">时间</th>
              <th class="px-4 py-3 font-medium">用户</th>
              <th class="px-4 py-3 font-medium">知识库</th>
              <th class="px-4 py-3 font-medium">问题（前 200 字）</th>
              <th class="px-4 py-3 font-medium">状态</th>
              <th class="px-4 py-3 font-medium text-right">检索</th>
              <th class="px-4 py-3 font-medium text-right">重排</th>
              <th class="px-4 py-3 font-medium text-right">首字</th>
              <th class="px-4 py-3 font-medium text-right">总耗时</th>
              <th class="px-4 py-3 font-medium">追踪</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-gray-100">
            <tr v-for="run in runs.items" :key="run.id">
              <td class="px-4 py-3 whitespace-nowrap tabular-nums">{{ formatTime(run.created_at) }}</td>
              <td class="px-4 py-3">{{ run.username }}</td>
              <td class="px-4 py-3">{{ run.kb_name }}</td>
              <td class="px-4 py-3 max-w-xs">
                <div class="truncate" :title="run.question_preview">{{ run.question_preview }}</div>
                <div v-if="run.error_summary" class="truncate text-red-500" :title="run.error_summary">{{ run.error_summary }}</div>
              </td>
              <td class="px-4 py-3"><span :class="statusClass(run.status)" class="rounded px-1.5 py-0.5">{{ statusLabel(run.status) }}</span></td>
              <td class="px-4 py-3 text-right tabular-nums">{{ formatMs(run.retrieval_ms) }}</td>
              <td class="px-4 py-3 text-right tabular-nums">{{ formatMs(run.rerank_ms) }}</td>
              <td class="px-4 py-3 text-right tabular-nums">{{ formatMs(run.first_token_ms) }}</td>
              <td class="px-4 py-3 text-right tabular-nums">{{ formatMs(run.total_ms) }}</td>
              <td class="px-4 py-3 whitespace-nowrap">
                <button v-if="run.trace_id" type="button" class="text-teal-700 underline" @click="openTrace(run.trace_id)">查看</button>
                <span v-else class="text-gray-300">-</span>
                <span v-if="run.feedback" class="ml-1">{{ run.feedback > 0 ? '👍' : '👎' }}</span>
              </td>
            </tr>
            <tr v-if="!runs.items.length"><td colspan="10" class="px-4 py-8 text-center text-gray-400">{{ runsLoading ? '加载中…' : '没有符合条件的记录' }}</td></tr>
          </tbody>
        </table>
      </div>
      <div class="px-5 pb-4">
        <PaginationControl
          :page="page"
          :page-size="pageSize"
          :total="runs.total"
          :total-pages="totalPages"
          :disabled="runsLoading"
          @change-page="changePage"
          @change-page-size="changePageSize"
        />
      </div>
    </div>

    <div class="grid grid-cols-1 xl:grid-cols-2 gap-4">
      <div class="bg-white border border-[#e1eaea] rounded-lg p-5 shadow-[0_1px_2px_rgba(18,53,57,0.04)]">
        <h3 class="font-semibold text-sm mb-3">最近失败的索引任务</h3>
        <p v-if="!jobs.length" class="text-xs text-gray-400">没有失败任务</p>
        <ul v-else class="space-y-2 text-xs">
          <li v-for="job in jobs" :key="job.id" class="border-b border-gray-100 pb-2">
            <div class="flex justify-between gap-2">
              <span class="font-medium text-gray-700 truncate">{{ job.kb_name }} / {{ job.filename || '（整库）' }}</span>
              <span class="text-gray-400 tabular-nums whitespace-nowrap">{{ formatTime(job.created_at) }}</span>
            </div>
            <div class="text-red-500 break-words">{{ job.error_message || '无错误信息' }}</div>
          </li>
        </ul>
      </div>

      <div class="bg-white border border-[#e1eaea] rounded-lg p-5 shadow-[0_1px_2px_rgba(18,53,57,0.04)]">
        <div class="mb-3 flex items-center justify-between">
          <h3 class="font-semibold text-sm">后端日志（末尾）</h3>
          <select v-model="logLevel" aria-label="日志级别" class="h-8 rounded-md border border-gray-200 bg-white px-2 text-xs" @change="loadLogs">
            <option value="">全部级别</option>
            <option v-for="level in logLevels" :key="level" :value="level">{{ level }}</option>
          </select>
        </div>
        <p v-if="!logs.available" class="text-xs text-gray-400">{{ logs.message || '暂无日志' }}</p>
        <pre v-else class="max-h-80 overflow-auto rounded bg-slate-900 p-3 text-[11px] leading-relaxed text-slate-100 whitespace-pre-wrap break-all" aria-label="后端日志内容">{{ logs.lines.join('\n') || '（没有符合条件的日志）' }}</pre>
      </div>
    </div>
    <div v-if="traceView.open" class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" @click.self="closeTrace" @keydown.esc="closeTrace">
      <div class="max-h-[85vh] w-full max-w-3xl overflow-y-auto rounded-lg bg-white p-5 shadow-xl" role="dialog" aria-label="问答追踪">
        <div class="mb-3 flex items-center justify-between">
          <h3 class="text-sm font-semibold">问答追踪</h3>
          <button type="button" class="text-gray-400 hover:text-gray-600" aria-label="关闭" @click="closeTrace">✕</button>
        </div>
        <ChatTraceDetail v-if="traceView.data" :trace="traceView.data" />
        <p v-else class="text-xs text-gray-400">加载中…</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue';
import { RefreshCw } from 'lucide-vue-next';

import { apiGet, authorizedFetch } from '../apiClient';
import ChatTraceDetail from '../components/ChatTraceDetail.vue';
import PaginationControl from '../components/PaginationControl.vue';
import { toast } from '../toast';

const statusOptions = [
  { value: 'answered', label: '已回答' },
  { value: 'no_evidence', label: '无依据' },
  { value: 'direct', label: '直接回答' },
  { value: 'interrupted', label: '中断' },
  { value: 'cancelled', label: '已取消' },
  { value: 'error', label: '失败' }
];
const logLevels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];

const dependencyLabels = { sqlite: 'SQLite', milvus: 'Milvus' };

const days = ref(7);
const dependencies = ref([]);
const overview = ref({ total: 0, truncated: false, status_counts: {}, error_rate: 0, latency: {}, daily: [] });
const runs = ref({ items: [], total: 0 });
const runsLoading = ref(false);
const filters = reactive({ status: '', kb: '', slowMs: '', feedback: '' });
const traceView = reactive({ open: false, data: null });
const page = ref(1);
const pageSize = ref(20);
const jobs = ref([]);
const logs = ref({ available: true, lines: [], message: '' });
const logLevel = ref('');

const totalPages = computed(() => Math.max(1, Math.ceil(runs.value.total / pageSize.value)));
const maxDailyRuns = computed(() => Math.max(1, ...overview.value.daily.map(point => point.runs)));

/** 毫秒数显示为 ms 或秒；缺失时显示短横线。 */
const formatMs = value => {
  if (value === null || value === undefined) return '-';
  return value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`;
};
const formatTime = value => (value || '').replace('T', ' ').slice(0, 19);
const dependencyLabel = name => dependencyLabels[name] || name;
const statusLabel = value => statusOptions.find(item => item.value === value)?.label || value;
const statusClass = value => ({
  answered: 'bg-green-50 text-green-600',
  direct: 'bg-green-50 text-green-600',
  no_evidence: 'bg-amber-50 text-amber-600',
  interrupted: 'bg-amber-50 text-amber-600',
  cancelled: 'bg-gray-100 text-gray-500',
  error: 'bg-red-50 text-red-600'
})[value] || 'bg-gray-100 text-gray-500';

const latencyText = key => {
  const item = overview.value.latency[key];
  return item ? `P50 ${formatMs(item.p50)} / P95 ${formatMs(item.p95)}` : '';
};
const cards = computed(() => [
  { title: '问答次数', value: String(overview.value.total) },
  { title: '失败率', value: `${(overview.value.error_rate * 100).toFixed(1)}%` },
  { title: '总耗时', value: formatMs(overview.value.latency.total?.p95), hint: latencyText('total') },
  { title: '首字耗时', value: formatMs(overview.value.latency.first_token?.p95), hint: latencyText('first_token') }
]);

/** 单个区块加载失败只提示，不影响其他区块。 */
const guarded = async (label, task) => {
  try {
    await task();
  } catch (error) {
    toast.error(`${label}加载失败：${error.message}`);
  }
};

const loadDependencies = () => guarded('依赖状态', async () => {
  dependencies.value = await apiGet('/api/ops/dependencies');
});
const loadOverview = () => guarded('概览', async () => {
  overview.value = await apiGet(`/api/ops/overview?days=${days.value}`);
});
const loadRuns = () => guarded('问答记录', async () => {
  runsLoading.value = true;
  try {
    const params = new URLSearchParams({
      limit: String(pageSize.value),
      offset: String((page.value - 1) * pageSize.value)
    });
    if (filters.status) params.set('status', filters.status);
    if (filters.kb) params.set('kb', filters.kb);
    if (filters.slowMs) params.set('slow_ms', filters.slowMs);
    if (filters.feedback) params.set('feedback', filters.feedback);
    runs.value = await apiGet(`/api/ops/chat-runs?${params}`);
  } finally {
    runsLoading.value = false;
  }
});
const loadJobs = () => guarded('索引任务', async () => {
  jobs.value = await apiGet('/api/ops/index-jobs?status=failed&limit=10');
});
const loadLogs = () => guarded('日志', async () => {
  const params = new URLSearchParams({ lines: '200' });
  if (logLevel.value) params.set('level', logLevel.value);
  logs.value = await apiGet(`/api/ops/logs?${params}`);
});

const openTrace = async traceId => {
  traceView.open = true;
  traceView.data = null;
  try {
    traceView.data = await apiGet(`/api/chat/traces/${encodeURIComponent(traceId)}`);
  } catch (error) {
    traceView.open = false;
    toast.error(`追踪记录加载失败：${error.message}`);
  }
};
const closeTrace = () => {
  traceView.open = false;
  traceView.data = null;
};
/** 下载评为没用的问答；gold 预填为回答引用的文件，需人工核对后再放进评测集。 */
const exportDownvoted = async () => {
  try {
    const response = await authorizedFetch('/api/ops/chat-traces/export?rating=down');
    if (!response.ok) throw new Error(response.statusText || `HTTP ${response.status}`);
    const text = await response.text();
    if (!text) {
      toast.success('还没有评为没用的问答');
      return;
    }
    const url = URL.createObjectURL(new Blob([text], { type: 'application/x-ndjson' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'questions.jsonl';
    link.click();
    URL.revokeObjectURL(url);
  } catch (error) {
    toast.error(`导出失败：${error.message}`);
  }
};

const applyFilters = () => {
  page.value = 1;
  loadRuns();
};
const changePage = next => {
  page.value = next;
  loadRuns();
};
const changePageSize = size => {
  pageSize.value = size;
  page.value = 1;
  loadRuns();
};
const refreshAll = () => Promise.all([loadDependencies(), loadOverview(), loadRuns(), loadJobs(), loadLogs()]);

onMounted(refreshAll);
</script>
