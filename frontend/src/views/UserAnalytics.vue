<template>
  <div class="flex-1 overflow-y-auto p-6 bg-[#f6f9f9] w-full text-slate-800">
    <div class="mb-6 flex items-end justify-between">
      <div><p class="text-[11px] font-semibold text-[#649092]">运营</p><h2 class="mt-1 text-lg font-semibold text-[#1e3437]">用户分析</h2></div>
      <div class="flex items-center gap-3">
        <label class="flex items-center gap-1.5 text-xs text-[#819497]">
          <span>统计范围</span>
          <select v-model.number="days" aria-label="统计范围" class="h-8 rounded-md border border-gray-200 bg-white px-2 text-xs text-gray-700" @change="refreshAll">
            <option :value="7">最近 7 天</option>
            <option :value="30">最近 30 天</option>
            <option :value="90">最近 90 天</option>
          </select>
        </label>
        <button type="button" class="flex h-8 items-center gap-1.5 rounded-md border border-gray-200 bg-white px-3 text-xs text-gray-600 hover:bg-gray-50" @click="refreshAll">
          <RefreshCw class="h-3.5 w-3.5" aria-hidden="true" /> 刷新
        </button>
      </div>
    </div>

    <div class="grid grid-cols-2 xl:grid-cols-6 gap-4 mb-4" aria-label="用户概览">
      <div v-for="card in cards" :key="card.title" class="bg-white border border-[#e1eaea] rounded-lg p-5 shadow-[0_1px_2px_rgba(18,53,57,0.04)]">
        <div class="text-xl font-semibold text-[#233b3e] tabular-nums">{{ card.value }}</div>
        <div class="text-xs text-[#819295] mt-1">{{ card.title }}</div>
        <div v-if="card.hint" class="text-[11px] text-gray-400 mt-1">{{ card.hint }}</div>
      </div>
    </div>
    <p v-if="!allKnowledgeBases" class="mb-4 text-xs text-[#819497]">提问和评价只统计你可读的知识库；账户和登录统计不分知识库。</p>

    <div class="grid grid-cols-1 xl:grid-cols-3 gap-4 mb-4">
      <div class="xl:col-span-2 bg-white border border-[#e1eaea] rounded-lg p-5 shadow-[0_1px_2px_rgba(18,53,57,0.04)]">
        <div class="mb-3 flex items-center justify-between">
          <h3 class="font-semibold text-sm">按天趋势</h3>
          <div class="flex items-center gap-4 text-xs text-gray-500" aria-hidden="true">
            <span v-for="line in trendSeries" :key="line.key" class="flex items-center gap-1.5"><i class="h-2 w-2 rounded-full" :style="{ background: line.color }"></i>{{ line.label }}</span>
          </div>
        </div>
        <div class="relative h-48" role="img" :aria-label="`最近 ${overview.days} 天提问数、提问人数和登录人数趋势`">
          <div class="absolute bottom-6 left-0 top-0 flex w-7 flex-col justify-between text-right text-[10px] text-gray-400">
            <span v-for="tick in trendTicks" :key="tick">{{ tick }}</span>
          </div>
          <div class="absolute bottom-6 left-9 right-2 top-0" @mouseleave="trendIndex = null">
            <svg class="h-full w-full overflow-visible" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
              <line v-for="tick in [0, 25, 50, 75]" :key="tick" x1="0" x2="100" :y1="tick" :y2="tick" stroke="#eef2f2" stroke-dasharray="2 2" vector-effect="non-scaling-stroke" />
              <line x1="0" x2="100" y1="100" y2="100" stroke="#e5e7eb" vector-effect="non-scaling-stroke" />
              <path v-for="line in trendLines" :key="line.key" :d="line.path" fill="none" :stroke="line.color" stroke-width="2" vector-effect="non-scaling-stroke" />
              <line v-if="trendPoint" :x1="trendPoint.x" :x2="trendPoint.x" y1="0" y2="100" stroke="#cbd5e1" vector-effect="non-scaling-stroke" />
            </svg>
            <div class="absolute inset-0 flex">
              <div v-for="(point, index) in overview.daily" :key="point.date" class="h-full flex-1" @mouseenter="trendIndex = index"></div>
            </div>
            <div v-if="trendPoint" class="pointer-events-none absolute top-0 z-10 rounded-md border border-[#dfe8e8] bg-white px-3 py-2 text-xs shadow-lg" :class="trendPoint.x > 70 ? '-translate-x-full' : ''" :style="{ left: `${trendPoint.x}%` }">
              <div class="mb-1 font-medium text-[#33484b]">{{ trendPoint.date }}</div>
              <div v-for="line in trendSeries" :key="line.key" class="flex justify-between gap-4 text-gray-500"><span>{{ line.label }}</span><strong :style="{ color: line.color }">{{ trendPoint[line.key] }}</strong></div>
            </div>
          </div>
          <div class="absolute bottom-0 left-9 right-2 h-5 text-[10px] text-gray-400">
            <span v-for="label in trendLabels" :key="label.date" class="absolute -translate-x-1/2 whitespace-nowrap" :style="{ left: `${label.x}%` }">{{ label.date.slice(5) }}</span>
          </div>
        </div>
      </div>

      <div class="bg-white border border-[#e1eaea] rounded-lg p-5 shadow-[0_1px_2px_rgba(18,53,57,0.04)]">
        <h3 class="font-semibold text-sm mb-3">提问时段分布</h3>
        <div class="flex h-40 items-end gap-[2px]" role="img" aria-label="按小时的提问分布">
          <div v-for="(count, hour) in overview.hourly" :key="hour" class="group relative flex-1 h-full flex items-end">
            <div class="w-full rounded-t bg-teal-500/80 group-hover:bg-teal-600" :style="{ height: `${(count / hourlyMax) * 100}%`, minHeight: count ? '2px' : '0' }"></div>
            <span class="pointer-events-none absolute top-0 left-1/2 z-10 hidden -translate-x-1/2 whitespace-nowrap rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-white group-hover:block">{{ hour }} 时 · {{ count }}</span>
          </div>
        </div>
        <div class="mt-1 flex justify-between text-[10px] text-gray-400"><span>0 时</span><span>6</span><span>12</span><span>18</span><span>23 时</span></div>
      </div>
    </div>

    <div class="bg-white border border-[#e1eaea] rounded-lg p-5 shadow-[0_1px_2px_rgba(18,53,57,0.04)] mb-4">
      <h3 class="font-semibold text-sm mb-3">知识库使用情况</h3>
      <p v-if="!overview.knowledge_bases.length" class="text-xs text-gray-400">统计范围内没有提问</p>
      <ul v-else class="space-y-2" aria-label="知识库使用情况">
        <li v-for="kb in overview.knowledge_bases" :key="kb.kb_name" class="flex items-center gap-3 text-xs">
          <span class="w-32 truncate text-gray-700" :title="kb.kb_name">{{ kb.kb_name }}</span>
          <div class="flex-1 h-2 rounded bg-gray-100"><div class="h-2 rounded bg-blue-500" :style="{ width: `${(kb.questions / kbMax) * 100}%` }"></div></div>
          <span class="w-20 text-right tabular-nums text-gray-600">{{ kb.questions }} 次</span>
          <span class="w-16 text-right tabular-nums text-gray-500">{{ kb.users }} 人</span>
          <span class="w-14 text-right tabular-nums" :class="kb.downvotes ? 'text-red-500' : 'text-gray-300'">👎 {{ kb.downvotes }}</span>
        </li>
      </ul>
    </div>

    <div class="bg-white border border-[#e1eaea] rounded-lg shadow-[0_1px_2px_rgba(18,53,57,0.04)] overflow-hidden mb-4">
      <div class="px-5 py-4 border-b border-[#e5eded] flex flex-wrap items-center justify-between gap-3">
        <h3 class="font-semibold text-sm">用户列表</h3>
        <input v-model="query" aria-label="搜索用户" placeholder="搜索用户名或姓名" maxlength="80" class="h-8 w-48 rounded-md border border-gray-200 px-2 text-xs" @input="scheduleUsers" />
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left text-xs text-gray-600">
          <thead class="bg-[#f6f9f9] text-[#6c8083]">
            <tr>
              <th class="px-4 py-3 font-medium">用户</th>
              <th class="px-4 py-3 font-medium">角色</th>
              <th class="px-4 py-3 font-medium">状态</th>
              <th v-for="column in sortColumns" :key="column.key" class="px-4 py-3 font-medium" :class="column.align" :aria-sort="sort === column.key ? (order === 'desc' ? 'descending' : 'ascending') : 'none'">
                <button type="button" class="inline-flex items-center gap-1 hover:text-[#1e3437]" @click="toggleSort(column.key)">
                  {{ column.label }}<span class="text-[10px]">{{ sort === column.key ? (order === 'desc' ? '▼' : '▲') : '' }}</span>
                </button>
              </th>
              <th class="px-4 py-3 font-medium text-right">会话</th>
              <th class="px-4 py-3 font-medium text-right">👍</th>
              <th class="px-4 py-3 font-medium text-right">登录</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-gray-100">
            <tr v-for="user in users.items.value" :key="user.id" class="cursor-pointer hover:bg-[#f6f9f9]" :class="selectedUserId === user.id ? 'bg-teal-50/60' : ''" @click="openUser(user.id)">
              <td class="px-4 py-3"><div class="font-medium text-gray-800">{{ user.name }}</div><div class="text-gray-400">{{ user.username }}</div></td>
              <td class="px-4 py-3">{{ user.role }}</td>
              <td class="px-4 py-3"><span :class="user.status === '正常' ? 'bg-green-50 text-green-600' : 'bg-amber-50 text-amber-600'" class="rounded px-1.5 py-0.5">{{ user.status }}</span></td>
              <td class="px-4 py-3 text-right tabular-nums">{{ user.questions }}<span v-if="user.errors" class="text-red-500">（失败 {{ user.errors }}）</span></td>
              <td class="px-4 py-3 text-right tabular-nums" :class="user.downvotes ? 'text-red-500' : ''">{{ user.downvotes }}</td>
              <td class="px-4 py-3 whitespace-nowrap tabular-nums">{{ formatLocalTime(user.last_question_at) }}</td>
              <td class="px-4 py-3 whitespace-nowrap tabular-nums">{{ formatLocalTime(user.last_login_at) }}</td>
              <td class="px-4 py-3 whitespace-nowrap tabular-nums">{{ formatLocalTime(user.created_at) }}</td>
              <td class="px-4 py-3 text-right tabular-nums">{{ user.sessions }}</td>
              <td class="px-4 py-3 text-right tabular-nums">{{ user.upvotes }}</td>
              <td class="px-4 py-3 text-right tabular-nums">{{ user.logins }}<span v-if="user.login_failures" class="text-amber-600">（失败 {{ user.login_failures }}）</span></td>
            </tr>
            <tr v-if="!users.items.value.length"><td colspan="11" class="px-4 py-8 text-center text-gray-400">{{ users.loading.value ? '加载中…' : '没有符合条件的用户' }}</td></tr>
          </tbody>
        </table>
      </div>
      <div class="px-5 pb-4">
        <PaginationControl
          :page="users.page.value"
          :page-size="users.pageSize.value"
          :total="users.total.value"
          :total-pages="users.totalPages.value"
          :disabled="users.loading.value"
          @change-page="next => users.changePage(next, query)"
          @change-page-size="size => users.changePageSize(size, query)"
        />
      </div>
    </div>

    <section v-if="detail" class="bg-white border border-[#e1eaea] rounded-lg p-5 shadow-[0_1px_2px_rgba(18,53,57,0.04)] mb-4" aria-label="用户详情">
      <div class="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 class="text-sm font-semibold text-[#1e3437]">{{ detail.user.name }} <span class="font-normal text-gray-400">{{ detail.user.username }}</span></h3>
          <p class="mt-1 text-xs text-gray-500">{{ detail.user.role }} · {{ detail.user.status }} · 注册于 {{ formatLocalTime(detail.user.created_at) }}</p>
        </div>
        <button type="button" class="text-gray-400 hover:text-gray-600" aria-label="关闭用户详情" @click="closeUser">✕</button>
      </div>
      <div class="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4 text-xs">
        <div v-for="stat in detailStats" :key="stat.label" class="rounded-md bg-[#f6f9f9] px-3 py-2">
          <div class="text-base font-semibold tabular-nums text-[#233b3e]">{{ stat.value }}</div>
          <div class="text-[#819295]">{{ stat.label }}</div>
        </div>
      </div>

      <div class="grid grid-cols-1 xl:grid-cols-2 gap-4 mb-4">
        <div>
          <h4 class="mb-2 text-xs font-semibold text-gray-600">每天提问数</h4>
          <div class="flex h-24 items-end gap-[2px]" role="img" :aria-label="`${detail.user.username} 最近 ${detail.days} 天每天提问数`">
            <div v-for="point in detail.daily" :key="point.date" class="flex-1 h-full flex items-end" :title="`${point.date}：${point.questions} 次`">
              <div class="w-full rounded-t bg-blue-500/80" :style="{ height: `${(point.questions / detailDailyMax) * 100}%`, minHeight: point.questions ? '2px' : '0' }"></div>
            </div>
          </div>
        </div>
        <div>
          <h4 class="mb-2 text-xs font-semibold text-gray-600">常用知识库</h4>
          <p v-if="!detail.knowledge_bases.length" class="text-xs text-gray-400">统计范围内没有提问</p>
          <ul v-else class="space-y-1.5 text-xs">
            <li v-for="kb in detail.knowledge_bases" :key="kb.kb_name" class="flex items-center gap-2">
              <span class="w-28 truncate">{{ kb.kb_name }}</span>
              <div class="flex-1 h-2 rounded bg-gray-100"><div class="h-2 rounded bg-teal-500" :style="{ width: percent(kb.questions, detail.user.questions) }"></div></div>
              <span class="w-12 text-right tabular-nums">{{ kb.questions }}</span>
            </li>
          </ul>
        </div>
      </div>

      <div class="grid grid-cols-1 xl:grid-cols-3 gap-4">
        <div>
          <h4 class="mb-2 text-xs font-semibold text-gray-600">最近会话</h4>
          <p v-if="!detail.recent_sessions.length" class="text-xs text-gray-400">没有会话</p>
          <ul v-else class="divide-y divide-gray-100 text-xs" aria-label="最近会话">
            <li v-for="session in detail.recent_sessions" :key="session.id" class="flex items-center justify-between gap-2 py-2">
              <div class="min-w-0">
                <div class="truncate text-gray-700" :title="session.title">{{ session.title }}</div>
                <div class="text-gray-400">{{ session.kb_name }} · {{ session.questions }} 问 · {{ formatLocalTime(session.updated_at) }}</div>
              </div>
              <button v-if="canViewConversation" type="button" class="shrink-0 text-teal-700 underline" @click="openConversation(session.id)">查看对话</button>
            </li>
          </ul>
          <p v-if="!canViewConversation && detail.recent_sessions.length" class="mt-2 text-[11px] text-gray-400">查看对话正文需要“查看用户对话”权限。</p>
        </div>
        <div>
          <h4 class="mb-2 text-xs font-semibold text-gray-600">最近提问</h4>
          <p v-if="!detail.recent_questions.length" class="text-xs text-gray-400">没有提问</p>
          <ul v-else class="divide-y divide-gray-100 text-xs" aria-label="最近提问">
            <li v-for="(item, index) in detail.recent_questions" :key="index" class="py-2">
              <div class="truncate text-gray-700" :title="item.question_preview">{{ item.question_preview }}</div>
              <div class="text-gray-400">
                {{ item.kb_name }} · {{ runStatusLabels[item.status] || item.status }} · {{ formatLocalTime(item.created_at) }}
                <span v-if="item.feedback">{{ item.feedback > 0 ? '👍' : '👎' }}</span>
              </div>
            </li>
          </ul>
        </div>
        <div>
          <h4 class="mb-2 text-xs font-semibold text-gray-600">最近登录</h4>
          <p v-if="!detail.recent_logins.length" class="text-xs text-gray-400">没有登录记录</p>
          <ul v-else class="divide-y divide-gray-100 text-xs" aria-label="最近登录">
            <li v-for="(login, index) in detail.recent_logins" :key="index" class="flex justify-between gap-2 py-2">
              <span class="tabular-nums text-gray-600">{{ formatLocalTime(login.created_at) }}</span>
              <span class="truncate text-gray-400">{{ login.client || '-' }}</span>
              <span :class="login.success ? 'text-green-600' : 'text-red-500'">{{ login.success ? '成功' : (loginReasonLabels[login.reason] || '失败') }}</span>
            </li>
          </ul>
        </div>
      </div>
    </section>

    <div class="bg-white border border-[#e1eaea] rounded-lg shadow-[0_1px_2px_rgba(18,53,57,0.04)] overflow-hidden">
      <div class="px-5 py-4 border-b border-[#e5eded]">
        <h3 class="font-semibold text-sm">对话查看审计</h3>
        <p class="mt-1 text-[11px] text-gray-400">每次打开用户的完整对话都会记录在这里，删除账户后记录仍保留。</p>
      </div>
      <table class="w-full text-left text-xs text-gray-600" aria-label="对话查看审计">
        <thead class="bg-[#f6f9f9] text-[#6c8083]">
          <tr>
            <th class="px-4 py-3 font-medium">时间</th>
            <th class="px-4 py-3 font-medium">查看人</th>
            <th class="px-4 py-3 font-medium">被查看用户</th>
            <th class="px-4 py-3 font-medium">会话</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-gray-100">
          <tr v-for="record in audit.items.value" :key="record.id">
            <td class="px-4 py-3 whitespace-nowrap tabular-nums">{{ formatLocalTime(record.created_at) }}</td>
            <td class="px-4 py-3">{{ record.actor_username }}</td>
            <td class="px-4 py-3">{{ record.target_username }}</td>
            <td class="px-4 py-3 max-w-md truncate" :title="record.detail">{{ record.detail }}</td>
          </tr>
          <tr v-if="!audit.items.value.length"><td colspan="4" class="px-4 py-8 text-center text-gray-400">{{ audit.loading.value ? '加载中…' : '还没有查看记录' }}</td></tr>
        </tbody>
      </table>
      <div class="px-5 pb-4">
        <PaginationControl
          :page="audit.page.value"
          :page-size="audit.pageSize.value"
          :total="audit.total.value"
          :total-pages="audit.totalPages.value"
          :disabled="audit.loading.value"
          @change-page="next => audit.changePage(next)"
          @change-page-size="size => audit.changePageSize(size)"
        />
      </div>
    </div>

    <div v-if="conversation.open" class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" @click.self="closeConversation" @keydown.esc="closeConversation">
      <div class="max-h-[85vh] w-full max-w-3xl overflow-y-auto rounded-lg bg-white p-5 shadow-xl" role="dialog" aria-label="用户对话">
        <div class="mb-3 flex items-start justify-between gap-3">
          <div>
            <h3 class="text-sm font-semibold">{{ conversation.data?.title || '用户对话' }}</h3>
            <p v-if="conversation.data" class="mt-1 text-xs text-gray-400">{{ conversation.data.username }} · {{ conversation.data.kb_name }} · 本次查看已记入审计</p>
          </div>
          <button type="button" class="text-gray-400 hover:text-gray-600" aria-label="关闭" @click="closeConversation">✕</button>
        </div>
        <p v-if="!conversation.data" class="text-xs text-gray-400">加载中…</p>
        <ol v-else class="space-y-3">
          <li v-for="(message, index) in conversation.data.messages" :key="index" class="flex" :class="message.role === 'user' ? 'justify-end' : 'justify-start'">
            <div class="max-w-[85%] rounded-lg px-3 py-2 text-sm whitespace-pre-wrap break-words" :class="message.role === 'user' ? 'bg-teal-600 text-white' : 'bg-[#f3f7f7] text-gray-800'">
              {{ message.content }}
              <div v-if="message.source_names.length || message.feedback" class="mt-1.5 text-[11px]" :class="message.role === 'user' ? 'text-teal-100' : 'text-gray-400'">
                <span v-if="message.source_names.length">来源：{{ message.source_names.join('、') }}</span>
                <span v-if="message.feedback" class="ml-1">{{ message.feedback > 0 ? '👍' : '👎' }}</span>
              </div>
            </div>
          </li>
        </ol>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue';
import { RefreshCw } from 'lucide-vue-next';

import { apiGet } from '../apiClient';
import PaginationControl from '../components/PaginationControl.vue';
import { usePaginatedList } from '../composables/usePaginatedList';
import { store } from '../store';
import { toast } from '../toast';
import {
  chartLines, formatLocalTime, loginReasonLabels, niceMax, percent, runStatusLabels, sparseLabelIndexes
} from '../userAnalytics';

const BASE = '/api/user-analytics';
const trendSeries = [
  { key: 'questions', label: '提问数', color: '#3b82f6' },
  { key: 'active_users', label: '提问人数', color: '#10b981' },
  { key: 'login_users', label: '登录人数', color: '#f59e0b' }
];
const sortColumns = [
  { key: 'questions', label: '提问', align: 'text-right' },
  { key: 'downvotes', label: '👎', align: 'text-right' },
  { key: 'last_question_at', label: '最后提问', align: '' },
  { key: 'last_login_at', label: '最后登录', align: '' },
  { key: 'created_at', label: '注册时间', align: '' }
];

const days = ref(30);
const overview = ref({
  days: 30, total_users: 0, pending_users: 0, new_users: 0, active_users: 0, login_users: 0,
  login_failures: 0, questions: 0, upvotes: 0, downvotes: 0, daily: [], hourly: Array(24).fill(0), knowledge_bases: []
});
const query = ref('');
const sort = ref('questions');
const order = ref('desc');
const selectedUserId = ref(null);
const detail = ref(null);
const conversation = reactive({ open: false, data: null });
const trendIndex = ref(null);

const notifyError = label => error => toast.error(`${label}加载失败：${error.message}`);
/** 用户列表的排序和统计范围放进路径，分页、搜索仍交给 usePaginatedList。 */
const users = usePaginatedList(`${BASE}/users`, {
  apiGet: url => {
    const [path, search] = url.split('?');
    const params = new URLSearchParams(search);
    params.set('days', String(days.value));
    params.set('sort', sort.value);
    params.set('order', order.value);
    return apiGet(`${path}?${params}`);
  },
  onError: notifyError('用户列表')
});
const audit = usePaginatedList(`${BASE}/audit-log`, { onError: notifyError('审计记录') });

const allKnowledgeBases = computed(() => (store.user?.knowledgeBasePermissions || []).includes('全公司'));
const canViewConversation = computed(() => (store.user?.menuPermissions || []).includes('查看用户对话'));

const cards = computed(() => {
  const data = overview.value;
  const votes = data.upvotes + data.downvotes;
  return [
    { title: '账户总数', value: data.total_users, hint: data.pending_users ? `${data.pending_users} 个待审核` : '' },
    { title: '新注册', value: data.new_users, hint: `最近 ${data.days} 天` },
    { title: '提问人数', value: data.active_users, hint: `占账户 ${percent(data.active_users, data.total_users)}` },
    { title: '提问次数', value: data.questions, hint: data.active_users ? `人均 ${(data.questions / data.active_users).toFixed(1)} 次` : '' },
    { title: '登录人数', value: data.login_users, hint: data.login_failures ? `失败 ${data.login_failures} 次` : '' },
    { title: '好评率', value: votes ? percent(data.upvotes, votes) : '-', hint: `👍 ${data.upvotes} / 👎 ${data.downvotes}` }
  ];
});

const trendMax = computed(() => niceMax(overview.value.daily.flatMap(point => trendSeries.map(line => point[line.key]))));
const trendTicks = computed(() => [4, 3, 2, 1, 0].map(part => (trendMax.value * part) / 4));
const trendLines = computed(() => chartLines(
  trendSeries.map(line => ({ key: line.key, values: overview.value.daily.map(point => point[line.key]) })),
  trendMax.value
).map((line, index) => ({ ...line, color: trendSeries[index].color })));
const xOf = index => {
  const count = overview.value.daily.length;
  return count === 1 ? 50 : (index / (count - 1)) * 100;
};
const trendLabels = computed(() => sparseLabelIndexes(overview.value.daily.length)
  .map(index => ({ date: overview.value.daily[index].date, x: xOf(index) })));
const trendPoint = computed(() => trendIndex.value === null || !overview.value.daily[trendIndex.value]
  ? null
  : { ...overview.value.daily[trendIndex.value], x: xOf(trendIndex.value) });
const hourlyMax = computed(() => Math.max(1, ...overview.value.hourly));
const kbMax = computed(() => Math.max(1, ...overview.value.knowledge_bases.map(kb => kb.questions)));

const detailDailyMax = computed(() => Math.max(1, ...(detail.value?.daily || []).map(point => point.questions)));
const detailStats = computed(() => {
  const user = detail.value.user;
  return [
    { label: '提问', value: user.questions },
    { label: '会话', value: user.sessions },
    { label: '无依据', value: detail.value.no_evidence },
    { label: '👍 / 👎', value: `${user.upvotes} / ${user.downvotes}` },
    { label: '登录', value: user.logins }
  ];
});

const loadOverview = async () => {
  try {
    overview.value = await apiGet(`${BASE}/overview?days=${days.value}`);
  } catch (error) {
    notifyError('概览')(error);
  }
};
const loadUser = async userId => {
  try {
    detail.value = await apiGet(`${BASE}/users/${userId}?days=${days.value}`);
  } catch (error) {
    notifyError('用户详情')(error);
  }
};
const openUser = userId => {
  selectedUserId.value = userId;
  loadUser(userId);
};
const closeUser = () => {
  selectedUserId.value = null;
  detail.value = null;
};
const toggleSort = key => {
  if (sort.value === key) {
    order.value = order.value === 'desc' ? 'asc' : 'desc';
  } else {
    sort.value = key;
    order.value = 'desc';
  }
  users.page.value = 1;
  users.fetchPage(query.value);
};
const scheduleUsers = () => users.scheduleSearch(query.value);

const openConversation = async sessionId => {
  conversation.open = true;
  conversation.data = null;
  try {
    conversation.data = await apiGet(`${BASE}/users/${selectedUserId.value}/sessions/${encodeURIComponent(sessionId)}`);
    audit.fetchPage();
  } catch (error) {
    conversation.open = false;
    toast.error(`对话加载失败：${error.message}`);
  }
};
const closeConversation = () => {
  conversation.open = false;
  conversation.data = null;
};

const refreshAll = () => {
  const tasks = [loadOverview(), users.fetchPage(query.value), audit.fetchPage()];
  if (selectedUserId.value !== null) tasks.push(loadUser(selectedUserId.value));
  return Promise.all(tasks);
};

onMounted(refreshAll);
</script>
