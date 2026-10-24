<template>
  <div class="flex-1 overflow-y-auto p-6 bg-[#f6f9f9] w-full text-slate-800">
    <div class="mb-6 flex items-end justify-between">
      <div><p class="text-[11px] font-semibold text-[#649092]">概览</p><h2 class="mt-1 text-lg font-semibold text-[#1e3437]">知识库运行概况</h2></div>
      <span class="text-xs text-[#819497]">实时数据</span>
    </div>
    <div class="grid grid-cols-2 xl:grid-cols-4 gap-4 mb-4">
      <div v-for="(c, i) in topCards" :key="i" class="bg-white border border-[#e1eaea] rounded-lg p-5 flex items-center gap-4 shadow-[0_1px_2px_rgba(18,53,57,0.04)]">
        <div :class="`w-11 h-11 ${c.bg} rounded-lg flex items-center justify-center`">
          <component :is="c.icon" :class="c.iconColor" class="w-5 h-5" />
        </div>
        <div><div class="text-xl font-semibold text-[#233b3e]">{{ c.val }}</div><div class="text-xs text-[#819295] mt-1">{{ c.title }}</div></div>
      </div>
    </div>

    <!-- Middle Row -->
    <div class="grid grid-cols-1 xl:grid-cols-2 gap-4 mb-4">
      <!-- Document Parsing Status -->
      <div class="bg-white border border-[#e1eaea] rounded-lg p-5 shadow-[0_1px_2px_rgba(18,53,57,0.04)]">
        <h3 class="font-semibold text-sm mb-4">文档解析情况</h3>
        <div class="flex items-center space-x-8">
          <div class="relative w-32 h-32 flex-shrink-0">
            <div
              class="absolute inset-0 rounded-full"
              :style="pieStyle"
              role="img"
              :aria-label="pieLabel"
            ></div>
            <div class="absolute inset-[13px] rounded-full bg-white"></div>
            <div class="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
              <span class="text-xl font-bold text-green-500">{{ successRate }}%</span>
              <span class="text-[10px] text-gray-400">成功率</span>
            </div>
          </div>
          <div class="flex-1 space-y-3">
            <div v-for="(d, i) in pieData" :key="i" class="flex items-center text-sm">
              <div class="w-2 h-2 rounded-full mr-2" :style="{ backgroundColor: d.color }"></div>
              <span class="text-gray-500 w-16">{{ d.name }}</span>
              <span class="font-medium text-gray-800">{{ d.value }}</span>
            </div>
            <div class="mt-4 pt-2 w-full flex h-1.5 rounded-full overflow-hidden bg-gray-100">
              <div class="bg-green-500 h-full" :style="{ width: `${successRate}%` }"></div>
              <div class="bg-red-500 h-full" :style="{ width: `${failRate}%` }"></div>
            </div>
            <div class="text-right text-xs text-gray-400 mt-1">{{ failRate }}%</div>
          </div>
        </div>
      </div>

      <!-- Daily Q&A Trend -->
      <div class="bg-white border border-[#e1eaea] rounded-lg p-5 shadow-[0_1px_2px_rgba(18,53,57,0.04)] flex flex-col">
        <h3 class="font-semibold text-sm mb-2">每日问答趋势 (最近30天)</h3>
        <div class="flex-1 w-full min-h-[200px]">
          <DashboardTrendChart :data="stats.line_data || []" />
        </div>
      </div>
    </div>

    <!-- Bottom Table -->
    <div class="bg-white border border-[#e1eaea] rounded-lg shadow-[0_1px_2px_rgba(18,53,57,0.04)] overflow-hidden">
      <div class="px-5 py-4 border-b border-[#e5eded]">
        <h3 class="font-semibold text-sm">各知识库统计</h3>
      </div>
      <table class="w-full text-left text-sm text-gray-600">
        <thead class="bg-[#f6f9f9] text-[#6c8083] text-xs font-medium">
          <tr>
            <th class="px-5 py-3 font-medium">知识库</th>
            <th class="px-5 py-3 font-medium text-center">文档数</th>
            <th class="px-5 py-3 font-medium text-center">总大小</th>
            <th class="px-5 py-3 font-medium text-center">成功</th>
            <th class="px-5 py-3 font-medium text-center">失败</th>
            <th class="px-5 py-3 font-medium text-center">待处理</th>
            <th class="px-5 py-3 font-medium w-48">各类型占比</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-gray-100">
          <tr 
            v-for="(kb, i) in stats.kb_list" 
            :key="i" 
            class="hover:bg-gray-50/50 cursor-pointer transition-colors"
            @click="goToKb(kb)"
            title="点击进入知识库管理"
          >
            <td class="px-5 py-4 flex items-center text-blue-600 font-medium hover:text-blue-800">
              <FolderOpen class="w-4 h-4 mr-2" /> {{ kb.name }}
            </td>
            <td class="px-5 py-4 text-center">{{ kb.docs }}</td>
            <td class="px-5 py-4 text-center">{{ kb.size }}</td>
            <td class="px-5 py-4 text-center text-green-500">{{ kb.success }}</td>
            <td class="px-5 py-4 text-center text-red-500">{{ kb.fail }}</td>
            <td class="px-5 py-4 text-center text-gray-400">{{ kb.processing || 0 }}</td>
            <td class="px-5 py-4">
              <div class="space-y-1.5 text-[10px] text-gray-500">
                <template v-if="kb.types && kb.types.length > 0">
                  <div v-for="(type, tIdx) in kb.types" :key="tIdx" class="flex items-center justify-between">
                    <span class="w-8 truncate" :title="type.ext">{{ type.ext }}</span>
                    <span class="w-8 text-right">{{ type.count }}份</span>
                    <div :class="`flex-1 mx-2 h-1.5 ${getBgColor(tIdx)} rounded`">
                      <div :class="`${getColor(tIdx)} h-1.5 rounded`" :style="{ width: `${Math.round((type.count / kb.docs) * 100)}%` }"></div>
                    </div>
                  </div>
                </template>
                <span v-else class="text-gray-400">暂无文档</span>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue';
import { store } from '../store';
import { apiGet } from '../apiClient';
import { FolderOpen, FileText, HardDrive, FileTerminal } from 'lucide-vue-next';
import DashboardTrendChart from '../components/DashboardTrendChart.vue';

const stats = ref({
  kbs: 0, docs: 0, size: "0 MB", tasks: 0, kb_list: [],
  pie_data: [{ name: '成功', value: 0 }, { name: '失败', value: 0 }, { name: '处理中', value: 0 }],
  line_data: []
});

onMounted(async () => {
  try {
    stats.value = await apiGet('/api/dashboard/stats');
  } catch (err) {
    console.error("Failed to fetch dashboard stats", err);
  }
});
const topCards = computed(() => [
  { title: '知识库总数', val: stats.value.kbs.toString(), icon: FolderOpen, iconColor: 'text-blue-500', bg: 'bg-blue-50' },
  { title: '文档总数', val: stats.value.docs.toString(), icon: FileText, iconColor: 'text-green-500', bg: 'bg-green-50' },
  { title: '总大小', val: stats.value.size, icon: HardDrive, iconColor: 'text-orange-500', bg: 'bg-orange-50' },
  { title: '解析处理中', val: stats.value.tasks.toString(), icon: FileTerminal, iconColor: 'text-pink-500', bg: 'bg-pink-50' },
]);

const pieColors = ['#10b981', '#ef4444', '#6b7280'];
const pieData = computed(() => (stats.value.pie_data || []).map((d, i) => ({ ...d, color: pieColors[i] })));
const totalDocs = computed(() => pieData.value.reduce((s, d) => s + d.value, 0));
const successRate = computed(() => totalDocs.value > 0 ? Math.round((pieData.value[0]?.value || 0) / totalDocs.value * 100) : 0);
// 失败率只统计“失败”项；“处理中”的文档不算失败，也不并入红色进度条。
const failRate = computed(() => totalDocs.value > 0 ? Math.round((pieData.value[1]?.value || 0) / totalDocs.value * 100) : 0);
const pieStyle = computed(() => {
  if (!totalDocs.value) return { background: '#e5e7eb' };

  let start = 0;
  const stops = pieData.value.map((item) => {
    const end = start + (item.value / totalDocs.value) * 100;
    const stop = `${item.color} ${start}% ${end}%`;
    start = end;
    return stop;
  });
  return { background: `conic-gradient(${stops.join(', ')})` };
});
const pieLabel = computed(() => `文档解析情况：${pieData.value.map(item => `${item.name} ${item.value}`).join('，')}`);
/** 按排行位置循环返回进度条背景色。 */
const getBgColor = (idx) => ['bg-orange-200', 'bg-blue-200', 'bg-green-200'][idx % 3];
/** 按排行位置循环返回进度条主色。 */
const getColor = (idx) => ['bg-orange-500', 'bg-blue-500', 'bg-green-500'][idx % 3];
/** 从仪表盘跳转到指定知识库的管理视图。 */
const goToKb = (kb) => {
  store.targetKb = kb;
  store.activeMenu = '知识库';
};
</script>
