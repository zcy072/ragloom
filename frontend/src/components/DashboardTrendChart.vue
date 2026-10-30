<template>
  <div class="flex h-full min-h-[200px] flex-col" role="group" :aria-label="chartLabel">
    <div v-if="points.length" class="relative min-h-[164px] flex-1 pt-2">
      <div class="absolute bottom-6 left-0 top-2 flex w-7 flex-col justify-between text-right text-[10px] text-gray-400">
        <span v-for="tick in yLabels" :key="tick">{{ tick }}</span>
      </div>

      <div class="absolute bottom-6 left-9 right-2 top-2">
        <svg class="h-full w-full overflow-visible" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          <defs>
            <linearGradient id="question-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stop-color="#3b82f6" stop-opacity="0.2" />
              <stop offset="1" stop-color="#3b82f6" stop-opacity="0" />
            </linearGradient>
            <linearGradient id="answer-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0" stop-color="#10b981" stop-opacity="0.16" />
              <stop offset="1" stop-color="#10b981" stop-opacity="0" />
            </linearGradient>
          </defs>

          <line
            v-for="tick in gridLines"
            :key="tick"
            x1="0"
            x2="100"
            :y1="tick"
            :y2="tick"
            stroke="#eef2f2"
            stroke-dasharray="2 2"
            vector-effect="non-scaling-stroke"
          />
          <line x1="0" x2="100" y1="100" y2="100" stroke="#e5e7eb" vector-effect="non-scaling-stroke" />

          <path :d="questionArea" fill="url(#question-fill)" />
          <path :d="answerArea" fill="url(#answer-fill)" />
          <path :d="questionPath" fill="none" stroke="#3b82f6" stroke-width="2" vector-effect="non-scaling-stroke" />
          <path :d="answerPath" fill="none" stroke="#10b981" stroke-width="2" vector-effect="non-scaling-stroke" />
        </svg>

        <button
          v-for="(point, index) in points"
          :key="point.key"
          type="button"
          class="absolute inset-y-0 cursor-crosshair outline-none"
          :style="hitAreaStyle(index)"
          :aria-label="`${point.name}，提问数 ${point.questions}，回答数 ${point.answers}`"
          @mouseenter="activeIndex = index"
          @focus="activeIndex = index"
          @blur="activeIndex = null"
        ></button>

        <template v-if="activePoint">
          <span class="pointer-events-none absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-blue-500 shadow" :style="markerStyle(activePoint.x, activePoint.questionY)"></span>
          <span class="pointer-events-none absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-emerald-500 shadow" :style="markerStyle(activePoint.x, activePoint.answerY)"></span>
          <div class="pointer-events-none absolute z-10 min-w-28 rounded-md border border-[#dfe8e8] bg-white px-3 py-2 text-xs shadow-lg" :class="activePoint.x > 72 ? '-translate-x-full' : ''" :style="tooltipStyle">
            <div class="mb-1.5 font-medium text-[#33484b]">{{ activePoint.name }}</div>
            <div class="flex items-center justify-between gap-4 text-gray-500"><span>提问数</span><strong class="font-semibold text-blue-600">{{ activePoint.questions }}</strong></div>
            <div class="mt-1 flex items-center justify-between gap-4 text-gray-500"><span>回答数</span><strong class="font-semibold text-emerald-600">{{ activePoint.answers }}</strong></div>
          </div>
        </template>

        <span
          v-if="points.length === 1 && !activePoint"
          class="pointer-events-none absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-blue-500 shadow"
          :style="markerStyle(points[0].x, points[0].questionY)"
        ></span>
        <span
          v-if="points.length === 1 && !activePoint"
          class="pointer-events-none absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-emerald-500 shadow"
          :style="markerStyle(points[0].x, points[0].answerY)"
        ></span>
      </div>

      <div class="absolute bottom-0 left-9 right-2 h-5 text-[10px] text-gray-400">
        <span
          v-for="label in xLabels"
          :key="label.key"
          class="absolute -translate-x-1/2 whitespace-nowrap"
          :style="{ left: `${label.x}%` }"
        >{{ label.name }}</span>
      </div>
    </div>

    <div v-else class="flex min-h-[164px] flex-1 items-center justify-center text-xs text-gray-400">暂无趋势数据</div>

    <div class="mt-2 flex h-5 items-center justify-center gap-5 text-xs text-gray-500" aria-hidden="true">
      <span class="flex items-center gap-1.5"><i class="h-2 w-2 rounded-full bg-blue-500"></i>提问数</span>
      <span class="flex items-center gap-1.5"><i class="h-2 w-2 rounded-full bg-emerald-500"></i>回答数</span>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue';

const props = defineProps({
  data: {
    type: Array,
    default: () => []
  }
});

const activeIndex = ref(null);
const normalizedData = computed(() => props.data.map((item, index) => ({
  key: `${item.name || index}-${index}`,
  name: item.name || '',
  questions: Number(item['提问数']) || 0,
  answers: Number(item['回答数']) || 0
})));
const yMax = computed(() => {
  const rawMax = Math.max(0, ...normalizedData.value.flatMap(item => [item.questions, item.answers]));
  const step = Math.max(1, Math.ceil(rawMax / 4));
  return step * 4;
});
const points = computed(() => normalizedData.value.map((item, index, items) => {
  const x = items.length === 1 ? 50 : (index / (items.length - 1)) * 100;
  return {
    ...item,
    x,
    questionY: 100 - (item.questions / yMax.value) * 100,
    answerY: 100 - (item.answers / yMax.value) * 100
  };
}));

const gridLines = [0, 25, 50, 75];
const yLabels = computed(() => [4, 3, 2, 1, 0].map(part => Math.round((yMax.value * part) / 4)));
/** 根据数据点生成平滑的 SVG 贝塞尔折线路径。 */
const smoothPath = (items, valueKey) => {
  if (!items.length) return '';
  if (items.length === 1) return `M ${items[0].x} ${items[0][valueKey]}`;

  return items.slice(1).reduce((path, point, index) => {
    const previous = items[index];
    const midpoint = (previous.x + point.x) / 2;
    return `${path} C ${midpoint} ${previous[valueKey]}, ${midpoint} ${point[valueKey]}, ${point.x} ${point[valueKey]}`;
  }, `M ${items[0].x} ${items[0][valueKey]}`);
};
/** 在折线路径基础上补齐到底部，形成面积图路径。 */
const areaPath = (items, valueKey) => {
  if (!items.length) return '';
  const line = smoothPath(items, valueKey);
  return `${line} L ${items.at(-1).x} 100 L ${items[0].x} 100 Z`;
};
const questionPath = computed(() => smoothPath(points.value, 'questionY'));
const answerPath = computed(() => smoothPath(points.value, 'answerY'));
const questionArea = computed(() => areaPath(points.value, 'questionY'));
const answerArea = computed(() => areaPath(points.value, 'answerY'));
const xLabels = computed(() => {
  if (points.value.length <= 6) return points.value;
  const indexes = new Set(Array.from({ length: 6 }, (_, index) => Math.round(index * (points.value.length - 1) / 5)));
  return points.value.filter((_, index) => indexes.has(index));
});
const activePoint = computed(() => activeIndex.value === null ? null : points.value[activeIndex.value]);
const tooltipStyle = computed(() => ({
  left: `${activePoint.value?.x || 0}%`,
  top: `${Math.max(0, Math.min(activePoint.value?.questionY || 0, activePoint.value?.answerY || 0) - 8)}%`
}));
const chartLabel = computed(() => normalizedData.value.length
  ? `最近30天问答趋势，共 ${normalizedData.value.length} 个数据点`
  : '最近30天问答趋势，暂无数据');
/** 返回图表数据点的绝对定位样式。 */
const markerStyle = (x, y) => ({ left: `${x}%`, top: `${y}%` });
/** 计算每个数据点用于鼠标交互的横向命中区域。 */
const hitAreaStyle = (index) => {
  const count = points.value.length;
  if (count === 1) return { left: '0', width: '100%' };
  const width = 100 / (count - 1);
  const left = index === 0 ? 0 : points.value[index].x - width / 2;
  const right = index === count - 1 ? 100 : points.value[index].x + width / 2;
  return { left: `${left}%`, width: `${right - left}%` };
};
</script>
