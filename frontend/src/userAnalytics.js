/** 用户分析页的纯函数：图表坐标、标签和格式化，单独放出来便于单元测试。 */

export const loginReasonLabels = {
  invalid_credentials: '用户名或密码错误',
  account_not_active: '账号未启用',
  rate_limited: '尝试过于频繁'
};

export const runStatusLabels = {
  answered: '已回答',
  no_evidence: '无依据',
  direct: '直接回答',
  interrupted: '中断',
  cancelled: '已取消',
  error: '失败'
};

/** 取一个能被 4 整除的纵轴上限，刻度都是整数。 */
export function niceMax(values) {
  const rawMax = Math.max(0, ...values);
  return Math.max(1, Math.ceil(rawMax / 4)) * 4;
}

/**
 * 把多条序列换算成 0~100 的 SVG 坐标。
 * series: [{ key, values: number[] }]，所有序列长度相同。
 */
export function chartLines(series, yMax) {
  return series.map(item => {
    const count = item.values.length;
    const points = item.values.map((value, index) => ({
      x: count === 1 ? 50 : (index / (count - 1)) * 100,
      y: 100 - (value / yMax) * 100
    }));
    const path = points.map((point, index) => `${index ? 'L' : 'M'} ${point.x} ${point.y}`).join(' ');
    return { key: item.key, points, path };
  });
}

/** 日期较多时只挑大约 6 个均匀分布的横轴标签。 */
export function sparseLabelIndexes(count, wanted = 6) {
  if (count <= wanted) return Array.from({ length: count }, (_, index) => index);
  return [...new Set(Array.from({ length: wanted }, (_, index) => Math.round(index * (count - 1) / (wanted - 1))))];
}

/** UTC ISO 时间显示为浏览器本地时间；空值显示短横线。 */
export function formatLocalTime(value) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const pad = number => String(number).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

/** 0 次显示为 0%，避免除零。 */
export function percent(part, whole) {
  return whole ? `${Math.round((part / whole) * 100)}%` : '0%';
}
