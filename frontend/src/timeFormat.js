/** 时间格式化纯函数，供多个页面共用。 */

/** UTC ISO 时间显示为浏览器本地时间；空值显示短横线。 */
export function formatLocalTime(value) {
  if (!value) return '-';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const pad = number => String(number).padStart(2, '0');
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}`;
}
