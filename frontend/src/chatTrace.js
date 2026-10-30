// 与后端 services/chat/service.py 中 trace.start(...) 的阶段名保持一致。
const STAGE_LABELS = {
  image: '图片识别',
  plan: '问题改写 / 检索规划',
  variant: '书面化补充查询',
  retrieval: '召回',
  rerank: '重排',
  agentic: '补充检索',
  web_search: '联网搜索',
  context: '上下文组装',
  generate: '生成回答'
};
const STATUS_LABELS = {
  ok: '完成',
  skipped: '跳过',
  degraded: '降级',
  error: '失败',
  cancelled: '已取消'
};
const DETAIL_LABELS = {
  ocr_chars: '识别字数',
  should_search: '需要检索',
  rewritten: '已改写',
  query: '查询',
  candidates: '候选数',
  results: '结果数',
  fallback: '原问题回退',
  kept: '保留',
  below_threshold: '低于下限',
  top_score: '最高分',
  rounds: '轮数',
  evidence: '证据数',
  spans: '段数',
  chars: '字数',
  first_token_ms: '首字耗时 ms',
  answer_chars: '回答字数',
  cited: '引用数',
  reason: '原因',
  error: '错误'
};

export const stageLabel = name => STAGE_LABELS[name] || name;
export const stageStatusLabel = status => STATUS_LABELS[status] || status;

/** 把阶段详情转为“名称：值”的短文本，空值不显示。 */
export const stageDetailText = detail => Object.entries(detail || {})
  .filter(([, value]) => value !== null && value !== undefined && value !== '')
  .map(([key, value]) => {
    let text = value;
    if (typeof value === 'boolean') text = value ? '是' : '否';
    else if (typeof value === 'number' && !Number.isInteger(value)) text = value.toFixed(3);
    return `${DETAIL_LABELS[key] || key}：${text}`;
  })
  .join('；');

export const formatElapsed = ms => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);
