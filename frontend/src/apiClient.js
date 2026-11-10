export class ApiError extends Error {
  /** 保存后端返回的状态码和响应体，便于页面按错误类型给出提示。 */
  constructor(message, status, data) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}
/** 后端要求所有写请求携带的固定标记头；跨站表单无法设置它。 */
export const WRITE_REQUEST_HEADER = 'X-RAGLoom-Request';
const SAFE_METHODS = new Set(['GET', 'HEAD', 'OPTIONS']);
// 身份失效后只重载一次；晚到的旧请求 401 不会形成重载风暴。
let authReloadStarted = false;
/** 因身份失效或切换而整页重载，同一页面生命周期内至多一次。 */
export const reloadForAuthChange = () => {
  if (authReloadStarted) return;
  authReloadStarted = true;
  globalThis.location.reload();
};
/** 只允许当前站点的 /api 路径，避免把带 Cookie 的请求逻辑用于任意地址。 */
const sameOriginApiUrl = (url) => {
  const origin = globalThis.location?.origin;
  let parsed;
  try {
    parsed = new URL(String(url), origin);
  } catch {
    parsed = null;
  }
  if (!parsed || parsed.origin !== origin || !(parsed.pathname === '/api' || parsed.pathname.startsWith('/api/'))) {
    throw new TypeError(`只允许请求本站 /api 接口：${url}`);
  }
  return parsed;
};
/**
 * 发起携带会话 Cookie 的请求：写请求附加固定标记头，受保护请求返回 401 时整页重载一次。
 * authReload: false 用于登录和启动身份检查，它们的 401 是正常的“未登录”结果。
 */
export const authorizedFetch = async (url, options = {}) => {
  const { authReload = true, ...fetchOptions } = options;
  const parsed = sameOriginApiUrl(url);
  const method = String(fetchOptions.method || 'GET').toUpperCase();
  const headers = new Headers(fetchOptions.headers || {});
  if (!SAFE_METHODS.has(method)) headers.set(WRITE_REQUEST_HEADER, '1');
  const response = await fetch(url, { ...fetchOptions, headers, credentials: 'same-origin' });
  // 不重放原请求、不清 Cookie：重载后由 /me 决定去向。
  if (response.status === 401 && authReload && parsed.pathname !== '/api/login') {
    reloadForAuthChange();
  }
  return response;
};
/** 常见请求字段的中文名；未列出的字段只显示错误本身。 */
const FIELD_LABELS = {
  username: '账号名称',
  name: '名称',
  phone: '手机号码',
  password: '密码',
  old_password: '原密码',
  new_password: '新密码',
  description: '描述',
  question: '问题',
  title: '标题',
  tags: '标签',
  api_key: 'API Key',
  base_url: 'Base URL'
};
/** 把一条 pydantic 校验错误转成中文；不认识的类型保留原文。 */
const validationItemText = (item) => {
  const field = Array.isArray(item?.loc) ? item.loc[item.loc.length - 1] : '';
  const label = FIELD_LABELS[field] || '';
  const ctx = item?.ctx || {};
  let reason;
  switch (item?.type) {
    case 'missing':
      reason = '不能为空';
      break;
    case 'string_too_short':
      reason = ctx.min_length === 1 ? '不能为空' : `至少需要 ${ctx.min_length} 个字符`;
      break;
    case 'string_too_long':
      reason = `不能超过 ${ctx.max_length} 个字符`;
      break;
    case 'too_long':
      reason = `不能超过 ${ctx.max_length} 项`;
      break;
    default:
      return item?.msg || '';
  }
  return label ? `${label}${reason}` : `输入${reason}`;
};
/** FastAPI 校验失败时 detail 是 {msg} 数组；转成可直接展示的文本。 */
const errorDetailText = (detail) => {
  if (Array.isArray(detail)) {
    return detail.map(validationItemText).filter(Boolean).join('；');
  }
  return typeof detail === 'string' ? detail : '';
};
/** 统一处理 JSON 请求、响应解析以及非成功状态的异常转换。 */
export const apiRequest = async (url, options = {}) => {
  const { json, headers: initialHeaders, ...fetchOptions } = options;
  const headers = new Headers(initialHeaders || {});

  if (json !== undefined) {
    headers.set('Content-Type', 'application/json');
    fetchOptions.body = JSON.stringify(json);
  }

  const response = await authorizedFetch(url, { ...fetchOptions, headers });
  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!response.ok) {
    const message = errorDetailText(data?.detail) || data?.message || response.statusText || '请求失败';
    throw new ApiError(message, response.status, data);
  }
  return data;
};
/** 使用 GET 方法请求接口。 */
export const apiGet = (url, options) => apiRequest(url, options);
/** 使用 POST 方法提交 JSON 数据。 */
export const apiPost = (url, json, options = {}) => apiRequest(url, { ...options, method: 'POST', json });
/** 使用 PUT 方法更新 JSON 数据。 */
export const apiPut = (url, json, options = {}) => apiRequest(url, { ...options, method: 'PUT', json });
/** 使用 DELETE 方法删除资源。 */
export const apiDelete = (url, options = {}) => apiRequest(url, { ...options, method: 'DELETE' });
