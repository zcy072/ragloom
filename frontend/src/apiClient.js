import { clearAuthSession, getAccessToken } from './authSession.js';
export class ApiError extends Error {
  /** 保存后端返回的状态码和响应体，便于页面按错误类型给出提示。 */
  constructor(message, status, data) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}
/** 发起带登录令牌的请求，并在令牌失效时统一清理前端登录状态。 */
export const authorizedFetch = async (url, options = {}) => {
  const headers = new Headers(options.headers || {});
  const token = getAccessToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const response = await fetch(url, { ...options, headers });
  // 只处理“发出请求时用的令牌”失效。若存储里的令牌已被其他标签页替换或清除，
  // 这次旧请求的 401 不能清掉新会话；界面同步由 storage 事件负责。
  if (response.status === 401 && token && getAccessToken() === token) {
    clearAuthSession();
    window.dispatchEvent(new CustomEvent('auth-expired'));
  }
  return response;
};
/** FastAPI 校验失败时 detail 是 {msg} 数组；转成可直接展示的文本。 */
const errorDetailText = (detail) => {
  if (Array.isArray(detail)) {
    return detail.map(item => item?.msg).filter(Boolean).join('；');
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
