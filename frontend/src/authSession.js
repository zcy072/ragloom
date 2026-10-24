export const AUTH_STORAGE_KEY = 'pymodel.auth';
/** 从浏览器本地存储读取结构完整的登录会话。 */
export const loadAuthSession = () => {
  try {
    const session = JSON.parse(localStorage.getItem(AUTH_STORAGE_KEY) || 'null');
    return session?.token && session?.user ? session : null;
  } catch {
    return null;
  }
};
/** 持久化访问令牌和当前用户信息。 */
export const saveAuthSession = (token, user) => {
  const session = { token, user };
  localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(session));
  return session;
};
/** 清除本地登录会话。 */
export const clearAuthSession = () => localStorage.removeItem(AUTH_STORAGE_KEY);
/** 返回当前会话中的访问令牌，未登录时返回空字符串。 */
export const getAccessToken = () => loadAuthSession()?.token || '';
