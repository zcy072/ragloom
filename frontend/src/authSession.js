// 会话保存在 HttpOnly Cookie 中，脚本既读不到也不保存任何令牌。
// 这里只负责跨标签页的“身份已变化”通知和整页重载。

/** 旧版本把令牌写进 localStorage 用的键；启动时删除，不读取。 */
const LEGACY_AUTH_STORAGE_KEY = 'pymodel.auth';
/** 登录、退出、改密成功后写入的随机标记；不含任何身份信息。 */
export const AUTH_EVENT_STORAGE_KEY = 'ragloom.auth-event';

/** 删除旧版本残留的令牌条目；浏览器禁用存储时忽略。 */
export const removeLegacyAuthSession = () => {
  try {
    localStorage.removeItem(LEGACY_AUTH_STORAGE_KEY);
  } catch {
    // 存储不可用时旧条目也无法被读取，无需处理。
  }
};

/** 生成一次性随机标记，保证每次写入都会触发其他标签页的 storage 事件。 */
const randomMarker = () => {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
};

/** 同源标签页间的身份变化频道；localStorage 不可用（部分隐私模式）时仍能通知。 */
export const AUTH_EVENT_CHANNEL = 'ragloom.auth-event';

/** 打开身份变化频道；浏览器不支持时返回 null。 */
export const openAuthChannel = () => {
  try {
    return typeof BroadcastChannel === 'function' ? new BroadcastChannel(AUTH_EVENT_CHANNEL) : null;
  } catch {
    return null;
  }
};

/**
 * 通知其他标签页身份已变化：localStorage 和 BroadcastChannel 都发，任一可用即可送达。
 * 两条都不可用时降级为重新可见时同步。
 */
export const announceAuthChange = () => {
  try {
    localStorage.setItem(AUTH_EVENT_STORAGE_KEY, randomMarker());
  } catch {
    // 不让通知失败影响登录、退出或改密本身。
  }
  const channel = openAuthChannel();
  if (!channel) return;
  try {
    channel.postMessage('changed');
  } catch {
    // 同上。
  } finally {
    channel.close();
  }
};

/** 通知其他标签页后重载当前页，由启动时的 /me 确认最终身份。 */
export const finishAuthChange = () => {
  announceAuthChange();
  globalThis.location.reload();
};
