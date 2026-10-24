import { ref } from 'vue';

import { apiDelete, apiGet, apiPut } from '../apiClient.js';
import { toast } from '../toast.js';
import { store } from '../store.js';
/** 管理聊天会话列表、当前会话消息以及会话删除流程。 */
export function useChatSessions({ messages, selectedKb }, dependencies = {}) {
  const get = dependencies.apiGet || apiGet;
  const remove = dependencies.apiDelete || apiDelete;
  const put = dependencies.apiPut || apiPut;
  const notifications = dependencies.toast || toast;
  const sessionStore = dependencies.store || store;
  const sessions = ref([]);
  const loadingSession = ref(false);
  let loadVersion = 0;
  // 会话列表请求序号：快速切换时，较早发出的响应后到达不能覆盖较新的结果。
  let sessionsRequestVersion = 0;
  const deletingSessionId = ref(null);
  // 只有用户主动点击“新对话”时置为 true，页面恢复会话时保持 false。
  const newSessionPending = ref(false);
  /** 从后端刷新当前用户可见的聊天会话。 */
  const loadSessions = async () => {
    const requestVersion = ++sessionsRequestVersion;
    try {
      const loaded = await get('/api/chat/sessions');
      if (requestVersion === sessionsRequestVersion) sessions.value = loaded;
    } catch (error) {
      console.error('Failed to load chat sessions', error);
      // 保留已有列表；否则用户看到空列表会误以为历史会话丢失。
      if (requestVersion === sessionsRequestVersion) {
        notifications.error(error?.message || '会话列表加载失败，请刷新页面重试');
      }
    }
  };
  /** 加载指定会话的历史消息，并同步该会话使用的知识库。 */
  const loadSessionMessages = async (sessionId) => {
    const version = ++loadVersion;
    loadingSession.value = true;
    sessionStore.activeSessionId = sessionId;
    newSessionPending.value = false;
    try {
      const loaded = await get(`/api/chat/sessions/${encodeURIComponent(sessionId)}`);
      if (version !== loadVersion) return;
      messages.value = loaded;
      const sessionInfo = sessions.value.find(session => session.id === sessionId);
      if (sessionInfo) selectedKb.value = sessionInfo.kb_name;
    } catch (error) {
      if (version === loadVersion) {
        messages.value = [];
        notifications.error(error?.message || '会话加载失败，请稍后重试');
      }
      console.error('Failed to load chat session', error);
    } finally {
      if (version === loadVersion) loadingSession.value = false;
    }
  };
  /** 清空当前消息并切换到尚未持久化的新会话。 */
  const startNewChat = () => {
    loadVersion += 1;
    loadingSession.value = false;
    sessionStore.activeSessionId = null;
    newSessionPending.value = true;
    messages.value = [];
  };
  /** 删除指定会话；若正在查看该会话则同时回到新会话状态。 */
  const deleteSession = async (sessionId) => {
    if (deletingSessionId.value === sessionId) return;
    deletingSessionId.value = sessionId;
    try {
      await remove(`/api/chat/sessions/${encodeURIComponent(sessionId)}`);
      if (sessionStore.activeSessionId === sessionId) startNewChat();
      await loadSessions();
    } catch (error) {
      console.error('Failed to delete session', error);
      notifications.error(error?.message || '删除会话失败，请稍后重试');
    } finally {
      deletingSessionId.value = null;
    }
  };
  /** 修改会话标题；成功后先在本地更新，再用服务端数据校准。失败时保持原标题。 */
  const renameSession = async (sessionId, title) => {
    const cleanTitle = String(title || '').trim();
    const session = sessions.value.find(item => item.id === sessionId);
    if (!session || !cleanTitle || cleanTitle === session.title) return false;
    try {
      await put(`/api/chat/sessions/${encodeURIComponent(sessionId)}/title`, { title: cleanTitle });
    } catch (error) {
      console.error('Failed to rename session', error);
      notifications.error(error?.message || '重命名失败，请稍后重试');
      return false;
    }
    session.title = cleanTitle;
    await loadSessions();
    return true;
  };

  return {
    loadingSession,
    sessions,
    deletingSessionId,
    newSessionPending,
    loadSessions,
    loadSessionMessages,
    startNewChat,
    deleteSession,
    renameSession
  };
}
