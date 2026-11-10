import { reactive } from 'vue';

// 用户信息只保存在内存中，页面启动时由 /api/me 填充。
export const store = reactive({
  user: null,
  activeMenu: '问答对话',
  targetKb: null,
  activeSessionId: null,
  chatDefaultKb: null
});
