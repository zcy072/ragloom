import { reactive } from 'vue';
import { loadAuthSession } from './authSession.js';

const authSession = loadAuthSession();

export const store = reactive({
  user: authSession?.user || null,
  activeMenu: '问答对话',
  targetKb: null,
  activeSessionId: null,
  chatDefaultKb: null
});
