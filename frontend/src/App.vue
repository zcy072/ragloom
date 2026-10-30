<template>
  <div id="app-root">
    <div v-if="!store.user">
      <ToastContainer />
      <LoginView @login="onLogin" />
    </div>

    <div v-else class="app-shell flex h-screen text-slate-900 overflow-hidden">
      <ToastContainer />

      <button v-if="sidebarOpen" class="fixed inset-0 z-30 bg-slate-950/30 md:hidden" @click="sidebarOpen = false" aria-label="关闭导航"></button>

      <aside class="app-sidebar flex flex-col flex-shrink-0" :class="{ 'is-open': sidebarOpen }">
        <div class="app-brand flex items-center gap-3 font-semibold text-[15px]">
          <div class="app-brand-mark flex items-center justify-center"><Database class="w-5 h-5" /></div>
          <div>
            <div>知见</div>
            <div class="text-[10px] font-medium text-[#8faeaf] mt-0.5">Knowledge Workspace</div>
          </div>
        </div>

        <div class="flex-1 py-5 overflow-y-auto">
          <div class="app-nav-label px-6 mb-2">工作台</div>
          <nav>
            <button
              v-for="menu in menus"
              :key="menu.id"
              @click="selectMenu(menu.id)"
              class="app-nav-item w-[calc(100%-24px)] flex items-center text-sm text-left"
              :class="{ 'is-active': store.activeMenu === menu.id }"
            >
              <component :is="menu.icon" class="app-nav-icon mr-3 h-4 w-4" />
              {{ menu.id }}
            </button>
          </nav>
        </div>

        <div class="px-5 py-4 border-t border-white/8 text-[11px] leading-relaxed text-[#789496]">
          文档解析、检索与问答
        </div>
      </aside>

      <div class="flex-1 flex flex-col overflow-hidden min-w-0">
        <header class="app-topbar flex items-center justify-between z-10 flex-shrink-0">
          <div class="flex items-center gap-3 min-w-0">
            <button @click="sidebarOpen = true" class="app-icon-button flex items-center justify-center md:hidden" title="打开导航">
              <Menu class="w-4 h-4" />
            </button>
            <div class="min-w-0">
              <h1 class="surface-title font-semibold text-[#17282a] text-[17px] truncate">{{ store.activeMenu }}</h1>
              <p class="hidden sm:block text-[11px] text-[#849698] mt-0.5">企业知识资产工作台</p>
            </div>
          </div>
          <div class="flex items-center text-sm gap-3">
            <span class="status-badge hidden sm:inline-flex text-[#a84444] bg-[#fff2f0] px-2.5 py-1 text-[11px] border border-[#f4d9d5]">
              {{ store.user.role }}
            </span>
            <div class="hidden sm:flex items-center gap-2 border-l border-[#e1e8e8] pl-3">
              <div class="w-7 h-7 rounded-full bg-[#e3f4f2] text-[#246169] flex items-center justify-center text-xs font-semibold">
                {{ (store.user.username || store.user.name || 'A').slice(0, 1).toUpperCase() }}
              </div>
              <span class="text-[#4e6265] text-[13px] max-w-24 truncate">{{ store.user.username || store.user.name || 'admin' }}</span>
            </div>
            <button
              @click="showPwdModal = true"
              class="app-icon-button flex items-center justify-center"
              title="修改密码"
            >
              <Key class="w-4 h-4" />
            </button>
            <button
              @click="logout"
              class="app-icon-button flex items-center justify-center hover:text-red-600 hover:bg-red-50 hover:border-red-200"
              title="退出登录"
            >
              <LogOut class="w-4 h-4" />
            </button>
          </div>
        </header>

        <main class="app-content flex-1 overflow-hidden flex bg-white">
          <Suspense>
            <template #default>
              <div class="contents">
                <component :is="activePage.component" v-bind="activePage.props" />
              </div>
            </template>
            <template #fallback>
              <div class="flex flex-1 items-center justify-center text-[#6f8587]" aria-live="polite">
                <LoaderCircle class="h-5 w-5 animate-spin" />
                <span class="sr-only">页面加载中</span>
              </div>
            </template>
          </Suspense>
        </main>
      </div>

      <!-- Password Change Modal -->
      <div v-if="showPwdModal" class="fixed inset-0 bg-black/40 flex items-center justify-center z-50" @keydown.esc="closePwdModal">
        <div class="bg-white rounded-lg shadow-xl w-96 p-6" role="dialog" aria-modal="true" aria-labelledby="pwd-dialog-title">
          <h3 id="pwd-dialog-title" class="text-lg font-semibold text-gray-800 mb-4 flex items-center">
            <Key class="w-5 h-5 mr-2 text-blue-500" /> 修改密码
          </h3>
          <form @submit.prevent="handlePwdChange" class="space-y-4">
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">原密码</label>
              <div class="relative">
                <input
                  :type="showOldPwd ? 'text' : 'password'" required v-focus v-model="oldPwd"
                  class="w-full bg-gray-50 border border-gray-200 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-md pl-3.5 pr-10 py-2.5 text-sm outline-none"
                  placeholder="请输入当前密码"
                />
                <button type="button" @click="showOldPwd = !showOldPwd" class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600">
                  <component :is="showOldPwd ? EyeOff : Eye" class="w-4 h-4" />
                </button>
              </div>
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">新密码</label>
              <div class="relative">
                <input
                  :type="showNewPwd ? 'text' : 'password'" required v-model="newPwd"
                  class="w-full bg-gray-50 border border-gray-200 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-md pl-3.5 pr-10 py-2.5 text-sm outline-none"
                  placeholder="请输入新密码"
                />
                <button type="button" @click="showNewPwd = !showNewPwd" class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600">
                  <component :is="showNewPwd ? EyeOff : Eye" class="w-4 h-4" />
                </button>
              </div>
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">确认新密码</label>
              <div class="relative">
                <input
                  :type="showConfirmPwd ? 'text' : 'password'" required v-model="confirmPwd"
                  class="w-full bg-gray-50 border border-gray-200 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-md pl-3.5 pr-10 py-2.5 text-sm outline-none"
                  placeholder="请再次输入新密码"
                />
                <button type="button" @click="showConfirmPwd = !showConfirmPwd" class="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600">
                  <component :is="showConfirmPwd ? EyeOff : Eye" class="w-4 h-4" />
                </button>
              </div>
            </div>
            <div class="pt-2 flex justify-end gap-3">
              <button type="button" @click="closePwdModal" class="px-4 py-2 text-sm text-gray-600 bg-gray-50 hover:bg-gray-100 rounded-md transition-colors">取消</button>
              <button type="submit" :disabled="changingPwd" class="px-4 py-2 text-sm text-white bg-blue-500 hover:bg-blue-600 rounded-md transition-colors shadow-sm disabled:opacity-50">{{ changingPwd ? '提交中...' : '确认修改' }}</button>
            </div>
          </form>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, defineAsyncComponent, onMounted, onUnmounted } from 'vue';
import { store } from './store';
import LoginView from './views/LoginView.vue';
import ToastContainer from './components/ToastContainer.vue';
import { toast } from './toast';
import { apiGet, apiPost } from './apiClient';
import { AUTH_STORAGE_KEY, clearAuthSession, loadAuthSession, saveAuthSession } from './authSession';
import { canManageKnowledgeBases, canOpenKnowledgeBase } from './knowledgeAccess';
import { LayoutDashboard, FolderOpen, MessageCircle, Settings, Users, UserCog, LogOut, Key, Eye, EyeOff, Database, Menu, LoaderCircle, Activity, BarChart3 } from 'lucide-vue-next';

const ChatWorkspace = defineAsyncComponent(() => import('./views/ChatWorkspace.vue'));
const DashboardView = defineAsyncComponent(() => import('./views/DashboardView.vue'));
const KnowledgeBaseManagement = defineAsyncComponent(() => import('./views/KnowledgeBaseManagement.vue'));
const ModelConfiguration = defineAsyncComponent(() => import('./views/ModelConfiguration.vue'));
const UserManagement = defineAsyncComponent(() => import('./views/UserManagement.vue'));
const RoleManagement = defineAsyncComponent(() => import('./views/RoleManagement.vue'));
const OpsMonitor = defineAsyncComponent(() => import('./views/OpsMonitor.vue'));
const UserAnalytics = defineAsyncComponent(() => import('./views/UserAnalytics.vue'));

// 与后端 services/core/config.py 的 MIN_PASSWORD_LENGTH 保持一致。
const MIN_PASSWORD_LENGTH = 8;
const showPwdModal = ref(false);
const oldPwd = ref('');
const newPwd = ref('');
const confirmPwd = ref('');

const showOldPwd = ref(false);
const showNewPwd = ref(false);
const showConfirmPwd = ref(false);
const sidebarOpen = ref(false);
const changingPwd = ref(false);
const vFocus = { mounted: element => element.focus() };

const allMenus = [
  { id: '仪表盘统计', icon: LayoutDashboard, permission: '仪表盘统计', component: DashboardView },
  { id: '知识库', icon: FolderOpen, permission: '知识库', component: KnowledgeBaseManagement },
  { id: '问答对话', icon: MessageCircle, permission: '问答对话', component: ChatWorkspace },
  { id: '模型管理', icon: Settings, permission: '模型配置', component: ModelConfiguration },
  { id: '用户管理', icon: Users, permission: '用户管理', component: UserManagement },
  { id: '角色管理', icon: UserCog, permission: '角色管理', component: RoleManagement },
  { id: '运行监控', icon: Activity, permission: '运行监控', component: OpsMonitor },
  { id: '用户分析', icon: BarChart3, permission: '用户分析', component: UserAnalytics },
];
const menus = computed(() => {
  if (!store.user) return [];
  return allMenus.filter((menu) =>
    menu.id === '知识库'
      ? canOpenKnowledgeBase(store.user)
      : store.user.menuPermissions?.includes(menu.permission)
  );
});
const knowledgeBaseReadonly = computed(() => !canManageKnowledgeBases(store.user));
const activePage = computed(() => {
  // 只渲染当前用户有权限的菜单；上一个用户留下的菜单不可见时回到第一个可见菜单。
  const menu = menus.value.find(item => item.id === store.activeMenu)
    || menus.value[0]
    || allMenus[0];
  return {
    component: menu.component,
    props: menu.id === '知识库' ? { readonly: knowledgeBaseReadonly.value } : {}
  };
});
/** 登录成功后把用户信息写入全局状态。 */
const onLogin = (userData) => {
  store.user = userData;
};
/** 清除登录信息和当前聊天会话，返回登录页。 */
const logout = () => {
  clearAuthSession();
  store.user = null;
  store.activeSessionId = null;
  store.activeMenu = '问答对话';
  store.targetKb = null;
  store.chatDefaultKb = null;
};

/** 其他标签页登出、换账号或令牌被清除时，本标签页同步回到登录页。 */
const syncAuthAcrossTabs = (event) => {
  if (event.key !== AUTH_STORAGE_KEY && event.key !== null) return;
  const session = loadAuthSession();
  if (!session) {
    if (store.user) logout();
  } else if (store.user && session.user?.username !== store.user.username) {
    // 另一个标签页换了账号登录：重新载入，避免继续用旧账号的界面状态操作新令牌。
    window.location.reload();
  }
};

/** 重新读取实时权限；管理员改了角色后，已登录用户无需等令牌过期就能看到新菜单。 */
const refreshPermissions = async () => {
  if (!store.user) return;
  try {
    const fresh = await apiGet('/api/me');
    if (!store.user || fresh.username !== store.user.username) return;
    store.user = { ...store.user, ...fresh };
    const session = loadAuthSession();
    if (session) saveAuthSession(session.token, store.user);
  } catch (_) {
    // 401 已由 apiClient 统一处理；网络波动时保留当前菜单，下次获得焦点再试。
  }
};
const refreshOnFocus = () => {
  if (document.visibilityState === 'visible') refreshPermissions();
};

onMounted(() => {
  window.addEventListener('auth-expired', logout);
  window.addEventListener('storage', syncAuthAcrossTabs);
  document.addEventListener('visibilitychange', refreshOnFocus);
  refreshPermissions();
});
onUnmounted(() => {
  window.removeEventListener('auth-expired', logout);
  window.removeEventListener('storage', syncAuthAcrossTabs);
  document.removeEventListener('visibilitychange', refreshOnFocus);
});
/** 切换主菜单，并在移动端同时收起侧边栏。 */
const selectMenu = (menu) => {
  store.activeMenu = menu;
  sidebarOpen.value = false;
};
/** 关闭改密弹窗并清空已输入的密码。 */
const closePwdModal = () => {
  if (changingPwd.value) return;
  showPwdModal.value = false;
  oldPwd.value = '';
  newPwd.value = '';
  confirmPwd.value = '';
};
/** 校验两次新密码后提交改密，成功时强制重新登录。 */
const handlePwdChange = async () => {
  if (changingPwd.value) return;
  if (newPwd.value !== confirmPwd.value) {
    toast.error('两次输入的新密码不一致');
    return;
  }
  if (newPwd.value.length < MIN_PASSWORD_LENGTH) {
    toast.error(`新密码至少需要 ${MIN_PASSWORD_LENGTH} 个字符`);
    return;
  }
  changingPwd.value = true;
  try {
    const data = await apiPost('/api/change_password', {
        old_password: oldPwd.value,
        new_password: newPwd.value
    });
    toast.success(data.message || '修改成功');
    showPwdModal.value = false;
    oldPwd.value = '';
    newPwd.value = '';
    confirmPwd.value = '';
    logout();
  } catch (err) {
    toast.error(err.message || '网络错误，修改失败');
  } finally {
    changingPwd.value = false;
  }
};
</script>
