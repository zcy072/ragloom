<template>
  <div id="app-root">
    <div v-if="authState === 'loading'" class="min-h-screen flex items-center justify-center text-[#6f8587]" aria-live="polite">
      <LoaderCircle class="h-5 w-5 animate-spin" />
      <span class="ml-2 text-sm">正在确认登录状态...</span>
    </div>

    <div v-else-if="authState === 'error'" class="min-h-screen flex flex-col items-center justify-center gap-4 text-[#4e6265]" role="alert">
      <p class="text-sm">{{ authError }}</p>
      <button
        type="button"
        @click="loadIdentity"
        class="px-4 py-2 text-sm text-white bg-[#176d70] hover:bg-[#10585c] rounded-md transition-colors shadow-sm"
      >
        重试
      </button>
    </div>

    <div v-else-if="!store.user">
      <ToastContainer />
      <LoginView />
    </div>

    <div v-else class="app-shell flex h-screen text-slate-900 overflow-hidden">
      <ToastContainer />

      <button v-if="sidebarOpen" class="fixed inset-0 z-30 bg-slate-950/30 md:hidden" @click="sidebarOpen = false" aria-label="关闭导航"></button>

      <aside class="app-sidebar flex flex-col flex-shrink-0" :class="{ 'is-open': sidebarOpen }">
        <div class="app-brand flex items-center gap-3 font-semibold text-[15px]">
          <div class="app-brand-mark flex items-center justify-center"><Database class="w-5 h-5" /></div>
          <div>
            <div>RAGLoom</div>
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
                {{ (store.user.username || 'A').slice(0, 1).toUpperCase() }}
              </div>
              <span class="text-[#4e6265] text-[13px] max-w-24 truncate">{{ store.user.username }}</span>
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
              :disabled="loggingOut"
              class="app-icon-button flex items-center justify-center hover:text-red-600 hover:bg-red-50 hover:border-red-200"
              title="退出登录"
            >
              <LogOut class="w-4 h-4" />
            </button>
          </div>
        </header>

        <main class="app-content flex-1 overflow-hidden flex bg-white">
          <Suspense :key="activePage.id">
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
              <PasswordInput v-model="oldPwd" autofocus required placeholder="请输入当前密码" />
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">新密码</label>
              <PasswordInput v-model="newPwd" required placeholder="至少 8 个字符，最多 72 字节" />
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">确认新密码</label>
              <PasswordInput v-model="confirmPwd" required placeholder="请再次输入新密码" />
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
import PasswordInput from './components/PasswordInput.vue';
import { toast } from './toast';
import { ApiError, apiGet, apiPost, reloadForAuthChange } from './apiClient';
import { AUTH_EVENT_STORAGE_KEY, finishAuthChange, openAuthChannel } from './authSession';
import { existingPasswordProblem, newPasswordProblem } from './passwordPolicy';
import { canManageKnowledgeBases, canOpenKnowledgeBase } from './knowledgeAccess';
import { LayoutDashboard, FolderOpen, MessageCircle, Settings, Users, UserCog, LogOut, Key, Database, Menu, LoaderCircle, Activity, BarChart3 } from 'lucide-vue-next';

const ChatWorkspace = defineAsyncComponent(() => import('./views/ChatWorkspace.vue'));
const DashboardView = defineAsyncComponent(() => import('./views/DashboardView.vue'));
const KnowledgeBaseManagement = defineAsyncComponent(() => import('./views/KnowledgeBaseManagement.vue'));
const ModelConfiguration = defineAsyncComponent(() => import('./views/ModelConfiguration.vue'));
const UserManagement = defineAsyncComponent(() => import('./views/UserManagement.vue'));
const RoleManagement = defineAsyncComponent(() => import('./views/RoleManagement.vue'));
const OpsMonitor = defineAsyncComponent(() => import('./views/OpsMonitor.vue'));
const UserAnalytics = defineAsyncComponent(() => import('./views/UserAnalytics.vue'));

// loading：启动时等待 /me；ready：已确定登录或未登录；error：网络或服务异常，可重试。
const authState = ref('loading');
const authError = ref('');
const loggingOut = ref(false);
const showPwdModal = ref(false);
const oldPwd = ref('');
const newPwd = ref('');
const confirmPwd = ref('');

const sidebarOpen = ref(false);
const changingPwd = ref(false);

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
    id: menu.id,
    component: menu.component,
    props: menu.id === '知识库' ? { readonly: knowledgeBaseReadonly.value } : {}
  };
});
/** 启动时向 /me 确认身份；只有 401 视为未登录，网络或服务错误显示可重试页面。 */
const loadIdentity = async () => {
  authState.value = 'loading';
  try {
    store.user = await apiGet('/api/me', { authReload: false });
    authState.value = 'ready';
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) {
      store.user = null;
      authState.value = 'ready';
      return;
    }
    authError.value = err instanceof ApiError
      ? `服务暂时不可用（${err.status}），请稍后重试`
      : '无法连接服务器，请检查网络后重试';
    authState.value = 'error';
  }
};
/** 服务端退出成功后通知其他标签页并重载；失败时保留当前页面。 */
const logout = async () => {
  if (loggingOut.value) return;
  loggingOut.value = true;
  try {
    await apiPost('/api/logout');
    finishAuthChange();
  } catch (_) {
    toast.error('退出未完成，请重试');
    loggingOut.value = false;
  }
};

/** 其他标签页登录、退出或改密后会写入随机标记，本标签页随之重载。 */
const syncAuthAcrossTabs = (event) => {
  if (event.key === AUTH_EVENT_STORAGE_KEY) reloadForAuthChange();
};

// 同一时间只保留一个 /me 刷新请求。
let refreshingIdentity = false;
/** 重新读取实时权限；同一账号更新菜单，账号已变化则重载，401 由请求封装统一重载。 */
const refreshPermissions = async () => {
  if (!store.user || refreshingIdentity) return;
  refreshingIdentity = true;
  try {
    const fresh = await apiGet('/api/me');
    if (!store.user) return;
    if (fresh.id !== store.user.id) {
      reloadForAuthChange();
      return;
    }
    store.user = { ...store.user, ...fresh };
  } catch (_) {
    // 403、网络波动或 5xx 时保留当前菜单，下次获得焦点再试。
  } finally {
    refreshingIdentity = false;
  }
};
const refreshOnFocus = () => {
  if (document.visibilityState === 'visible') refreshPermissions();
};

let authChannel = null;
onMounted(() => {
  window.addEventListener('storage', syncAuthAcrossTabs);
  authChannel = openAuthChannel();
  if (authChannel) authChannel.onmessage = reloadForAuthChange;
  document.addEventListener('visibilitychange', refreshOnFocus);
  loadIdentity();
});
onUnmounted(() => {
  window.removeEventListener('storage', syncAuthAcrossTabs);
  authChannel?.close();
  authChannel = null;
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
/** 校验两次新密码后提交改密，成功后重载页面并重新登录。 */
const handlePwdChange = async () => {
  if (changingPwd.value) return;
  if (newPwd.value !== confirmPwd.value) {
    toast.error('两次输入的新密码不一致');
    return;
  }
  const problem = existingPasswordProblem(oldPwd.value, '原密码') || newPasswordProblem(newPwd.value, '新密码');
  if (problem) {
    toast.error(problem);
    return;
  }
  changingPwd.value = true;
  try {
    const data = await apiPost('/api/change_password', {
        old_password: oldPwd.value,
        new_password: newPwd.value
    });
    // 服务端已撤销全部会话并清除 Cookie；保持提交中状态直到页面重载。
    toast.success(data?.message || '修改成功，请重新登录');
    finishAuthChange();
  } catch (err) {
    toast.error(err.message || '网络错误，修改失败');
    changingPwd.value = false;
  }
};
</script>
