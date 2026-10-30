<template>
  <div class="flex-1 p-6 bg-[#f8f9fa] w-full text-slate-800 relative">
    <div class="mb-6 flex flex-wrap items-center justify-between gap-3">
      <button 
        @click="showCreate = true"
        class="bg-[#3b82f6] hover:bg-blue-600 text-white px-4 py-2 rounded shadow-sm text-sm font-medium transition-colors">
        新建角色
      </button>
      <input
        v-model="searchTerm"
        type="search"
        placeholder="搜索角色名称或描述"
        aria-label="搜索角色名称或描述"
        class="w-full rounded-md border border-gray-200 px-3 py-2 text-sm outline-none focus:border-blue-400 sm:w-64"
      />
    </div>

    <div v-if="showCreate" class="mb-6 bg-white p-4 rounded-lg border border-gray-200 flex items-center space-x-4">
      <input v-model="newRoleName" placeholder="角色名称" class="border px-3 py-1.5 rounded text-sm outline-none" />
      <input v-model="newRoleDesc" placeholder="描述" class="border px-3 py-1.5 rounded text-sm outline-none w-64" />
      <button @click="handleCreateRole" :disabled="submitting" class="bg-green-500 text-white px-4 py-1.5 rounded text-sm disabled:opacity-50">保存</button>
      <button @click="showCreate = false" class="text-gray-500 text-sm">取消</button>
    </div>

    <!-- Table -->
    <div class="bg-white border border-gray-100 rounded-lg overflow-hidden">
      <table class="w-full text-left text-sm text-gray-600">
        <thead class="bg-gray-50 text-gray-500 font-medium text-[13px]">
          <tr>
            <th class="px-6 py-3">ID</th>
            <th class="px-6 py-3">角色名称</th>
            <th class="px-6 py-3">描述</th>
            <th class="px-6 py-3">知识库权限</th>
            <th class="px-6 py-3 text-right pr-12">操作</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-gray-100">
          <tr v-for="r in roles" :key="r.id" class="hover:bg-gray-50/50">
            <td class="px-6 py-4">{{ r.id }}</td>
            <td class="px-6 py-4 text-blue-600 font-medium">
              {{ r.name }} 
              <span v-if="r.name === '超级管理员'" class="text-red-400 bg-red-50 px-1.5 py-0.5 rounded text-[10px] ml-1">系统保护</span>
            </td>
            <td class="px-6 py-4">{{ r.description || '-' }}</td>
            <td class="px-6 py-4">
              <span class="text-gray-500 text-xs">{{ (r.knowledge_base_permissions && r.knowledge_base_permissions.length > 0) ? r.knowledge_base_permissions.join(', ') : '无' }}</span>
            </td>
            <td class="px-6 py-4 text-[13px] text-right pr-6 space-x-3">
              <button v-if="r.name !== '超级管理员'" class="text-green-500 hover:underline" @click="openEditModal(r)">配置权限</button>
              <button v-if="r.name !== '超级管理员'" class="text-red-500 hover:underline" @click="openDeletePopover(r, $event)">删除</button>
            </td>
          </tr>
          <tr v-if="!loading && roles.length === 0">
            <td colspan="5" class="px-6 py-12 text-center text-sm text-gray-400">暂无匹配的角色</td>
          </tr>
          <tr v-if="loading">
            <td colspan="5" class="px-6 py-12 text-center text-sm text-gray-400">正在加载角色...</td>
          </tr>
        </tbody>
      </table>
    </div>
    <PaginationControl
      :page="currentPage"
      :page-size="pageSize"
      :total="totalRoles"
      :total-pages="totalPages"
      :disabled="loading"
      @change-page="changePage"
      @change-page-size="changePageSize"
    />

    <Teleport to="body">
      <div v-if="deleteRole" class="fixed inset-0 z-[60]" @click="closeDeletePopover">
        <div
          class="absolute flex w-56 flex-col gap-3 rounded-lg border border-gray-100 bg-white p-3 text-left shadow-[0_10px_30px_rgba(0,0,0,0.1)]"
          :style="deletePopoverStyle"
          role="dialog"
          aria-modal="true"
          aria-labelledby="delete-role-title"
          @click.stop
        >
          <div class="flex items-start gap-2.5">
            <div class="mt-0.5 flex flex-shrink-0 rounded-full bg-red-50 p-1.5 text-red-500">
              <AlertTriangle class="h-4 w-4" aria-hidden="true" />
            </div>
            <div>
              <h3 id="delete-role-title" class="text-[13px] font-semibold leading-tight text-gray-800">删除角色</h3>
              <p class="mt-1 text-[11px] leading-relaxed text-gray-500">确定要删除“{{ deleteRole.name }}”吗？</p>
            </div>
          </div>
          <div class="mt-1 flex justify-end gap-2">
            <button
              type="button"
              :disabled="deletingRoleId !== null"
              class="rounded-md bg-gray-50 px-3 py-1.5 text-xs font-medium text-gray-600 transition-colors hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50"
              @click="closeDeletePopover"
            >
              取消
            </button>
            <button
              type="button"
              :disabled="deletingRoleId !== null"
              class="rounded-md bg-red-500 px-3 py-1.5 text-xs font-medium text-white shadow-sm transition-colors hover:bg-red-600 disabled:cursor-not-allowed disabled:opacity-50"
              @click="handleDeleteRole(deleteRole.id)"
            >
              {{ deletingRoleId === deleteRole.id ? '删除中...' : '确定删除' }}
            </button>
          </div>
        </div>
      </div>
    </Teleport>

    <!-- Modal Overlay -->
    <div v-if="showModal" class="absolute inset-0 bg-gray-500/50 flex items-center justify-center z-50 rounded-md">
      <div class="bg-white rounded-lg shadow-xl w-[500px] overflow-hidden flex flex-col">
        <div class="px-6 py-4 border-b border-gray-100 flex justify-between items-center">
          <h3 class="font-semibold text-gray-800">配置菜单权限</h3>
          <button @click="showModal = false" class="text-gray-400 hover:text-gray-600">&times;</button>
        </div>
        <div class="p-6">
          <p class="text-sm text-gray-600 mb-4">当前配置角色: <span class="font-medium text-blue-600">{{ editingRole?.name }}</span></p>
          
          <div class="bg-gray-50 p-4 rounded border border-gray-100 mb-4">
            <p class="text-sm font-medium text-gray-700 mb-3">左侧菜单权限</p>
            <div class="grid grid-cols-2 gap-3 text-sm text-gray-600">
              <label v-for="menu in menuList" :key="menu.id" class="flex items-center space-x-2">
                <input 
                  type="checkbox" 
                  :checked="editingRole?.menu_permissions?.includes(menu.id) || false"
                  @change="toggleMenuPerm(menu.id)"
                  class="rounded text-blue-500" 
                /> 
                <span>{{ menu.label }}</span>
              </label>
            </div>
          </div>

          <div class="bg-gray-50 p-4 rounded border border-gray-100">
            <p class="text-sm font-medium text-gray-700 mb-3">允许访问的知识库</p>
            <div class="grid grid-cols-2 gap-3 text-sm text-gray-600">
              <label class="flex items-center space-x-2">
                <input 
                  type="checkbox" 
                  :checked="editingRole?.knowledge_base_permissions?.includes('全公司') || false"
                  @change="toggleKnowledgeBasePermission('全公司')"
                  class="rounded text-blue-500" 
                /> 
                <span class="font-medium text-red-500">全公司 (所有)</span>
              </label>
              <label v-for="kbName in kbs" :key="kbName" class="flex items-center space-x-2">
                <input 
                  type="checkbox" 
                  :checked="editingRole?.knowledge_base_permissions?.includes(kbName) || false"
                  @change="toggleKnowledgeBasePermission(kbName)"
                  class="rounded text-blue-500" 
                /> 
                <span>{{ kbName }}</span>
              </label>
            </div>
          </div>
        </div>
        <div class="px-6 py-4 border-t border-gray-100 bg-gray-50 flex justify-end space-x-2">
          <button @click="showModal = false" class="px-4 py-2 bg-white border border-gray-200 text-gray-600 rounded text-sm hover:bg-gray-50">取消</button>
          <button @click="handleSaveRole" :disabled="submitting" class="px-4 py-2 bg-blue-500 text-white rounded text-sm hover:bg-blue-600 disabled:opacity-50">保存权限</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue';
import { AlertTriangle } from 'lucide-vue-next';
import PaginationControl from '../components/PaginationControl.vue';
import { usePaginatedList } from '../composables/usePaginatedList';
import { toast } from '../toast';
import { apiDelete, apiGet, apiPost, apiPut } from '../apiClient';

const showModal = ref(false);
const editingRole = ref(null);
const newRoleName = ref('');
const newRoleDesc = ref('');
const showCreate = ref(false);
const kbs = ref([]);
const searchTerm = ref('');
const deleteRoleId = ref(null);
const deletingRoleId = ref(null);
const submitting = ref(false);
const deletePopoverStyle = ref({});

const roleList = usePaginatedList('/api/roles', {
  onError: error => console.error('Failed to fetch roles', error)
});
const {
  items: roles,
  page: currentPage,
  pageSize,
  total: totalRoles,
  totalPages,
  loading
} = roleList;
const deleteRole = computed(() => roles.value.find(role => role.id === deleteRoleId.value) || null);

const DELETE_POPOVER_WIDTH = 224;
const DELETE_POPOVER_HEIGHT = 132;
const VIEWPORT_MARGIN = 12;
const menuList = [
  { id: '仪表盘统计', label: '仪表盘统计' },
  { id: '知识库', label: '知识库浏览' },
  { id: '知识库管理', label: '知识库管理' },
  { id: '问答对话', label: '问答对话' },
  { id: '模型配置', label: '模型配置' },
  { id: '用户管理', label: '用户管理' },
  { id: '角色管理', label: '角色管理' },
  { id: '运行监控', label: '运行监控（可查看所有用户的问题前 200 字）' },
  { id: '用户分析', label: '用户分析（各用户的使用统计和问题预览）' },
  { id: '查看用户对话', label: '查看用户对话（可打开用户的完整对话，每次查看都记入审计）' },
  { id: '联网搜索', label: '联网搜索（提问时可搜索网页，按次计费）' },
];
const fetchRoles = () => roleList.fetchPage(searchTerm.value);
const scheduleSearch = () => roleList.scheduleSearch(searchTerm.value);
const changePage = page => roleList.changePage(page, searchTerm.value);
const changePageSize = size => roleList.changePageSize(size, searchTerm.value);
/** 读取全部知识库名称，供角色授权时选择。 */
const fetchKbs = async () => {
  try {
    const data = await apiGet('/api/kbs');
    kbs.value = data.map(k => k.name);
  } catch (e) {
    console.error("Failed to fetch kbs", e);
  }
};

onMounted(() => {
  fetchRoles();
  fetchKbs();
});
watch(searchTerm, () => {
  scheduleSearch();
});
onBeforeUnmount(() => roleList.invalidate());
/** 创建新角色并刷新角色列表。 */
const handleCreateRole = async () => {
  if (!newRoleName.value || submitting.value) return;
  submitting.value = true;
  try {
    await apiPost('/api/roles', { name: newRoleName.value, description: newRoleDesc.value });
    showCreate.value = false;
    newRoleName.value = '';
    newRoleDesc.value = '';
    currentPage.value = 1;
    await fetchRoles();
    toast.success('创建成功');
  } catch (error) {
    toast.error(error.message || '创建失败');
  } finally {
    submitting.value = false;
  }
};
/** 将轻量确认框定位在操作按钮附近，并在底部空间不足时向上展开。 */
const openDeletePopover = (role, event) => {
  const rect = event.currentTarget.getBoundingClientRect();
  const left = Math.min(
    Math.max(VIEWPORT_MARGIN, rect.right - DELETE_POPOVER_WIDTH),
    window.innerWidth - DELETE_POPOVER_WIDTH - VIEWPORT_MARGIN
  );
  const top = window.innerHeight - rect.bottom >= DELETE_POPOVER_HEIGHT
    ? rect.bottom + 8
    : Math.max(VIEWPORT_MARGIN, rect.top - DELETE_POPOVER_HEIGHT - 8);

  deletePopoverStyle.value = { left: `${left}px`, top: `${top}px` };
  deleteRoleId.value = role.id;
};
/** 关闭删除确认框；提交删除时保留确认框直到请求结束。 */
const closeDeletePopover = () => {
  if (deletingRoleId.value === null) deleteRoleId.value = null;
};
/** 删除指定角色并清理确认状态。 */
const handleDeleteRole = async (id) => {
  if (deletingRoleId.value !== null) return;
  deletingRoleId.value = id;
  try {
    await apiDelete(`/api/roles/${id}`);
    await fetchRoles();
    deleteRoleId.value = null;
    toast.success('删除成功');
  } catch (error) {
    toast.error(error.message || '删除失败');
  } finally {
    deletingRoleId.value = null;
  }
};
/** 深拷贝角色数据后打开权限编辑弹窗，避免未保存操作污染列表。 */
const openEditModal = (r) => {
  editingRole.value = JSON.parse(JSON.stringify(r)); // deep copy
  showModal.value = true;
};
/** 保存角色的菜单权限和知识库权限。 */
const handleSaveRole = async () => {
  if (!editingRole.value || submitting.value) return;
  submitting.value = true;
  try {
    await apiPut(`/api/roles/${editingRole.value.id}`, {
        menu_permissions: editingRole.value.menu_permissions || [],
        knowledge_base_permissions: editingRole.value.knowledge_base_permissions || []
    });
    showModal.value = false;
    await fetchRoles();
    toast.success('保存成功');
  } catch (error) {
    toast.error(error.message || '保存失败');
  } finally {
    submitting.value = false;
  }
};
/** 切换单项菜单权限。 */
const toggleMenuPerm = (menu) => {
  const perms = editingRole.value.menu_permissions || [];
  if (perms.includes(menu)) {
    editingRole.value.menu_permissions = perms.filter(m => m !== menu);
  } else {
    editingRole.value.menu_permissions = [...perms, menu];
  }
};
/** 切换知识库权限，并保证“全公司”和单库授权互斥。 */
const toggleKnowledgeBasePermission = (kbName) => {
  const perms = editingRole.value.knowledge_base_permissions || [];
  if (kbName === '全公司') {
    editingRole.value.knowledge_base_permissions = perms.includes(kbName) ? [] : [kbName];
    return;
  }

  const scopedPerms = perms.filter(permission => permission !== '全公司');
  if (scopedPerms.includes(kbName)) {
    editingRole.value.knowledge_base_permissions = scopedPerms.filter(permission => permission !== kbName);
  } else {
    editingRole.value.knowledge_base_permissions = [...scopedPerms, kbName];
  }
};
</script>
