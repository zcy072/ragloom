<template>
  <div class="flex-1 p-6 bg-white w-full text-slate-800 overflow-y-auto">
    <!-- Search Bar -->
    <div class="flex items-center space-x-2 mb-6">
      <input 
        type="text" 
        placeholder="搜索用户名或姓名"
        v-model="searchTerm"
        class="border border-gray-200 rounded px-3 py-1.5 text-sm focus:outline-none focus:border-blue-400 w-64"
      />
    </div>

    <!-- Table -->
    <div class="border border-gray-100 rounded-lg overflow-hidden">
      <table class="w-full text-left text-sm text-gray-600">
        <thead class="bg-gray-50 text-gray-500 font-medium text-[13px]">
          <tr>
            <th class="px-6 py-3">ID</th>
            <th class="px-6 py-3">账号名称</th>
            <th class="px-6 py-3">真实姓名</th>
            <th class="px-6 py-3">手机号</th>
            <th class="px-6 py-3">状态</th>
            <th class="px-6 py-3">角色</th>
            <th class="px-6 py-3">注册时间</th>
            <th class="px-6 py-3">操作</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-gray-100">
          <tr v-for="u in users" :key="u.id" class="hover:bg-gray-50/50">
            <td class="px-6 py-4">{{ u.id }}</td>
            <td class="px-6 py-4">{{ u.username }}</td>
            <td class="px-6 py-4">{{ u.name }}</td>
            <td class="px-6 py-4">{{ u.phone || '-' }}</td>
            <td class="px-6 py-4">
              <span :class="['px-2 py-0.5 rounded border text-xs', 
                u.status === '正常' ? 'text-green-500 bg-green-50 border-green-100' : 
                u.status === '审核中' ? 'text-yellow-600 bg-yellow-50 border-yellow-200' : 
                'text-orange-500 bg-orange-50 border-orange-100'
              ]">
                {{ u.status }}
              </span>
            </td>
            <td class="px-6 py-4">
              <span class="text-blue-500 bg-blue-50 px-2 py-0.5 rounded border border-blue-100 text-xs">
                {{ u.role }}
              </span>
            </td>
            <td class="px-6 py-4 text-gray-400 text-[13px]">{{ u.time }}</td>
            <td class="px-6 py-4 text-[13px] relative">
              <button v-if="u.status !== '正常'" @click="statusToggleUser = u.id" class="text-green-500 hover:text-green-700 mr-3">审核通过</button>
              <button v-else @click="statusToggleUser = u.id" class="text-orange-500 hover:text-orange-700 mr-3">停用</button>

              <button @click="openRoleModal(u.id)" class="text-blue-500 hover:text-blue-700 mr-3">分配角色</button>
              <button @click="deleteUserId = u.id" class="text-red-500 hover:text-red-700">删除</button>
            </td>
          </tr>
          <tr v-if="!usersLoading && users.length === 0">
            <td colspan="8" class="px-6 py-12 text-center text-sm text-gray-400">暂无匹配的用户</td>
          </tr>
          <tr v-if="usersLoading">
            <td colspan="8" class="px-6 py-12 text-center text-sm text-gray-400">正在加载用户...</td>
          </tr>
        </tbody>
      </table>
    </div>
    <PaginationControl
      :page="userPage"
      :page-size="userPageSize"
      :total="userTotal"
      :total-pages="userTotalPages"
      :disabled="usersLoading"
      @change-page="changeUserPage"
      @change-page-size="changeUserPageSize"
    />

    <!-- Status Toggle Confirm Modal -->
    <div v-if="statusToggleUser" class="fixed inset-0 bg-black/40 flex items-center justify-center z-50" @click="statusToggleUser = null">
      <div class="bg-white rounded-lg shadow-xl w-80 p-5 overflow-hidden" @click.stop>
        <div class="flex items-start gap-3 mb-4">
          <div class="bg-yellow-50 p-2 rounded-full text-yellow-500 flex-shrink-0">
            <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
          </div>
          <div>
            <h4 class="text-[15px] font-semibold text-gray-800">更改状态</h4>
            <p class="text-sm text-gray-500 mt-1">确定要更改该用户的状态吗？</p>
          </div>
        </div>
        <div class="flex justify-end gap-2">
          <button @click="statusToggleUser = null" class="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-50 hover:bg-gray-100 rounded-md transition-colors">取消</button>
          <button @click="handleStatusToggle(statusToggleUser)" :disabled="submitting" class="px-4 py-2 text-sm font-medium text-white bg-yellow-500 hover:bg-yellow-600 rounded-md transition-colors shadow-sm disabled:opacity-50">确定更改</button>
        </div>
      </div>
    </div>

    <!-- Delete User Confirm Modal -->
    <div v-if="deleteUserId" class="fixed inset-0 bg-black/40 flex items-center justify-center z-50" @click="deleteUserId = null">
      <div class="bg-white rounded-lg shadow-xl w-80 p-5 overflow-hidden" @click.stop>
        <div class="flex items-start gap-3 mb-4">
          <div class="bg-red-50 p-2 rounded-full text-red-500 flex-shrink-0">
            <svg class="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
          </div>
          <div>
            <h4 class="text-[15px] font-semibold text-gray-800">删除用户</h4>
            <p class="text-sm text-gray-500 mt-1">确定要删除该用户吗？此操作不可恢复。</p>
          </div>
        </div>
        <div class="flex justify-end gap-2">
          <button @click="deleteUserId = null" class="px-4 py-2 text-sm font-medium text-gray-600 bg-gray-50 hover:bg-gray-100 rounded-md transition-colors">取消</button>
          <button @click="handleDelete(deleteUserId)" :disabled="submitting" class="px-4 py-2 text-sm font-medium text-white bg-red-500 hover:bg-red-600 rounded-md transition-colors shadow-sm disabled:opacity-50">确定删除</button>
        </div>
      </div>
    </div>

    <!-- Role Assignment Modal -->
    <div v-if="roleModalUser" class="fixed inset-0 bg-black/40 flex items-center justify-center z-50" @click="closeRoleModal">
      <div class="bg-white rounded-lg shadow-xl w-[420px] p-0 overflow-hidden" @click.stop>
        <!-- Header -->
        <div class="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
          <h3 class="text-base font-semibold text-gray-800">分配角色</h3>
          <button @click="closeRoleModal" class="text-gray-400 hover:text-gray-600 text-xl leading-none">&times;</button>
        </div>
        
        <!-- Body -->
        <div class="px-6 py-5">
          <p class="text-sm text-gray-500 mb-4">
            为用户 <span class="font-medium text-gray-700">{{ users.find(u => u.id === roleModalUser)?.name }}</span> 选择角色：
          </p>
          <input
            v-model="roleSearchTerm"
            type="search"
            placeholder="搜索角色名称或描述"
            aria-label="搜索可分配角色"
            class="mb-3 w-full rounded-md border border-gray-200 px-3 py-2 text-sm outline-none focus:border-blue-400"
            @input="scheduleRoleSearch"
          />
          <div class="space-y-2 max-h-[224px] overflow-y-auto">
            <label
              v-for="role in roleOptions" :key="role.id"
              class="flex items-center p-3 rounded-lg border cursor-pointer transition-all"
              :class="selectedRoleId === role.id ? 'border-blue-400 bg-blue-50/60 shadow-sm' : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'"
              @click="selectedRoleId = role.id"
            >
              <div :class="['w-4 h-4 rounded-full border-2 mr-3 flex items-center justify-center flex-shrink-0', selectedRoleId === role.id ? 'border-blue-500' : 'border-gray-300']">
                <div v-if="selectedRoleId === role.id" class="w-2 h-2 rounded-full bg-blue-500"></div>
              </div>
              <div class="flex-1 min-w-0">
                <div class="text-sm font-medium text-gray-800">{{ role.name }}</div>
                <div class="text-xs text-gray-400 truncate">{{ role.description || '暂无描述' }}</div>
              </div>
              <span v-if="role.knowledge_base_permissions?.length > 0" class="text-[11px] text-gray-400 bg-gray-100 px-2 py-0.5 rounded ml-2 whitespace-nowrap">
                {{ role.knowledge_base_permissions.join(', ') }}
              </span>
            </label>
            <div v-if="roleOptionsLoading" class="py-4 text-center text-sm text-gray-400">正在加载角色...</div>
            <div v-else-if="roleOptions.length === 0" class="py-4 text-center text-sm text-gray-400">暂无匹配的角色</div>
          </div>
          <PaginationControl
            :page="rolePage"
            :page-size="rolePageSize"
            :total="roleTotal"
            :total-pages="roleTotalPages"
            :disabled="roleOptionsLoading"
            @change-page="changeRolePage"
            @change-page-size="changeRolePageSize"
          />
        </div>

        <!-- Footer -->
        <div class="px-6 py-4 border-t border-gray-100 flex justify-end space-x-3">
          <button
            @click="closeRoleModal"
            class="px-4 py-2 text-sm text-gray-600 border border-gray-200 rounded-md hover:bg-gray-50 transition-colors"
          >
            取消
          </button>
          <button
            @click="handleAssignRole"
            :disabled="!selectedRoleId || submitting"
            class="px-4 py-2 text-sm text-white bg-[#3b82f6] rounded-md hover:bg-blue-600 transition-colors disabled:opacity-50"
          >
            确认分配
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref, watch } from 'vue';
import PaginationControl from '../components/PaginationControl.vue';
import { usePaginatedList } from '../composables/usePaginatedList';
import { toast } from '../toast';
import { apiDelete, apiPut } from '../apiClient';

const roleModalUser = ref(null);
const selectedRoleId = ref(null);
const deleteUserId = ref(null);
const statusToggleUser = ref(null);
const submitting = ref(false);
const searchTerm = ref('');
const roleSearchTerm = ref('');
const userList = usePaginatedList('/api/users', {
  onError: error => console.error('Failed to fetch users', error)
});
const roleOptionList = usePaginatedList('/api/roles', {
  onError: error => console.error('Failed to fetch role options', error)
});
const {
  items: users,
  page: userPage,
  pageSize: userPageSize,
  total: userTotal,
  totalPages: userTotalPages,
  loading: usersLoading
} = userList;
const {
  items: roleOptions,
  page: rolePage,
  pageSize: rolePageSize,
  total: roleTotal,
  totalPages: roleTotalPages,
  loading: roleOptionsLoading
} = roleOptionList;

const fetchUsers = () => userList.fetchPage(searchTerm.value);
const fetchRoleOptions = () => roleOptionList.fetchPage(roleSearchTerm.value);
const scheduleUserSearch = () => userList.scheduleSearch(searchTerm.value);
const scheduleRoleSearch = () => roleOptionList.scheduleSearch(roleSearchTerm.value);
const changeUserPage = page => userList.changePage(page, searchTerm.value);
const changeUserPageSize = size => userList.changePageSize(size, searchTerm.value);
const changeRolePage = page => roleOptionList.changePage(page, roleSearchTerm.value);
const changeRolePageSize = size => roleOptionList.changePageSize(size, roleSearchTerm.value);

onMounted(fetchUsers);
watch(searchTerm, () => {
  scheduleUserSearch();
});
onBeforeUnmount(() => {
  userList.invalidate();
  roleOptionList.invalidate();
});
/** 删除指定用户并刷新列表。 */
const handleDelete = async (userId) => {
  if (submitting.value) return;
  submitting.value = true;
  try {
    await apiDelete(`/api/users/${userId}`);
    await fetchUsers();
    deleteUserId.value = null;
    toast.success('删除成功');
  } catch (error) {
    toast.error(error.message || '删除失败');
  } finally {
    submitting.value = false;
  }
};
/** 打开角色分配弹窗，并按需加载首个角色选项页面。 */
const openRoleModal = (userId) => {
  roleModalUser.value = userId;
  selectedRoleId.value = null;
  roleSearchTerm.value = '';
  rolePage.value = 1;
  roleOptions.value = [];
  fetchRoleOptions();
};
/** Close the role selector and invalidate any late lookup response. */
const closeRoleModal = () => {
  roleModalUser.value = null;
  selectedRoleId.value = null;
  roleOptionList.invalidate();
};
/** 将弹窗中选定的角色分配给当前用户。 */
const handleAssignRole = async () => {
  if (!selectedRoleId.value || !roleModalUser.value || submitting.value) return;
  submitting.value = true;
  try {
    await apiPut(`/api/users/${roleModalUser.value}/role`, {
      role_id: parseInt(selectedRoleId.value, 10)
    });
    closeRoleModal();
    await fetchUsers();
    toast.success('分配成功');
  } catch (error) {
    toast.error(error.message || '分配角色失败');
  } finally {
    submitting.value = false;
  }
};
/** 在正常和停用之间切换用户状态。 */
const handleStatusToggle = async (id) => {
  const user = users.value.find(u => u.id === id);
  if (!user || submitting.value) return;
  const nextStatus = user.status === '正常' ? '停用' : '正常';
  submitting.value = true;
  try {
    await apiPut(`/api/users/${id}/status`, { status: nextStatus });
    await fetchUsers();
    statusToggleUser.value = null;
    toast.success('状态更新成功');
  } catch (error) {
    toast.error(error.message || '状态更新失败');
  } finally {
    submitting.value = false;
  }
};
</script>
