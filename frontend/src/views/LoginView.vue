<template>
  <div class="login-screen min-h-screen flex items-center justify-center p-4 sm:p-8">
    <div class="w-full max-w-5xl min-h-[600px] grid grid-cols-1 lg:grid-cols-[1fr_430px] overflow-hidden bg-white border border-[#dce8e8] rounded-lg shadow-[0_24px_70px_rgba(18,54,57,0.16)]">
      <section class="login-aside hidden lg:flex flex-col justify-between p-12 text-white">
        <div class="flex items-center gap-3">
          <div class="login-logo w-10 h-10 flex items-center justify-center"><BookOpen class="w-5 h-5" /></div>
          <span class="font-semibold text-lg">RAGLoom</span>
        </div>
        <div>
          <p class="text-[#94d6d1] text-xs font-semibold tracking-[0.08em]">KNOWLEDGE WORKSPACE</p>
          <h1 class="mt-4 text-[32px] font-semibold leading-tight">让每一份知识<br />都能被准确找到</h1>
        </div>
        <p class="text-sm text-[#91aeb0]">Enterprise knowledge workspace</p>
      </section>

      <div class="login-card w-full max-w-none p-7 sm:p-10 lg:p-12 relative overflow-y-auto bg-white">
        <div class="flex justify-center mb-6 lg:hidden">
          <div class="login-logo h-12 w-12 flex items-center justify-center"><BookOpen class="w-6 h-6" /></div>
        </div>

        <div class="text-center lg:text-left mb-8">
          <p class="text-[11px] font-semibold tracking-[0.08em] text-[#438b8b] mb-2">WELCOME</p>
          <h2 class="text-2xl font-semibold text-[#1b2d30]">
          {{ view === 'login' ? '欢迎访问 RAG 智能问答系统' : '申请账号权限' }}
        </h2>
          <p class="text-sm text-[#718385] mt-2">
            {{ view === 'login' ? '使用你的账号继续访问工作台' : '填写信息后，由管理员完成账号审核' }}
          </p>
        </div>

      <template v-if="view === 'login'">
        <form @submit.prevent="handleLogin" class="space-y-4">
          <div v-if="error" class="text-sm text-red-500 text-center bg-red-50 py-2 rounded">
            {{ error }}
          </div>

          <div class="space-y-4">
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">用户名</label>
              <input type="text" required maxlength="80" v-model="username" class="w-full bg-gray-50 border border-gray-200 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-md px-3.5 py-2.5 text-sm text-gray-800 transition-all outline-none" placeholder="输入账号名称" />
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">密码</label>
              <PasswordInput v-model="password" required placeholder="输入密码" />
            </div>
          </div>

          <div class="pt-4 flex gap-4">
            <button
              type="submit"
              :disabled="loading"
              class="flex-1 bg-[#176d70] hover:bg-[#10585c] text-white font-medium py-2.5 rounded-md transition-colors shadow-sm disabled:opacity-70 text-sm"
            >
              {{ loading ? '正在登录...' : '登录系统' }}
            </button>
            <button
              type="button"
              @click="setView('register')"
              class="flex-1 bg-white hover:bg-[#f4f8f8] text-[#4d6669] border border-[#d8e4e4] font-medium py-2.5 rounded-md transition-colors text-sm"
            >
              申请权限
            </button>
          </div>
        </form>
      </template>

      <template v-else>
        <form @submit.prevent="handleRegister" class="space-y-4">
          <div v-if="error" class="text-sm text-red-500 text-center bg-red-50 py-2 rounded">
            {{ error }}
          </div>

          <div class="space-y-4">
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">账号名称</label>
              <input
                type="text"
                required
                v-model="regUsername"
                maxlength="80"
                class="w-full bg-gray-50 border border-gray-200 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-md px-3.5 py-2.5 text-sm text-gray-800 transition-all outline-none"
                placeholder="作为登录账号"
              />
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">真实姓名</label>
              <input
                type="text"
                required
                v-model="regName"
                maxlength="80"
                class="w-full bg-gray-50 border border-gray-200 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-md px-3.5 py-2.5 text-sm text-gray-800 transition-all outline-none"
                placeholder="如：雷勇"
              />
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">手机号码（选填）</label>
              <input
                type="text"
                v-model="regPhone"
                maxlength="32"
                class="w-full bg-gray-50 border border-gray-200 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-md px-3.5 py-2.5 text-sm text-gray-800 transition-all outline-none"
                placeholder="联系方式"
              />
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">设置密码</label>
              <PasswordInput v-model="regPassword" required placeholder="至少 8 个字符，最多 72 字节" />
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">确认密码</label>
              <PasswordInput v-model="regConfirmPassword" required placeholder="再次输入密码" />
            </div>
          </div>

          <div class="pt-4 flex gap-4">
            <button
              type="submit"
              :disabled="loading"
              class="flex-1 bg-[#176d70] hover:bg-[#10585c] text-white font-medium py-2.5 rounded-md transition-colors shadow-sm disabled:opacity-70 text-sm"
            >
              {{ loading ? '提交中...' : '提交申请' }}
            </button>
            <button
              type="button"
              @click="setView('login')"
              class="flex-1 bg-white hover:bg-[#f4f8f8] text-[#4d6669] border border-[#d8e4e4] font-medium py-2.5 rounded-md transition-colors text-sm"
            >
              返回登录
            </button>
          </div>
        </form>
      </template>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { BookOpen } from 'lucide-vue-next';
import { toast } from '../toast';
import { apiPost } from '../apiClient';
import { finishAuthChange } from '../authSession';
import { existingPasswordProblem, newPasswordProblem } from '../passwordPolicy';
import PasswordInput from '../components/PasswordInput.vue';

const username = ref('');
const password = ref('');

const error = ref('');
const loading = ref(false);

const view = ref('login'); // 'login' or 'register'

const regUsername = ref('');
const regName = ref('');
const regPhone = ref('');
const regPassword = ref('');
const regConfirmPassword = ref('');

/** 切换登录和注册视图，并清空旧错误。 */
const setView = (newView) => {
  view.value = newView;
  error.value = '';
};
/** 提交账号凭证；后端写入会话 Cookie 后重载页面，由启动时的 /me 确认身份。 */
const handleLogin = async () => {
  if (loading.value) return;
  error.value = '';
  const problem = existingPasswordProblem(password.value);
  if (problem) {
    error.value = problem;
    return;
  }
  loading.value = true;

  try {
    await apiPost('/api/login', {
      username: username.value,
      password: password.value
    }, { authReload: false });
    // 保持“正在登录”状态直到页面重载。
    finishAuthChange();
  } catch (err) {
    error.value = err.message || '网络连接失败';
    loading.value = false;
  }
};
/** 校验注册表单后提交普通用户注册申请。 */
const handleRegister = async () => {
  if (loading.value) return;
  error.value = '';
  if (!regUsername.value.trim() || !regName.value.trim()) {
    error.value = '账号名称和真实姓名不能为空';
    return;
  }
  if (regPassword.value !== regConfirmPassword.value) {
    error.value = '两次输入的密码不一致';
    return;
  }
  const problem = newPasswordProblem(regPassword.value);
  if (problem) {
    error.value = problem;
    return;
  }

  loading.value = true;

  try {
    const data = await apiPost('/api/register', {
        username: regUsername.value,
        name: regName.value,
        phone: regPhone.value,
        password: regPassword.value
    });
    toast.success(data.message || '注册申请已提交');
    setView('login');
  } catch (err) {
    error.value = err.message || '注册失败';
  } finally {
    loading.value = false;
  }
};
</script>
