<template>
  <div class="login-screen min-h-screen flex items-center justify-center p-4 sm:p-8">
    <div class="w-full max-w-5xl min-h-[600px] grid grid-cols-1 lg:grid-cols-[1fr_430px] overflow-hidden bg-white border border-[#dce8e8] rounded-lg shadow-[0_24px_70px_rgba(18,54,57,0.16)]">
      <section class="login-aside hidden lg:flex flex-col justify-between p-12 text-white">
        <div class="flex items-center gap-3">
          <div class="login-logo w-10 h-10 flex items-center justify-center"><BookOpen class="w-5 h-5" /></div>
          <span class="font-semibold text-lg">知见</span>
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
              <input type="text" required v-model="username" class="w-full bg-gray-50 border border-gray-200 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-md px-3.5 py-2.5 text-sm text-gray-800 transition-all outline-none" placeholder="输入账号名称" />
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">密码</label>
              <div class="relative">
                <input :type="showLoginPwd ? 'text' : 'password'" required v-model="password" class="w-full bg-gray-50 border border-gray-200 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-md px-3.5 py-2.5 pr-10 text-sm text-gray-800 transition-all outline-none" placeholder="输入密码" />
                <button type="button" @click="showLoginPwd = !showLoginPwd" class="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600">
                  <Eye v-if="!showLoginPwd" class="w-4 h-4" />
                  <EyeOff v-else class="w-4 h-4" />
                </button>
              </div>
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
                class="w-full bg-gray-50 border border-gray-200 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-md px-3.5 py-2.5 text-sm text-gray-800 transition-all outline-none"
                placeholder="如：雷勇"
              />
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">手机号码（选填）</label>
              <input
                type="text"
                v-model="regPhone"
                class="w-full bg-gray-50 border border-gray-200 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-md px-3.5 py-2.5 text-sm text-gray-800 transition-all outline-none"
                placeholder="联系方式"
              />
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">设置密码</label>
              <div class="relative">
                <input
                  :type="showRegPwd ? 'text' : 'password'"
                  required
                  v-model="regPassword"
                  class="w-full bg-gray-50 border border-gray-200 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-md px-3.5 py-2.5 pr-10 text-sm text-gray-800 transition-all outline-none"
                  placeholder="设置登录密码"
                />
                <button type="button" @click="showRegPwd = !showRegPwd" class="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600">
                  <Eye v-if="!showRegPwd" class="w-4 h-4" />
                  <EyeOff v-else class="w-4 h-4" />
                </button>
              </div>
            </div>
            <div class="space-y-1.5">
              <label class="text-xs font-medium text-gray-500 ml-1">确认密码</label>
              <div class="relative">
                <input
                  :type="showRegConfirmPwd ? 'text' : 'password'"
                  required
                  v-model="regConfirmPassword"
                  class="w-full bg-gray-50 border border-gray-200 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-md px-3.5 py-2.5 pr-10 text-sm text-gray-800 transition-all outline-none"
                  placeholder="再次输入密码"
                />
                <button type="button" @click="showRegConfirmPwd = !showRegConfirmPwd" class="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600">
                  <Eye v-if="!showRegConfirmPwd" class="w-4 h-4" />
                  <EyeOff v-else class="w-4 h-4" />
                </button>
              </div>
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
import { Eye, EyeOff, BookOpen } from 'lucide-vue-next';
import { toast } from '../toast';
import { apiPost } from '../apiClient';
import { saveAuthSession } from '../authSession';

// 与后端 services/core/config.py 的 MIN_PASSWORD_LENGTH 保持一致。
const MIN_PASSWORD_LENGTH = 8;

const emit = defineEmits(['login']);

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

const showLoginPwd = ref(false);
const showRegPwd = ref(false);
const showRegConfirmPwd = ref(false);

/** 切换登录和注册视图，并清空旧错误。 */
const setView = (newView) => {
  view.value = newView;
  error.value = '';
};
/** 提交统一账号凭证并保存后端签发的会话。 */
const handleLogin = async () => {
  error.value = '';
  loading.value = true;

  try {
    const data = await apiPost('/api/login', {
      username: username.value,
      password: password.value
    });
    saveAuthSession(data.access_token, data.user);
    emit('login', data.user);
  } catch (err) {
    error.value = err.message || '网络连接失败';
  } finally {
    loading.value = false;
  }
};
/** 校验注册表单后提交普通用户注册申请。 */
const handleRegister = async () => {
  error.value = '';
  if (regPassword.value !== regConfirmPassword.value) {
    error.value = '两次输入的密码不一致';
    return;
  }
  if (regPassword.value.length < MIN_PASSWORD_LENGTH) {
    error.value = `密码至少需要 ${MIN_PASSWORD_LENGTH} 个字符`;
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
