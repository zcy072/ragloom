<template>
  <div class="flex-1 p-6 bg-[#f6f9f9] w-full text-slate-800 overflow-y-auto">
    <div class="mb-6"><p class="text-[11px] font-semibold text-[#649092]">模型</p><h2 class="mt-1 text-lg font-semibold text-[#1e3437]">模型连接配置</h2></div>
    <div class="bg-[#edf7f6] text-[#466568] px-4 py-3 rounded-md text-[13px] mb-6 flex items-center border border-[#cfe5e2]">
      <span class="mr-2 text-[#267b79]">i</span> API Key 加密保存在服务端，页面不回显。“已保存密钥”只表示已存有密钥，不代表服务可连接；保存后可设为当前默认模型。
    </div>

    <div class="grid grid-cols-1 xl:grid-cols-2 gap-5">
      <div v-for="m in models" :key="m.name" class="bg-white rounded-lg shadow-[0_1px_2px_rgba(18,53,57,0.04)] border border-[#e0e9e9] p-6 flex flex-col">
        <div class="flex justify-between items-center mb-5">
          <h2 class="text-[15px] font-semibold text-[#20373a]">{{ m.name }}</h2>
          <div class="flex space-x-2">
            <span :class="['px-2 py-0.5 rounded text-[10px] border', m.statusColor]">{{ m.statusTag }}</span>
            <span :class="['px-2 py-0.5 rounded text-[10px] border', m.activeColor]">{{ m.activeTag }}</span>
          </div>
        </div>

        <div v-if="m.msg" :class="['px-4 py-2 rounded text-[12px] mb-4', m.msgBg]">
          {{ m.msg }}
        </div>

        <div class="space-y-4 text-sm flex-1">
          <div>
            <label class="block text-gray-500 mb-1 text-[11px]">API Key</label>
            <input type="password" v-model="m.apiKey" maxlength="4096" autocomplete="off" :placeholder="m.keyPlaceholder || '请输入 API Key'" class="w-full border border-[#d8e4e4] rounded-md px-3 py-2 text-gray-700 focus:outline-none focus:border-[#65aaa6]" />
          </div>
          
          <div>
            <label class="block text-gray-500 mb-1 text-[11px]">Base URL</label>
            <input type="text" v-model="m.url" placeholder="https://..." class="w-full border border-[#d8e4e4] rounded-md px-3 py-2 text-gray-700 bg-[#fbfdfd] focus:bg-white focus:outline-none focus:border-[#65aaa6]" />
          </div>
          
          <div>
            <label class="block text-gray-500 mb-1 text-[11px]">默认模型</label>
            <input type="text" v-model="m.defaultModel" class="w-full border border-[#d8e4e4] rounded-md px-3 py-2 text-gray-700 focus:outline-none focus:border-[#65aaa6]" />
          </div>
        </div>

        <p class="pt-4 text-[11px] text-[#7d9294]">“清除 API Key”会删除已保存的密钥并停用该凭证，不会回退到环境变量。</p>
        <div class="pt-4 flex flex-wrap gap-2 mt-auto">
          <button @click="handleSave(m)" :disabled="m.busy" class="bg-[#176d70] text-white px-4 py-2 rounded-md text-[13px] hover:bg-[#10585c] transition-colors">保存</button>
          <button @click="handleActivate(m)" :disabled="m.isActive || !m.configured" class="bg-[#e6f5f3] text-[#176d70] px-4 py-2 rounded-md text-[13px] hover:bg-[#d7eeeb] transition-colors disabled:opacity-50 disabled:cursor-not-allowed">{{ m.isActive ? '当前默认模型' : '设为当前模型' }}</button>
          <button @click="handleTest(m)" class="bg-white border border-[#d8e4e4] text-[#586e71] px-4 py-2 rounded-md text-[13px] hover:bg-[#f4f8f8] transition-colors">检查配置</button>
          <button @click="handleClearKey(m)" :disabled="m.busy || !m.configured" title="删除已保存的 API Key，该模型将无法调用，直到重新填写" class="bg-white border border-[#d8e4e4] text-[#a84444] px-4 py-2 rounded-md text-[13px] hover:bg-[#fff2f0] transition-colors disabled:opacity-50 disabled:cursor-not-allowed">清除 API Key</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import { toast } from '../toast';
import { apiGet, apiPost, apiPut } from '../apiClient';

const models = ref([]);
/** 读取模型配置，并转换为页面卡片使用的展示字段。 */
const refreshModels = async () => {
  try {
    const data = await apiGet('/api/models');
    models.value = data.map(d => ({
      name: d.provider,
      configured: d.configured,
      isActive: d.is_active,
      statusTag: d.configured ? '已保存密钥' : '未保存密钥',
      statusColor: d.configured ? 'text-orange-500 bg-orange-50 border-orange-100' : 'text-gray-400 bg-gray-50 border-gray-100',
      activeTag: d.is_active ? '当前默认' : '未激活',
      activeColor: d.is_active ? 'text-green-500 bg-green-50 border-green-100' : 'text-gray-400 bg-gray-50 border-gray-100',
      url: d.base_url,
      defaultModel: d.default_model,
      keyPlaceholder: d.configured ? '已保存密钥，留空不修改（修改 Base URL 时须重新填写）' : '',
      apiKey: '',
      busy: false,
      msg: '',
      msgBg: ''
    }));
  } catch (error) {
    console.error('Failed to fetch models', error);
    toast.error('模型配置加载失败');
  }
};

onMounted(refreshModels);
/** 保存模型地址、名称和可选的新 API Key；成功后清空输入框，Key 不留在页面里。 */
const handleSave = async (model) => {
  if (model.busy) return;
  model.busy = true;
  try {
    const data = await apiPost('/api/models', {
        provider: model.name,
        api_key: model.apiKey.trim(),
        base_url: model.url,
        default_model: model.defaultModel
    });
    model.apiKey = '';
    await refreshModels();
    toast.success(data.is_active ? `${model.name} 已保存并设为当前默认模型` : `${model.name} 配置已保存`);
  } catch (error) {
    toast.error(`保存失败: ${error.message || '网络请求异常'}`);
  } finally {
    model.busy = false;
  }
};
/** 将已保存密钥的模型切换为当前默认聊天模型。 */
const handleActivate = async (model) => {
  try {
    await apiPut(`/api/models/${encodeURIComponent(model.name)}/active`);
    await refreshModels();
    toast.success(`${model.name} 已设为当前默认模型`);
  } catch (error) {
    toast.error(`切换失败: ${error.message || '网络请求异常'}`);
  }
};
/** 在服务端本地检查该模型的已保存配置能否读取；不联网、不代表上游服务可用。 */
const handleTest = async (model) => {
  model.msg = '';
  try {
    const data = await apiGet(`/api/models/check?provider=${encodeURIComponent(model.name)}`);
    if (data.configured) {
      toast.success(`${data.provider} 已保存密钥且可读取，默认模型: ${data.model || model.defaultModel}`);
    } else {
      toast.error('尚未保存 API Key');
    }
  } catch (error) {
    if (error.status === 503 && error.data?.code === 'model_key_unavailable') {
      // 密钥无法解密时仍可直接替换或清除，不需要先读出旧值。
      model.msg = '已保存的密钥无法读取：请检查部署的主密钥配置，或重新填写 API Key 后保存，也可以直接清除。';
      model.msgBg = 'bg-amber-50 text-amber-800 border border-amber-100';
      return;
    }
    toast.error(`检查失败: ${error.message || '网络请求异常'}`);
  }
};
/** 删除服务端保存的 API Key；该凭证随即停用，不会回退到环境变量。 */
const handleClearKey = async (model) => {
  if (model.busy) return;
  if (!window.confirm(`清除 ${model.name} 的 API Key 后，该模型将无法调用，直到重新填写。确定清除吗？`)) return;
  model.busy = true;
  try {
    await apiPost('/api/models', {
        provider: model.name,
        clear_api_key: true,
        base_url: model.url,
        default_model: model.defaultModel
    });
    model.apiKey = '';
    await refreshModels();
    toast.success(`${model.name} 的 API Key 已清除，该凭证已停用`);
  } catch (error) {
    toast.error(`清除失败: ${error.message || '网络请求异常'}`);
  } finally {
    model.busy = false;
  }
};
</script>
