<template>
  <div class="flex-1 p-6 bg-[#f6f9f9] w-full text-slate-800 overflow-y-auto">
    <div class="mb-6"><p class="text-[11px] font-semibold text-[#649092]">模型</p><h2 class="mt-1 text-lg font-semibold text-[#1e3437]">模型连接配置</h2></div>
    <div class="bg-[#edf7f6] text-[#466568] px-4 py-3 rounded-md text-[13px] mb-6 flex items-center border border-[#cfe5e2]">
      <span class="mr-2 text-[#267b79]">i</span> 页面配置优先于环境变量。保存后可将已配置模型设为当前默认模型。
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
            <input type="password" v-model="m.apiKey" :placeholder="m.keyPlaceholder || '请输入 API Key'" class="w-full border border-[#d8e4e4] rounded-md px-3 py-2 text-gray-700 focus:outline-none focus:border-[#65aaa6]" />
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
        
        <div class="pt-6 flex flex-wrap gap-2 mt-auto">
          <button @click="handleSave(m)" class="bg-[#176d70] text-white px-4 py-2 rounded-md text-[13px] hover:bg-[#10585c] transition-colors">保存</button>
          <button @click="handleActivate(m)" :disabled="m.isActive || !m.configured" class="bg-[#e6f5f3] text-[#176d70] px-4 py-2 rounded-md text-[13px] hover:bg-[#d7eeeb] transition-colors disabled:opacity-50 disabled:cursor-not-allowed">{{ m.isActive ? '当前默认模型' : '设为当前模型' }}</button>
          <button @click="handleTest(m)" class="bg-white border border-[#d8e4e4] text-[#586e71] px-4 py-2 rounded-md text-[13px] hover:bg-[#f4f8f8] transition-colors">检查配置</button>
          <button @click="handleReset(m)" class="bg-white border border-[#d8e4e4] text-[#586e71] px-4 py-2 rounded-md text-[13px] hover:bg-[#f4f8f8] transition-colors">重置为环境变量</button>
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
      isActive: d.is_active === 1,
      statusTag: d.configured ? '已配置' : '未配置',
      statusColor: d.configured ? 'text-orange-500 bg-orange-50 border-orange-100' : 'text-gray-400 bg-gray-50 border-gray-100',
      activeTag: d.is_active ? '当前默认' : '未激活',
      activeColor: d.is_active ? 'text-green-500 bg-green-50 border-green-100' : 'text-gray-400 bg-gray-50 border-gray-100',
      url: d.base_url,
      defaultModel: d.default_model,
      keyPlaceholder: d.configured ? '已配置，留空不修改' : '',
      apiKey: ''
    }));
  } catch (error) {
    console.error('Failed to fetch models', error);
    toast.error('模型配置加载失败');
  }
};

onMounted(refreshModels);
/** 保存模型地址、名称和可选的新 API Key。 */
const handleSave = async (model) => {
  try {
    const data = await apiPost('/api/models', {
        provider: model.name,
        api_key: model.apiKey.trim(),
        base_url: model.url,
        default_model: model.defaultModel
    });
    await refreshModels();
    toast.success(data.is_active ? `${model.name} 已保存并设为当前默认模型` : `${model.name} 配置已保存`);
  } catch (error) {
    toast.error(`保存失败: ${error.message || '网络请求异常'}`);
  }
};
/** 将已配置的模型切换为当前默认聊天模型。 */
const handleActivate = async (model) => {
  try {
    await apiPut(`/api/models/${encodeURIComponent(model.name)}/active`);
    await refreshModels();
    toast.success(`${model.name} 已设为当前默认模型`);
  } catch (error) {
    toast.error(`切换失败: ${error.message || '网络请求异常'}`);
  }
};
/** 检查后端是否能取得该模型的完整配置。 */
const handleTest = async (model) => {
  try {
    const data = await apiGet(`/api/models/check?provider=${encodeURIComponent(model.name)}`);
    if (data.configured) {
      toast.success(`${data.provider} 配置完整，默认模型: ${data.model || model.defaultModel}`);
    } else {
      toast.error('尚未保存可用的 API Key');
    }
  } catch (error) {
    toast.error(`检查失败: ${error.message || '网络请求异常'}`);
  }
};
/** 清除数据库中的 API Key，使模型重新使用环境变量配置。 */
const handleReset = async (model) => {
  try {
    await apiPost('/api/models', {
        provider: model.name,
        api_key: '',
        clear_api_key: true,
        base_url: model.url,
        default_model: model.defaultModel
    });
    await refreshModels();
    toast.success(model.name + ' 已重置，将回退使用环境变量配置');
  } catch (error) {
    toast.error(`重置失败: ${error.message || '网络请求异常'}`);
  }
};
</script>
