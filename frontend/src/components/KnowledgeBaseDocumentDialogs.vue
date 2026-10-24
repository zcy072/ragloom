<template>
  <div v-if="metadataDocument" class="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 p-4">
    <div class="w-full max-w-lg rounded-lg bg-white shadow-xl">
      <header class="flex items-center justify-between border-b border-gray-100 px-5 py-4">
        <div class="min-w-0"><h3 class="text-sm font-semibold text-gray-800">文档元数据</h3><p class="mt-1 truncate text-xs text-gray-400">{{ metadataDocument.filename }}</p></div>
        <button @click="$emit('cancel-metadata')" class="inline-flex h-8 w-8 items-center justify-center rounded text-gray-400 hover:bg-gray-100" title="关闭"><X class="h-4 w-4" /></button>
      </header>
      <div class="grid grid-cols-1 gap-4 p-5 sm:grid-cols-2">
        <label v-for="field in metadataFields" :key="field.key" class="text-xs text-gray-600">{{ field.label }}
          <input :value="metadata[field.key]" @input="updateMetadata(field.key, $event.target.value)" :type="field.type" :maxlength="field.maxlength" class="mt-1.5 w-full border border-gray-200 rounded px-3 py-2 text-sm text-gray-800 focus:outline-none focus:border-blue-400" />
        </label>
      </div>
      <footer class="flex justify-end gap-2 border-t border-gray-100 bg-gray-50 px-5 py-4">
        <button @click="$emit('cancel-metadata')" :disabled="savingMetadata" class="px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-100 rounded-md disabled:opacity-50">取消</button>
        <button @click="$emit('save-metadata')" :disabled="savingMetadata" class="px-3 py-1.5 text-xs text-white bg-[#176d70] hover:bg-[#10585c] rounded-md disabled:opacity-50">{{ savingMetadata ? '保存中...' : '保存' }}</button>
      </footer>
    </div>
  </div>

  <div v-if="parserDocument" class="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 p-4">
    <div class="w-full max-w-lg rounded-lg bg-white shadow-xl">
      <header class="flex items-center justify-between border-b border-gray-100 px-5 py-4">
        <div class="min-w-0"><h3 class="text-sm font-semibold text-gray-800">重新解析文档</h3><p class="mt-1 truncate text-xs text-gray-400">{{ parserDocument.filename }}</p></div>
        <button @click="$emit('cancel-parser')" class="inline-flex h-8 w-8 items-center justify-center rounded text-gray-400 hover:bg-gray-100" title="关闭"><X class="h-4 w-4" /></button>
      </header>
      <div class="p-5">
        <label class="block text-xs text-gray-600">解析方式
          <select :value="parserMode" @change="$emit('update:parserMode', $event.target.value)" class="mt-1.5 w-full border border-gray-200 rounded px-3 py-2 text-sm text-gray-800 focus:outline-none focus:border-blue-400">
            <option v-for="option in parserOptions" :key="option.id" :value="option.id" :disabled="!option.available && option.id !== parserMode">{{ option.label }}{{ option.available ? '' : '（未配置）' }}</option>
          </select>
        </label>
        <p v-if="parserMode === 'mineru_cloud'" class="mt-3 text-xs text-amber-700">该文档将发送至 MinerU 云端解析。</p>
        <p v-else-if="parserMode === 'vision_model'" class="mt-3 text-xs text-amber-700">该图片会发送给已配置的视觉模型解析，失败时改用本地 Paddle OCR。</p>
      </div>
      <footer class="flex justify-end gap-2 border-t border-gray-100 bg-gray-50 px-5 py-4">
        <button @click="$emit('cancel-parser')" :disabled="savingParser" class="px-3 py-1.5 text-xs text-gray-600 hover:bg-gray-100 rounded-md disabled:opacity-50">取消</button>
        <button @click="$emit('save-parser')" :disabled="savingParser" class="px-3 py-1.5 text-xs text-white bg-[#176d70] hover:bg-[#10585c] rounded-md disabled:opacity-50">{{ savingParser ? '提交中...' : '重新解析' }}</button>
      </footer>
    </div>
  </div>
</template>

<script setup>
import { X } from 'lucide-vue-next';

const props = defineProps({
  metadataDocument: { type: Object, default: null },
  metadata: { type: Object, required: true },
  savingMetadata: { type: Boolean, default: false },
  parserDocument: { type: Object, default: null },
  parserMode: { type: String, default: 'native_text' },
  parserOptions: { type: Array, default: () => [] },
  savingParser: { type: Boolean, default: false }
});
const emit = defineEmits(['cancel-metadata', 'save-metadata', 'cancel-parser', 'save-parser', 'update:metadata', 'update:parserMode']);

const updateMetadata = (key, value) => {
  emit('update:metadata', { ...props.metadata, [key]: value });
};

const metadataFields = [
  { key: 'document_type', label: '文档类型', type: 'text', maxlength: 80 },
  { key: 'tags', label: '标签', type: 'text' },
  { key: 'effective_date', label: '生效日期', type: 'date' }
];
</script>
