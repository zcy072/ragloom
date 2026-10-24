<template>
  <div v-if="!readonly" class="mb-4 grid grid-cols-1 gap-3 border border-gray-200 bg-gray-50 p-4 rounded-lg">
    <details class="group">
      <summary class="flex cursor-pointer list-none items-center justify-between py-1 text-xs font-medium text-gray-600 [&::-webkit-details-marker]:hidden">
        <span>更多元数据（可选）</span>
        <ChevronDown class="h-4 w-4 text-gray-400 transition-transform group-open:rotate-180" />
      </summary>
      <div class="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <label class="text-xs text-gray-600">文档类型
          <input v-model="uploadMetadata.document_type" type="text" maxlength="80" placeholder="例如：制度、指南" class="mt-1.5 w-full border border-gray-200 rounded px-3 py-2 text-sm text-gray-800 focus:outline-none focus:border-blue-400" />
        </label>
        <label class="text-xs text-gray-600">标签
          <input v-model="uploadMetadata.tags" type="text" placeholder="使用逗号分隔" class="mt-1.5 w-full border border-gray-200 rounded px-3 py-2 text-sm text-gray-800 focus:outline-none focus:border-blue-400" />
        </label>
        <label class="text-xs text-gray-600">生效日期
          <input v-model="uploadMetadata.effective_date" type="date" class="mt-1.5 w-full border border-gray-200 rounded px-3 py-2 text-sm text-gray-800 focus:outline-none focus:border-blue-400" />
        </label>
      </div>
    </details>
    <label class="text-xs text-gray-600">PDF / DOCX 解析方式
      <select v-model="uploadParserMode" class="mt-1.5 w-full border border-gray-200 rounded px-3 py-2 text-sm text-gray-800 focus:outline-none focus:border-blue-400">
        <option v-for="option in parserOptions" :key="option.id" :value="option.id" :disabled="!option.available">{{ option.label }}{{ option.available ? '' : '（未配置）' }}</option>
      </select>
      <span v-if="uploadParserMode === 'mineru_cloud'" class="mt-1 block text-[11px] text-amber-700">本次上传的 PDF 或 DOCX 将发送至 MinerU 云端解析，其他格式仍在本地处理。</span>
    </label>
    <label class="text-xs text-gray-600">图片解析方式
      <select v-model="uploadImageParserMode" class="mt-1.5 w-full border border-gray-200 rounded px-3 py-2 text-sm text-gray-800 focus:outline-none focus:border-blue-400">
        <option v-for="option in imageParserOptions" :key="option.id" :value="option.id" :disabled="!option.available">{{ option.label }}{{ option.available ? '' : '（未配置）' }}</option>
      </select>
      <span v-if="uploadImageParserMode === 'vision_model'" class="mt-1 block text-[11px] text-amber-700">图片会发送给已配置的视觉模型解析，失败时改用本地 Paddle OCR。</span>
    </label>
  </div>
  <input v-if="!readonly" ref="fileInput" type="file" multiple class="hidden" :accept="uploadAccept" @change="$emit('upload', $event)" />
  <div class="flex items-center justify-between mb-3">
    <h4 class="text-sm font-semibold text-gray-700">已上传文档 ({{ documents.length }})</h4>
    <button v-if="!readonly" @click="fileInput?.click()" :disabled="isUploading" class="flex items-center gap-1.5 bg-blue-500 hover:bg-blue-600 text-white px-3 py-1.5 rounded-md text-xs font-medium transition-colors shadow-sm disabled:opacity-50">
      <Plus class="w-3.5 h-3.5" />
      {{ isUploading ? '上传中...' : '添加新文档' }}
    </button>
  </div>
  <div class="border border-gray-100 rounded-lg bg-white text-sm">
    <div v-if="documents.length === 0" class="p-8 text-center text-gray-400">暂无文档，请先上传</div>
    <div v-else class="divide-y divide-gray-100">
      <div v-for="document in documents" :key="document.filename" class="px-4 py-3 flex flex-col items-stretch justify-between gap-3 hover:bg-gray-50 relative first:rounded-t-lg last:rounded-b-lg sm:flex-row sm:items-center sm:gap-4">
        <div class="flex items-center flex-1 min-w-0 sm:pr-4">
          <FileText class="w-4 h-4 text-gray-400 mr-3 flex-shrink-0" />
          <div class="min-w-0">
            <div class="truncate text-gray-700 cursor-pointer text-blue-600 hover:underline max-w-[260px]" @click="$emit('preview', document)" :title="document.filename">{{ document.filename }}</div>
            <div v-if="document.document_type || document.tags?.length || document.effective_date || isSelectableParserDocument(document)" class="mt-1 flex min-w-0 flex-wrap items-center gap-1 text-[10px] text-gray-500">
              <span v-if="document.document_type" class="rounded bg-slate-100 px-1.5 py-0.5">{{ document.document_type }}</span>
              <span v-for="tag in document.tags || []" :key="tag" class="rounded bg-teal-50 px-1.5 py-0.5 text-teal-700">{{ tag }}</span>
              <span v-if="document.effective_date" class="rounded bg-amber-50 px-1.5 py-0.5 text-amber-700">{{ document.effective_date }}</span>
              <span v-if="isSelectableParserDocument(document)" class="rounded bg-blue-50 px-1.5 py-0.5 text-blue-700">{{ parserLabel(document.parser_mode) }}</span>
            </div>
          </div>
          <span v-if="document.status === 'processing'" class="ml-2 px-1.5 py-0.5 rounded text-[10px] bg-blue-50 text-blue-500 border border-blue-100 flex items-center whitespace-nowrap"><svg class="animate-spin -ml-0.5 mr-1 h-3 w-3 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>解析处理中</span>
          <span v-if="document.status === 'error'" :title="document.error_message || '索引任务失败'" class="ml-2 px-1.5 py-0.5 rounded text-[10px] bg-red-50 text-red-500 border border-red-100 whitespace-nowrap">解析失败</span>
        </div>
        <div class="flex w-full items-center justify-between gap-3 text-xs text-gray-400 relative sm:w-auto sm:flex-shrink-0 sm:justify-end">
          <span>{{ formatSize(document.size) }}</span>
          <span class="text-gray-500">{{ formatUploadedAt(document.uploaded_at) }}</span>
          <template v-if="!readonly">
            <button @click="$emit('edit-metadata', document)" class="inline-flex h-7 w-7 items-center justify-center rounded text-gray-400 hover:bg-blue-50 hover:text-blue-600" title="编辑文档元数据"><Pencil class="h-3.5 w-3.5" /></button>
            <button v-if="isSelectableParserDocument(document)" @click="$emit('edit-parser', document)" class="inline-flex h-7 w-7 items-center justify-center rounded text-gray-400 hover:bg-blue-50 hover:text-blue-600" title="更换解析方式"><RefreshCw class="h-3.5 w-3.5" /></button>
            <div v-if="deleteConfirmId === document.filename" class="flex items-center space-x-1.5 bg-red-50 p-0.5 pr-1 rounded-md animate-fade-in-right">
              <span class="text-red-500 font-medium px-2">确认删除?</span>
              <button @click="deleteConfirmId = null" class="px-2.5 py-1 hover:bg-white text-gray-500 rounded transition-colors" :disabled="deletingDocumentId === document.filename">取消</button>
              <button @click="$emit('delete', document.filename)" :disabled="deletingDocumentId === document.filename" :class="['px-2.5 py-1 text-white rounded transition-colors shadow-sm', deletingDocumentId === document.filename ? 'bg-red-400 cursor-not-allowed' : 'bg-red-500 hover:bg-red-600']">{{ deletingDocumentId === document.filename ? '删除中...' : '删除' }}</button>
            </div>
            <button v-else @click="deleteConfirmId = document.filename" class="flex-shrink-0 whitespace-nowrap text-red-400 hover:text-red-600 px-2 py-1 hover:bg-red-50 rounded transition-colors">删除</button>
          </template>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { ChevronDown, FileText, Pencil, Plus, RefreshCw } from 'lucide-vue-next';

defineProps({
  readonly: { type: Boolean, default: false },
  documents: { type: Array, default: () => [] },
  isUploading: { type: Boolean, default: false },
  deletingDocumentId: { type: String, default: null },
  parserOptions: { type: Array, default: () => [] },
  imageParserOptions: { type: Array, default: () => [] },
  uploadAccept: { type: String, default: '' },
  isSelectableParserDocument: { type: Function, required: true },
  parserLabel: { type: Function, required: true }
});
defineEmits(['upload', 'preview', 'edit-metadata', 'edit-parser', 'delete']);

const uploadMetadata = defineModel('uploadMetadata', { type: Object, required: true });
const uploadParserMode = defineModel('uploadParserMode', { type: String, required: true });
const uploadImageParserMode = defineModel('uploadImageParserMode', { type: String, required: true });
const deleteConfirmId = defineModel('deleteConfirmId', { type: String, default: null });
const fileInput = ref(null);
/** 将字节数格式化为便于阅读的 KB 或 MB。 */
const formatSize = size => size < 1024 * 1024
  ? `${(size / 1024).toFixed(1)} KB`
  : `${(size / 1024 / 1024).toFixed(2)} MB`;
/** 将带时区的 ISO 时间转换为本地中文时间；无法解析时原样显示。 */
const formatUploadedAt = value => {
  if (!value) return '';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime())
    ? value
    : parsed.toLocaleString('zh-CN', { hour12: false });
};
</script>
