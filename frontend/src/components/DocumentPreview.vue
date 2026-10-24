<template>
  <div v-if="filename && url" class="fixed inset-0 z-[70] flex items-center justify-center bg-black/40 p-2 sm:p-4">
    <div class="flex h-[94dvh] w-full max-w-[1680px] flex-col overflow-hidden rounded-lg bg-white shadow-xl sm:w-[94vw]">
      <div class="px-4 py-2 bg-gray-100 flex justify-between items-center gap-4">
        <span class="font-medium truncate">{{ contextLabel ? `${contextLabel} / ` : '' }}{{ filename }}</span>
        <button @click="$emit('close')" class="text-gray-500 hover:text-gray-800 text-xl font-bold" title="关闭">&times;</button>
      </div>
      <div v-if="evidence" class="px-4 py-3 border-b text-sm max-h-40 overflow-auto break-words">
        <div class="font-medium">{{ evidence.page }} {{ evidence.context_title }}</div>
        <p>{{ evidence.content }}</p>
      </div>
      <div v-if="loading" class="flex-1 flex items-center justify-center text-sm text-gray-500">文档加载中...</div>
      <div v-else-if="error" class="flex-1 flex items-center justify-center text-sm text-red-600">{{ error }}</div>
      <div v-else-if="isImage" class="flex-1 w-full h-full flex items-center justify-center bg-gray-50 p-4">
        <img :src="resolvedUrl" :alt="filename" class="max-w-full max-h-full object-contain" />
      </div>
      <iframe v-else :key="resolvedUrl" :src="resolvedUrl" class="flex-1 w-full h-full border-0" title="document-preview" />
    </div>
  </div>
</template>

<script setup>
import { computed, toRef } from 'vue';
import { useDocumentPreview } from '../composables/useDocumentPreview';

const props = defineProps({
  evidence: { type: Object, default: null },
  filename: { type: String, default: '' },
  url: { type: String, default: '' },
  contextLabel: { type: String, default: '' }
});
defineEmits(['close']);
const isImage = computed(() => /\.(jpeg|jpg|gif|png|webp)$/i.test(props.filename));
const { resolvedUrl, loading, error } = useDocumentPreview(toRef(props, 'url'));
</script>
