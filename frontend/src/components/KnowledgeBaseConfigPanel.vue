<template>
  <div class="mb-6">
    <div v-if="chunkConfig.index_needs_rebuild" class="mb-3 flex items-center justify-between gap-3 border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800 rounded-md">
      <span>检索索引需要更新</span>
      <button @click="$emit('rebuild')" :disabled="rebuildingIndex" class="inline-flex h-7 w-7 items-center justify-center rounded border border-amber-300 bg-white text-amber-700 hover:bg-amber-100 disabled:opacity-50" title="重建索引">
        <RefreshCw :class="['h-3.5 w-3.5', rebuildingIndex ? 'animate-spin' : '']" />
      </button>
    </div>
    <button @click="showChunkConfig = !showChunkConfig" class="flex items-center gap-1.5 text-xs font-medium text-gray-500 hover:text-blue-600 transition-colors">
      <svg :class="['w-3.5 h-3.5 transition-transform', showChunkConfig ? 'rotate-90' : '']" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
      ⚙️ 分段配置（当前：{{ chunkConfig.chunk_size }} 字符 / 重叠 {{ chunkConfig.chunk_overlap }} 字符）
    </button>
    <div v-if="showChunkConfig" class="mt-3 bg-gray-50 border border-gray-200 rounded-lg p-4 space-y-4 text-sm">
      <div class="grid grid-cols-2 gap-4">
        <div>
          <label class="block text-xs font-medium text-gray-700 mb-1">分段长度 <span class="text-gray-400 font-normal">（chunk_size）</span></label>
          <input type="number" min="100" max="4000" v-model="chunkConfig.chunk_size" class="w-full border border-gray-200 rounded px-2 py-1.5 text-sm focus:outline-none focus:border-blue-400" />
          <p class="text-[11px] text-gray-400 mt-1">每段最多多少字符。<br/>推荐：短问答 300，通用 800，长文 1500</p>
        </div>
        <div>
          <label class="block text-xs font-medium text-gray-700 mb-1">重叠长度 <span class="text-gray-400 font-normal">（chunk_overlap）</span></label>
          <input type="number" min="0" :max="Math.max(0, Number(chunkConfig.chunk_size || 0) - 1)" v-model="chunkConfig.chunk_overlap" class="w-full border border-gray-200 rounded px-2 py-1.5 text-sm focus:outline-none focus:border-blue-400" />
          <p class="text-[11px] text-gray-400 mt-1">相邻段之间重复的字符数，避免语义断裂。<br/>推荐：分段长度的 10%～20%</p>
        </div>
      </div>
      <div class="flex items-center justify-between pt-1 border-t border-gray-200">
        <button @click="$emit('reset-chunk')" class="text-xs text-gray-400 hover:text-gray-600 underline">恢复默认值</button>
        <div class="flex items-center gap-3">
          <p class="text-xs text-gray-400">保存后会重建全部文档索引</p>
          <button @click="$emit('save-chunk')" :disabled="savingChunk" class="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1.5 rounded text-xs font-medium disabled:opacity-50">{{ savingChunk ? '保存中...' : '保存配置' }}</button>
        </div>
      </div>
    </div>
  </div>

  <div class="mb-6">
    <button @click="showRetrievalConfig = !showRetrievalConfig" class="flex items-center gap-1.5 text-xs font-medium text-gray-500 hover:text-blue-600 transition-colors">
      <SlidersHorizontal class="h-3.5 w-3.5" />
      检索设置
    </button>
    <div v-if="showRetrievalConfig" class="mt-3 border border-gray-200 bg-gray-50 p-4 rounded-lg">
      <label class="flex items-center justify-between gap-4 text-sm text-gray-700">
        <span>追问查询改写</span>
        <span class="relative inline-flex h-5 w-9 flex-shrink-0 items-center">
          <input v-model="retrievalConfig.query_rewrite_enabled" @change="$emit('save-retrieval')" type="checkbox" class="peer sr-only" />
          <span class="absolute inset-0 rounded-full bg-gray-300 transition-colors peer-checked:bg-[#176d70]"></span>
          <span class="absolute left-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform peer-checked:translate-x-4"></span>
        </span>
      </label>
    </div>
  </div>
</template>

<script setup>
import { RefreshCw, SlidersHorizontal } from 'lucide-vue-next';

defineProps({
  savingChunk: { type: Boolean, default: false },
  rebuildingIndex: { type: Boolean, default: false }
});
defineEmits(['rebuild', 'reset-chunk', 'save-chunk', 'save-retrieval']);

const chunkConfig = defineModel('chunkConfig', { type: Object, required: true });
const retrievalConfig = defineModel('retrievalConfig', { type: Object, required: true });
const showChunkConfig = defineModel('showChunkConfig', { type: Boolean, default: false });
const showRetrievalConfig = defineModel('showRetrievalConfig', { type: Boolean, default: false });
</script>
