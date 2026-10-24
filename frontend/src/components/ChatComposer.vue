<template>
  <div class="p-4 border-t border-[#e5eded] bg-[#fbfdfd]">
    <div v-if="attachedImage" class="mb-2 flex items-center gap-2">
      <div class="relative inline-block">
        <img :src="attachedImage.previewUrl" alt="预览" class="h-16 w-16 object-cover rounded-lg border border-gray-200" />
        <button @click="$emit('remove-image')" class="absolute -top-1.5 -right-1.5 bg-red-500 text-white rounded-full w-4 h-4 flex items-center justify-center text-[10px] hover:bg-red-600">×</button>
      </div>
      <span class="text-xs text-gray-400 truncate max-w-[160px]">{{ attachedImage.file.name }}</span>
    </div>
    <input ref="imageInput" type="file" accept="image/jpeg,image/png,image/webp,image/gif" class="hidden" @change="$emit('image-select', $event)" />
    <div class="relative w-full border border-[#d7e4e4] rounded-lg overflow-hidden bg-white focus-within:border-[#62a9a5] focus-within:ring-2 focus-within:ring-[#d9f0ed] transition-colors flex flex-col">
      <textarea
        :value="modelValue"
        :maxlength="maxLength"
        :disabled="loading || disabled"
        @input="$emit('update:modelValue', $event.target.value)"
        @keydown.enter.exact.prevent="!$event.isComposing && $event.keyCode !== 229 && $emit('send')"
        @paste="$emit('image-paste', $event)"
        placeholder="输入问题，Enter 发送，Shift+Enter 换行"
        class="w-full resize-none outline-none text-[14px] p-4 text-gray-800 min-h-[80px]"
      ></textarea>
      <div class="flex items-center justify-between px-4 py-2 bg-gray-50/50 border-t border-gray-100">
        <div class="flex items-center space-x-3">
          <button
            @click="imageInput?.click()"
            :disabled="loading || disabled"
            title="上传图片"
            class="flex items-center justify-center w-8 h-8 rounded-full transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            :class="attachedImage ? 'text-blue-500 bg-blue-50' : 'text-gray-400 hover:text-blue-500 hover:bg-blue-50'"
          >
            <ImageIcon class="h-4 w-4" />
          </button>
        </div>
        <button v-if="loading" @click="$emit('stop')" title="停止生成" class="flex items-center justify-center w-8 h-8 rounded-full bg-gray-100/80 hover:bg-gray-200 transition-colors">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="12" r="10" stroke="#1f2937" stroke-width="2"/>
            <rect x="8.5" y="8.5" width="7" height="7" fill="#1f2937" rx="1"/>
          </svg>
        </button>
        <button v-else @click="$emit('send')" :disabled="disabled || (!modelValue.trim() && !attachedImage)" class="flex items-center bg-[#176d70] text-white px-4 py-1.5 rounded-md text-[13px] hover:bg-[#10585c] transition-colors disabled:opacity-50">
          发送
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';
import { ImageIcon } from 'lucide-vue-next';

defineProps({
  maxLength: { type: Number, default: undefined },
  modelValue: { type: String, default: '' },
  loading: { type: Boolean, default: false },
  disabled: { type: Boolean, default: false },
  attachedImage: { type: Object, default: null }
});
defineEmits(['update:modelValue', 'send', 'stop', 'image-select', 'image-paste', 'remove-image']);

const imageInput = ref(null);
</script>
