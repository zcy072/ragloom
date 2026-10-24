<template>
  <div class="border-b border-[#e5eded] bg-white px-6 py-3">
    <div class="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-5 xl:items-center">
      <select :value="modelValue.source" @change="update('source', $event.target.value)" class="h-9 min-w-0 rounded-md border border-[#d8e4e4] bg-white px-2 text-xs text-gray-700 focus:outline-none focus:border-[#68aaa7]">
        <option value="">全部来源</option>
        <option v-for="source in options.sources" :key="source" :value="source">{{ source }}</option>
      </select>
      <select :value="modelValue.document_type" @change="update('document_type', $event.target.value)" class="h-9 min-w-0 rounded-md border border-[#d8e4e4] bg-white px-2 text-xs text-gray-700 focus:outline-none focus:border-[#68aaa7]">
        <option value="">全部类型</option>
        <option v-for="type in options.document_types" :key="type" :value="type">{{ type }}</option>
      </select>
      <select :value="modelValue.tag" @change="update('tag', $event.target.value)" class="h-9 min-w-0 rounded-md border border-[#d8e4e4] bg-white px-2 text-xs text-gray-700 focus:outline-none focus:border-[#68aaa7]">
        <option value="">全部标签</option>
        <option v-for="tag in options.tags" :key="tag" :value="tag">{{ tag }}</option>
      </select>
      <div class="flex min-w-0 items-center gap-2">
        <input :value="modelValue.effective_date_from" @input="update('effective_date_from', $event.target.value)" type="date" class="h-9 min-w-0 flex-1 rounded-md border border-[#d8e4e4] px-2 text-xs text-gray-700 focus:outline-none focus:border-[#68aaa7]" title="生效开始日期" />
        <span class="text-xs text-gray-400">至</span>
        <input :value="modelValue.effective_date_to" @input="update('effective_date_to', $event.target.value)" type="date" class="h-9 min-w-0 flex-1 rounded-md border border-[#d8e4e4] px-2 text-xs text-gray-700 focus:outline-none focus:border-[#68aaa7]" title="生效结束日期" />
      </div>
      <button @click="$emit('reset')" class="inline-flex h-9 w-9 items-center justify-center rounded-md border border-[#d8e4e4] text-gray-500 hover:bg-[#f4f8f8]" title="清除筛选">
        <RotateCcw class="h-3.5 w-3.5" />
      </button>
    </div>
  </div>
</template>

<script setup>
import { RotateCcw } from 'lucide-vue-next';

const props = defineProps({
  modelValue: { type: Object, required: true },
  options: {
    type: Object,
    default: () => ({ sources: [], document_types: [], tags: [] })
  }
});
const emit = defineEmits(['update:modelValue', 'reset']);
/** 更新单个筛选字段，并通过 v-model 向父组件提交新对象。 */
const update = (field, value) => {
  emit('update:modelValue', { ...props.modelValue, [field]: value });
};
</script>
