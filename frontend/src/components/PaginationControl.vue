<template>
  <div class="mt-4 flex flex-wrap items-center justify-between gap-3 text-sm text-gray-500">
    <span>共 {{ total }} 条</span>
    <div class="flex items-center gap-2">
      <label class="flex items-center gap-1.5 text-xs">
        <span>每页</span>
        <select
          :value="pageSize"
          :disabled="disabled"
          aria-label="每页显示数量"
          class="h-8 rounded-md border border-gray-200 bg-white px-2 text-xs text-gray-700 outline-none focus:border-blue-400 disabled:cursor-not-allowed disabled:bg-gray-50"
          @change="updatePageSize"
        >
          <option v-for="size in pageSizes" :key="size" :value="size">{{ size }}</option>
        </select>
        <span>条</span>
      </label>
      <span class="min-w-16 text-center text-xs tabular-nums">{{ page }} / {{ totalPages }}</span>
      <button
        type="button"
        title="上一页"
        aria-label="上一页"
        :disabled="disabled || page <= 1"
        class="flex h-8 w-8 items-center justify-center rounded-md border border-gray-200 bg-white text-gray-500 transition-colors hover:border-gray-300 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40"
        @click="$emit('change-page', page - 1)"
      >
        <ChevronLeft class="h-4 w-4" aria-hidden="true" />
      </button>
      <button
        type="button"
        title="下一页"
        aria-label="下一页"
        :disabled="disabled || page >= totalPages"
        class="flex h-8 w-8 items-center justify-center rounded-md border border-gray-200 bg-white text-gray-500 transition-colors hover:border-gray-300 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-40"
        @click="$emit('change-page', page + 1)"
      >
        <ChevronRight class="h-4 w-4" aria-hidden="true" />
      </button>
    </div>
  </div>
</template>

<script setup>
import { ChevronLeft, ChevronRight } from 'lucide-vue-next';

defineProps({
  page: { type: Number, required: true },
  pageSize: { type: Number, required: true },
  total: { type: Number, required: true },
  totalPages: { type: Number, required: true },
  disabled: { type: Boolean, default: false },
  pageSizes: { type: Array, default: () => [10, 20, 50] }
});

const emit = defineEmits(['change-page', 'change-page-size']);

/** Convert the native select value before requesting a new server page. */
const updatePageSize = event => emit('change-page-size', Number(event.target.value));
</script>
