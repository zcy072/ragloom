<template>
  <div class="relative">
    <input
      ref="inputRef"
      :type="visible ? 'text' : 'password'"
      :value="modelValue"
      :required="required"
      :placeholder="placeholder"
      class="w-full bg-gray-50 border border-gray-200 focus:bg-white focus:border-blue-500 focus:ring-2 focus:ring-blue-100 rounded-md px-3.5 py-2.5 pr-10 text-sm text-gray-800 transition-all outline-none"
      @input="updateValue"
    />
    <button
      type="button"
      class="absolute inset-y-0 right-0 pr-3 flex items-center text-gray-400 hover:text-gray-600"
      :aria-label="visible ? '隐藏密码' : '显示密码'"
      @click="visible = !visible"
    >
      <EyeOff v-if="visible" class="w-4 h-4" />
      <Eye v-else class="w-4 h-4" />
    </button>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue';
import { Eye, EyeOff } from 'lucide-vue-next';

const props = defineProps({
  modelValue: { type: String, default: '' },
  placeholder: { type: String, default: '' },
  required: { type: Boolean, default: false },
  autofocus: { type: Boolean, default: false }
});

const emit = defineEmits(['update:modelValue']);

// 每个密码框各自控制是否明文显示，切换表单视图时不会互相影响。
const visible = ref(false);
const inputRef = ref(null);

const updateValue = event => emit('update:modelValue', event.target.value);

onMounted(() => {
  if (props.autofocus) inputRef.value?.focus();
});
</script>
