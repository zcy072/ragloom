<template>
  <div class="fixed top-4 left-1/2 -translate-x-1/2 z-[9999] flex flex-col items-center gap-2 pointer-events-none" role="status" aria-live="polite">
    <TransitionGroup name="toast">
      <div 
        v-for="t in toasts" 
        :key="t.id"
        class="px-4 py-3 rounded shadow-lg flex items-center min-w-[250px] max-w-md pointer-events-auto transition-all duration-300"
        :class="[
          t.type === 'error' ? 'bg-red-50 text-red-800 border-l-4 border-red-500' :
          t.type === 'success' ? 'bg-green-50 text-green-800 border-l-4 border-green-500' :
          'bg-white text-gray-800 border-l-4 border-blue-500'
        ]"
      >
        <component :is="getIcon(t.type)" class="w-5 h-5 mr-3 shrink-0" 
          :class="[
            t.type === 'error' ? 'text-red-500' :
            t.type === 'success' ? 'text-green-500' :
            'text-blue-500'
          ]"
        />
        <p class="text-sm font-medium">{{ t.message }}</p>
        <button @click="toast.remove(t.id)" class="ml-auto pl-3 text-gray-400 hover:text-gray-600">
          <X class="w-4 h-4" />
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>

<script setup>
import { toasts, toast } from '../toast';
import { AlertCircle, CheckCircle, Info, X } from 'lucide-vue-next';
/** 根据提示类型选择对应图标组件。 */
const getIcon = (type) => {
  if (type === 'error') return AlertCircle;
  if (type === 'success') return CheckCircle;
  return Info;
};
</script>

<style scoped>
.toast-enter-active,
.toast-leave-active {
  transition: all 0.3s ease;
}
.toast-enter-from {
  opacity: 0;
  transform: translateY(-20px) scale(0.95);
}
.toast-leave-to {
  opacity: 0;
  transform: translateY(-10px) scale(0.95);
}
</style>
