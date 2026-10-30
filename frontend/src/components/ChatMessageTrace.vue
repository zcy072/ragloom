<template>
  <div class="mt-2 text-xs text-gray-600">
    <div v-if="message.trace_id" class="flex items-center gap-1">
      <button type="button" :class="feedbackClass(1)" class="app-icon-button" title="回答有用" aria-label="回答有用" :disabled="saving" @click="rate(1)"><ThumbsUp class="h-3.5 w-3.5" /></button>
      <button type="button" :class="feedbackClass(-1)" class="app-icon-button" title="回答没用" aria-label="回答没用" :disabled="saving" @click="rate(-1)"><ThumbsDown class="h-3.5 w-3.5" /></button>
    </div>
    <details v-if="message.trace_id || message.stages?.length" class="mt-1 border-t pt-2" @toggle="onToggle">
      <summary class="cursor-pointer">处理过程<span v-if="totalMs" class="ml-1 tabular-nums text-gray-400">{{ formatElapsed(totalMs) }}</span></summary>
      <div class="mt-2">
        <ChatTraceDetail v-if="trace" :trace="trace" />
        <template v-else>
          <ChatStageTimeline v-if="message.stages?.length" :stages="message.stages" />
          <p v-if="loadError" class="mt-1 text-red-500">{{ loadError }}</p>
          <p v-else-if="loadingTrace" class="mt-1 text-gray-400">正在加载完整追踪…</p>
        </template>
      </div>
    </details>
  </div>
</template>
<script setup>
import { computed, ref } from 'vue';
import { ThumbsDown, ThumbsUp } from 'lucide-vue-next';

import { apiGet, apiPut } from '../apiClient.js';
import { formatElapsed } from '../chatTrace.js';
import { toast } from '../toast.js';
import ChatStageTimeline from './ChatStageTimeline.vue';
import ChatTraceDetail from './ChatTraceDetail.vue';

const props = defineProps({ message: { type: Object, required: true } });
const emit = defineEmits(['feedback']);
const trace = ref(null);
const loadingTrace = ref(false);
const loadError = ref('');
const saving = ref(false);

const totalMs = computed(() => (trace.value?.stages || props.message.stages || [])
  .reduce((sum, stage) => sum + (stage.elapsed_ms || 0), 0));

/** 第一次展开时读取完整追踪；追踪只在回答保存后才可读，读不到时保留实时阶段。 */
const onToggle = async event => {
  if (!event.target.open || trace.value || loadingTrace.value || !props.message.trace_id) return;
  loadingTrace.value = true;
  loadError.value = '';
  try {
    trace.value = await apiGet(`/api/chat/traces/${encodeURIComponent(props.message.trace_id)}`);
  } catch (error) {
    loadError.value = `追踪记录加载失败：${error.message}`;
  } finally {
    loadingTrace.value = false;
  }
};

const feedbackClass = value => (props.message.feedback === value ? 'text-teal-700' : 'text-gray-400');

/** 再次点击同一评价即取消。 */
const rate = async value => {
  const next = props.message.feedback === value ? 0 : value;
  saving.value = true;
  try {
    await apiPut(`/api/chat/traces/${encodeURIComponent(props.message.trace_id)}/feedback`, { feedback: next });
    emit('feedback', next);
  } catch (error) {
    toast.error(`评价保存失败：${error.message}`);
  } finally {
    saving.value = false;
  }
};
</script>
