<template>
  <div class="chat-markdown" @click="onClick" v-html="rendered.html"></div>
</template>
<script setup>
import { computed } from 'vue';
import { renderAnswer } from '../chatMarkdown';
const props = defineProps({ content: { type: String, default: '' }, sources: { type: Array, default: () => [] } });
const emit = defineEmits(['citation']);
const rendered = computed(() => renderAnswer(props.content, props.sources));
const onClick = event => {
  const button = event.target.closest('button[data-citation]');
  const source = button && rendered.value.citations.get(Number(button.dataset.citation));
  if (source) emit('citation', source);
};
</script>
<style>
.chat-markdown { min-width: 0; overflow-wrap: anywhere; }
.chat-markdown > * + * { margin-top: 12px; }
.chat-markdown h1, .chat-markdown h2, .chat-markdown h3 { font-size: 18px; font-weight: 600; }
.chat-markdown ul { list-style: disc; padding-left: 24px; }
.chat-markdown ol { list-style: decimal; padding-left: 24px; }
.chat-markdown pre { overflow-x: auto; padding: 12px; background: #f3f6f6; border-radius: 4px; white-space: pre; }
.chat-markdown table { display: block; max-width: 100%; overflow-x: auto; border-collapse: collapse; }
.chat-markdown th, .chat-markdown td { border: 1px solid #dce7e7; padding: 6px 10px; }
.chat-markdown blockquote { border-left: 3px solid #8daaaa; padding-left: 12px; color: #52676a; }
.chat-markdown a, .chat-markdown .citation-link { color: #176d70; text-decoration: underline; }
</style>
