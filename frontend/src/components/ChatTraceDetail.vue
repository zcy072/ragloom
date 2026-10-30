<template>
  <div class="space-y-4 text-xs text-gray-600 break-words">
    <div v-if="trace.hints?.length" class="rounded-md border border-amber-200 bg-amber-50 p-3 text-amber-800">
      <div class="mb-1 font-medium">可能的问题</div>
      <ul class="list-disc space-y-1 pl-4">
        <li v-for="hint in trace.hints" :key="hint">{{ hint }}</li>
      </ul>
    </div>
    <dl class="space-y-1">
      <div><dt class="inline text-gray-400">原始问题：</dt><dd class="inline">{{ trace.question }}</dd></div>
      <div><dt class="inline text-gray-400">实际检索问题：</dt><dd class="inline">{{ trace.skipped ? '跳过检索' : trace.actual_query }}</dd></div>
      <div v-if="trace.variant_query"><dt class="inline text-gray-400">补充查询：</dt><dd class="inline">{{ trace.variant_query }}</dd></div>
      <div><dt class="inline text-gray-400">发给模型的总字数：</dt><dd class="inline">{{ trace.prompt_chars }}</dd></div>
    </dl>
    <section>
      <h4 class="mb-1 font-medium text-gray-700">处理阶段</h4>
      <ChatStageTimeline :stages="trace.stages || []" />
    </section>
    <section v-if="trace.agentic_queries?.length">
      <h4 class="mb-1 font-medium text-gray-700">补充检索</h4>
      <div v-for="(item, index) in trace.agentic_queries" :key="index">第 {{ index + 1 }} 轮：{{ item.query }}（新增 {{ item.results }} 条）</div>
    </section>
    <section v-if="trace.reranked?.length">
      <h4 class="mb-1 font-medium text-gray-700">重排结果（{{ trace.reranked.length }} 条）</h4>
      <TraceDocumentList :documents="trace.reranked" />
    </section>
    <section v-if="trace.web_results?.length">
      <h4 class="mb-1 font-medium text-gray-700">网页搜索结果（{{ trace.web_results.length }} 条）</h4>
      <TraceDocumentList :documents="trace.web_results" />
    </section>
    <section v-if="trace.candidates?.length">
      <details>
        <summary class="cursor-pointer font-medium text-gray-700">重排前候选（{{ trace.candidates.length }} 条）</summary>
        <TraceDocumentList class="mt-1" :documents="trace.candidates" />
      </details>
    </section>
    <section v-if="trace.context?.length">
      <h4 class="mb-1 font-medium text-gray-700">送给模型的上下文</h4>
      <div v-for="span in trace.context" :key="span.id" :class="cited.has(span.id) ? 'text-teal-700' : ''">
        [来源 {{ span.id }}] <template v-if="span.url">网页 / {{ span.source }}</template><template v-else>{{ span.kb_name }} / {{ span.source }} {{ span.page }}</template>，{{ span.chars }} 字<span v-if="span.truncated" class="text-amber-600">（已截断）</span><span v-if="cited.has(span.id)">，已引用</span>
      </div>
    </section>
  </div>
</template>
<script setup>
import { computed } from 'vue';

import ChatStageTimeline from './ChatStageTimeline.vue';
import TraceDocumentList from './TraceDocumentList.vue';

const props = defineProps({ trace: { type: Object, required: true } });
const cited = computed(() => new Set(props.trace.cited_ids || []));
</script>
