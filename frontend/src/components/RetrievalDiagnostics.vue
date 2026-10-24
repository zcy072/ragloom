<template>
  <details class="mt-3 border-t pt-2 text-xs text-gray-600 break-words">
    <summary class="cursor-pointer">检索诊断</summary>
    <dl class="mt-2 space-y-2">
      <div><dt>原始问题</dt><dd>{{ diagnostics.original_question }}</dd></div>
      <div><dt>实际检索问题</dt><dd>{{ diagnostics.skipped ? "跳过检索" : diagnostics.actual_query }}</dd></div>
      <div v-if="diagnostics.variant_query"><dt>书面化补充查询</dt><dd>{{ diagnostics.variant_query }}</dd></div>
      <div>问题改写：{{ diagnostics.rewritten ? "是" : "否" }}；原问题回退：{{ diagnostics.fallback_attempted ? "已尝试" : "未尝试" }}</div>
      <div>重排：{{ statusLabels[diagnostics.rerank_status] || "未知" }}</div>
      <div v-for="(count, key) in diagnostics.counts" :key="key">{{ countLabels[key] || key }}：{{ count }}</div>
      <div v-for="(duration, key) in diagnostics.elapsed_ms" :key="key">{{ timeLabels[key] || key }}：{{ duration }} ms</div>
    </dl>
    <div v-if="diagnostics.ranking?.length" class="mt-3">
      <div class="font-medium">排序依据：模型分数优先，完整编号仅在同分时优先</div>
      <div v-for="item in diagnostics.ranking" :key="item.position">候选 {{ item.position }}：模型分数 {{ item.model_score }}；编号命中 {{ item.exact_identifiers.join(', ') || '无' }}</div>
    </div>
    <div v-if="diagnostics.context_sources?.length" class="mt-3">
      <div class="font-medium">送给模型的证据</div>
      <div v-for="source in diagnostics.context_sources" :key="source.id">[来源 {{ source.id }}] {{ source.kb_name }} / {{ source.source }} {{ source.page }} {{ source.routes?.join(", ") }}</div>
    </div>
  </details>
</template>
<script setup>
defineProps({ diagnostics: { type: Object, required: true } });
const statusLabels = { success: "成功", degraded: "降级为候选顺序", not_run: "未执行" };
const countLabels = { deduplicated_candidates: "去重后候选块", rerank_input: "送入重排候选块", rerank_returned: "重排返回候选块", variant_candidates: "来自补充查询的候选块", context_spans: "最终上下文段" };
const timeLabels = { query_variant: "补充查询生成耗时", retrieval: "检索耗时", rerank: "重排耗时", context: "上下文构建耗时" };
</script>
