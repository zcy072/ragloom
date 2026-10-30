<template>
  <ol class="space-y-1">
    <li v-for="doc in documents" :key="doc.position" :class="doc.kept === false ? 'text-gray-400' : ''">
      <span class="tabular-nums">{{ doc.position }}. </span>
      <span v-if="doc.score !== undefined" class="tabular-nums font-medium">{{ doc.score.toFixed(3) }} </span>
      {{ doc.source }}<template v-if="doc.page"> 第 {{ doc.page }} 页</template> · {{ doc.route }}
      <a v-if="doc.url" :href="doc.url" target="_blank" rel="noopener noreferrer nofollow" class="text-teal-700 underline">{{ doc.url }}</a>
      <template v-if="doc.kept === false">{{ doc.url ? '（超出网页名额）' : '（低于下限）' }}</template>
      <div v-if="doc.snippet" class="pl-4 text-gray-500">{{ doc.snippet }}</div>
    </li>
  </ol>
</template>
<script setup>
defineProps({ documents: { type: Array, required: true } });
</script>
