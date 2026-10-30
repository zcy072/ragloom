<template>
  <ol class="space-y-0.5 text-xs text-gray-600">
    <li v-for="(stage, index) in stages" :key="`${stage.stage}-${index}`" class="flex flex-wrap gap-x-2">
      <span :class="statusClass(stage.status)" class="w-4 text-center" :title="stageStatusLabel(stage.status)">{{ statusIcon(stage.status) }}</span>
      <span class="font-medium text-gray-700">{{ stageLabel(stage.stage) }}</span>
      <span class="tabular-nums text-gray-400">{{ formatElapsed(stage.elapsed_ms) }}</span>
      <span v-if="stage.status !== 'ok'" :class="statusClass(stage.status)">{{ stageStatusLabel(stage.status) }}</span>
      <span class="min-w-0 basis-full pl-6 text-gray-500 sm:basis-auto sm:pl-0">{{ stageDetailText(stage.detail) }}</span>
    </li>
  </ol>
</template>
<script setup>
import { formatElapsed, stageDetailText, stageLabel, stageStatusLabel } from '../chatTrace.js';

defineProps({ stages: { type: Array, required: true } });
const statusIcon = status => ({ ok: '✓', skipped: '–', degraded: '!', error: '✗', cancelled: '–' })[status] || '·';
const statusClass = status => ({
  degraded: 'text-amber-600',
  error: 'text-red-600',
  skipped: 'text-gray-400',
  cancelled: 'text-gray-400'
})[status] || 'text-teal-700';
</script>
