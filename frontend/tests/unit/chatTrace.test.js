import assert from 'node:assert/strict';
import test from 'node:test';

import { formatElapsed, stageDetailText, stageLabel } from '../../src/chatTrace.js';

test('stage details are labelled and empty values are skipped', () => {
  assert.equal(
    stageDetailText({ candidates: 3, fallback: false, top_score: 0.91234, query: '', error: null }),
    '候选数：3；原问题回退：否；最高分：0.912'
  );
  assert.equal(stageLabel('rerank'), '重排');
  assert.equal(stageLabel('unknown'), 'unknown');
});

test('elapsed time switches to seconds at one second', () => {
  assert.equal(formatElapsed(412.4), '412 ms');
  assert.equal(formatElapsed(1530), '1.5 s');
});
