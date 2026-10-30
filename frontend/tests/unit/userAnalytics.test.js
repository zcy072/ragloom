import assert from 'node:assert/strict';
import test from 'node:test';

import { chartLines, niceMax, percent, sparseLabelIndexes } from '../../src/userAnalytics.js';

test('niceMax rounds up to a multiple of four and never returns zero', () => {
  assert.equal(niceMax([]), 4);
  assert.equal(niceMax([0, 0]), 4);
  assert.equal(niceMax([5, 9]), 12);
  assert.equal(niceMax([16]), 16);
});

test('chartLines maps values onto a 0-100 viewBox with y growing downward', () => {
  const [line] = chartLines([{ key: 'q', values: [0, 4, 2] }], 4);
  assert.deepEqual(line.points, [{ x: 0, y: 100 }, { x: 50, y: 0 }, { x: 100, y: 50 }]);
  assert.equal(line.path, 'M 0 100 L 50 0 L 100 50');

  const [single] = chartLines([{ key: 'q', values: [2] }], 4);
  assert.deepEqual(single.points, [{ x: 50, y: 50 }]);
});

test('sparseLabelIndexes keeps both ends and spaces labels evenly', () => {
  assert.deepEqual(sparseLabelIndexes(3), [0, 1, 2]);
  assert.deepEqual(sparseLabelIndexes(30), [0, 6, 12, 17, 23, 29]);
});

test('percent avoids division by zero', () => {
  assert.equal(percent(1, 0), '0%');
  assert.equal(percent(3, 4), '75%');
});
