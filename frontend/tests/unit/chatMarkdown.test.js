import assert from 'node:assert/strict';
import test from 'node:test';
import { renderAnswer } from '../../src/chatMarkdown.js';

test('renders Markdown but escapes HTML and rejects dangerous links', () => {
  const { html } = renderAnswer('**bold**\n\n<script>alert(1)</script>\n\n[x](javascript:alert(1))');
  assert.ok(html.includes('<strong>bold</strong>'));
  assert.ok(!html.includes('<script>'));
  assert.ok(!html.includes('href="javascript:'));
});

test('citations in code remain literal and unknown IDs are not buttons', () => {
  const sources = [{ evidence: [{ id: 1, source: 'a.pdf' }] }];
  const { html } = renderAnswer('[来源 1] [来源 99] ` [来源 1] `\n\n```text\n[来源 1]', sources);
  assert.equal((html.match(/data-citation=/g) || []).length, 1);
  assert.ok(html.includes('<pre><code'));
  assert.ok(html.includes('[来源 99]'));
});

test('tables and incomplete fences preserve content', () => {
  assert.ok(renderAnswer('| A | B |\n| --- | --- |\n| 1 | 2 |').html.includes('<table>'));
  assert.ok(renderAnswer('```js\nconst x = 1;').html.includes('const x = 1;'));
});

test('legacy grouped citations do not invent per-citation evidence', () => {
  const result = renderAnswer('[来源 1] [来源 2]', [{ id: 1, citation_ids: [1, 2], content: 'old preview' }]);
  assert.equal(result.citations.size, 0);
  assert.ok(!result.html.includes('data-citation'));
});

test('citations inside ordinary links remain link text', () => {
  const { html } = renderAnswer('[[来源 1]](https://example.com)', [{ evidence: [{ id: 1 }] }]);
  assert.ok(html.includes('<a href='));
  assert.ok(!html.includes('<button'));
});
