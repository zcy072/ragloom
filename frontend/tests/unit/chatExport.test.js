import assert from 'node:assert/strict';
import test from 'node:test';
import { sessionMarkdown, exportFilename } from '../../src/chatExport.js';

test('export preserves raw Markdown and scopes repeated citations to each answer', () => {
  const content = '**中文**\n\n```js\nconst x = 1;\n```\n\n| A | B |\n| --- | --- |\n| 1 | 2 |';
  const messages = ['甲库', '乙库'].map(kb_name => ({
    role: 'assistant', content, sources: [{ id: 1, citation_ids: [1], kb_name, source: '手册.pdf', page: '第 2 页', content: '证据' }]
  }));
  const output = sessionMarkdown(messages, '标题', '甲库');
  assert.equal(output.split(content).length - 1, 2);
  assert.equal(output.split('### 本轮引用来源').length - 1, 2);
  assert.ok(output.includes('甲库 / 手册.pdf'));
  assert.ok(output.includes('乙库 / 手册.pdf'));
});

test('empty sessions and attachments export without private URLs', () => {
  assert.equal(sessionMarkdown([], '标题', '库'), '');
  const output = sessionMarkdown([{ role: 'user', content: '问题', imageUrl: 'blob:private' }], '标题', '库');
  assert.ok(output.includes('图片附件未包含'));
  assert.ok(!output.includes('blob:private'));
});

test('download filename removes path separators and reserved characters', () => {
  assert.equal(exportFilename('a/b:c?'), 'a_b_c_.md');
  assert.equal(exportFilename('...'), '会话.md');
});

test('grouped evidence exports each citation and its own page', () => {
  const output = sessionMarkdown([{ role: 'assistant', content: '[来源 1] [来源 2]', sources: [{
    source: '手册.pdf', evidence: [
      { id: 1, source: '手册.pdf', page: '第 2 页', content: '第一条' },
      { id: 2, source: '手册.pdf', page: '第 8 页', content: '第二条' }
    ]
  }] }], '会话', '库');
  assert.ok(output.includes('[来源 1] 手册.pdf / 第 2 页'));
  assert.ok(output.includes('[来源 2] 手册.pdf / 第 8 页'));
  assert.ok(output.includes('> 第二条'));
});

test('web sources export with their address', () => {
  const output = sessionMarkdown([{ role: 'assistant', content: '[来源 1]', sources: [{
    source: '政策解读', evidence: [{ id: 1, source: '政策解读', page: '政府网', content: '摘要', url: 'https://example.com/a b' }]
  }] }], '会话', '库');
  assert.ok(output.includes('[来源 1] 政策解读 / 政府网 <https://example.com/a%20b>'));
});
