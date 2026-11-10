import { expect, test } from '@playwright/test';

import { COMMON_KB_ID, DEV_KB_ID, DOCUMENT_ID, prepareAuthenticatedPage } from './fixtures.js';

const SECOND_ID = '44444444-4444-4444-8444-444444444444';

test('leaving pending chat initialization cannot restore an old session after a library jump', async ({ page }) => {
  const sent = [];
  await prepareAuthenticatedPage(page, { onChatRequest: request => sent.push(request) });
  await expect(page.getByText('历史回答')).toBeVisible();
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  let calls = 0;
  let oldResponseSent = false;
  await page.route('**/api/chat/sessions', async route => {
    calls += 1;
    const old = calls === 1;
    if (old) await gate;
    await route.fulfill({ json: [
      { id: 'session-1', kb_id: COMMON_KB_ID, kb_name: '通用知识库', title: '旧会话' },
      { id: 'session-2', kb_id: DEV_KB_ID, kb_name: '研发知识库', title: '目标会话' },
    ] });
    if (old) oldResponseSent = true;
  });
  await page.reload();
  await expect.poll(() => calls).toBe(1);
  await page.getByRole('button', { name: '知识库', exact: true }).click();
  const target = page.getByText('研发知识库', { exact: true }).locator('xpath=..');
  await target.getByRole('button', { name: '开始问答' }).click();
  await expect(page.locator('select').filter({ has: page.locator(`option[value="${DEV_KB_ID}"]`) })).toHaveValue(DEV_KB_ID);
  await expect.poll(() => calls).toBe(2);
  const input = page.getByPlaceholder('输入问题，Enter 发送，Shift+Enter 换行');
  await expect(input).toBeEnabled();
  release();
  await expect.poll(() => oldResponseSent).toBe(true);
  await input.fill('验证目标知识库');
  await page.getByRole('button', { name: '发送', exact: true }).click();
  await expect.poll(() => sent.length).toBe(1);
  expect(sent[0].session_id).toBe('session-2');
  expect(sent[0].kb_id).toBe(DEV_KB_ID);
  await expect(page.locator('select').filter({ has: page.locator(`option[value="${DEV_KB_ID}"]`) })).toHaveValue(DEV_KB_ID);
});

test('same-name citation cards open their own document UUID', async ({ page }) => {
  await prepareAuthenticatedPage(page);
  const previewed = [];
  await page.route('**/api/chat/sessions/session-1', route => route.fulfill({ json: [{
    role: 'assistant', content: '第一份[来源 1]，第二份[来源 2]。',
    sources: [DOCUMENT_ID, SECOND_ID].map((document_id, index) => ({
      id: index + 1, citation_ids: [index + 1], document_id, kb_id: COMMON_KB_ID,
      source: '同名.txt', page: '', content: `第 ${index + 1} 份证据`,
    })),
  }] }));
  await page.route(`**/api/documents/${COMMON_KB_ID}/*?preview=true`, route => {
    previewed.push(new URL(route.request().url()).pathname.split('/')[4]);
    return route.fulfill({ contentType: 'text/plain', body: '文档预览' });
  });
  await page.reload();
  await expect(page.getByText('[1] 同名.txt', { exact: true })).toBeVisible();
  await expect(page.getByText('[2] 同名.txt', { exact: true })).toBeVisible();
  for (const citation of [2, 1]) {
    await page.getByText(`[${citation}] 同名.txt`, { exact: true }).click();
    await expect(page.getByTitle('document-preview')).toBeVisible();
    await page.getByTitle('关闭', { exact: true }).click();
  }
  expect(previewed).toEqual([SECOND_ID, DOCUMENT_ID]);
});

test('same-name documents preview, reindex and delete independently', async ({ page }) => {
  await prepareAuthenticatedPage(page);
  let documents = [DOCUMENT_ID, SECOND_ID].map(document_id => ({
    document_id, filename: '同名.txt', size: 100, uploaded_at: '2026-11-09T02:00:00+00:00',
    status: 'ready', parser_mode: 'native_text', tags: [],
  }));
  const reindexed = [];
  const previewed = [];
  await page.route(new RegExp(`/api/documents/${COMMON_KB_ID}(?:/|$)`), async route => {
    const request = route.request();
    const url = new URL(request.url());
    const documentId = url.pathname.split('/')[4];
    if (request.method() === 'DELETE') {
      documents = documents.filter(doc => doc.document_id !== documentId);
      return route.fulfill({ json: { status: 'success' } });
    }
    if (request.method() === 'POST') {
      reindexed.push(documentId);
      return route.fulfill({ json: { job_id: 'reindex-job' } });
    }
    if (documentId) {
      previewed.push(documentId);
      return route.fulfill({ contentType: 'text/plain', body: documentId === SECOND_ID ? '第二份文档' : '第一份文档' });
    }
    return route.fulfill({ json: documents });
  });
  await page.getByRole('button', { name: '知识库', exact: true }).click();
  await page.getByRole('button', { name: '管理文档' }).first().click();
  const first = page.locator(`[data-document-id="${DOCUMENT_ID}"]`);
  const second = page.locator(`[data-document-id="${SECOND_ID}"]`);
  await expect(first.getByText('同名.txt', { exact: true })).toBeVisible();
  await expect(second.getByText('同名.txt', { exact: true })).toBeVisible();
  await second.getByText('同名.txt', { exact: true }).click();
  await expect(page.getByTitle('document-preview')).toBeVisible();
  expect(previewed).toEqual([SECOND_ID]);
  await page.getByTitle('关闭', { exact: true }).last().click();
  await second.getByRole('button', { name: '重新索引', exact: true }).click();
  await expect.poll(() => reindexed).toEqual([SECOND_ID]);
  await first.getByRole('button', { name: '删除', exact: true }).click();
  await first.getByRole('button', { name: '删除', exact: true }).click();
  await expect(first).toHaveCount(0);
  await expect(second.getByText('同名.txt', { exact: true })).toBeVisible();
});

test('document filter sends the selected UUID for duplicate filenames', async ({ page }) => {
  const requests = [];
  await prepareAuthenticatedPage(page, { onChatRequest: request => requests.push(request) });
  await page.route('**/document_filters', route => route.fulfill({ json: {
    documents: [DOCUMENT_ID, SECOND_ID].map(document_id => ({ document_id, filename: '同名.txt' })),
    document_types: [], tags: [],
  } }));
  await page.reload();
  await expect(page.getByText('历史回答')).toBeVisible();
  await page.getByTitle('检索筛选').click();
  await page.locator('select').filter({ has: page.locator(`option[value="${SECOND_ID}"]`) }).selectOption(SECOND_ID);
  await expect(page.getByText('所选文档仅限定当前知识库')).toBeVisible();
  await page.getByPlaceholder('输入问题，Enter 发送，Shift+Enter 换行').fill('审批规则');
  await page.getByRole('button', { name: '发送', exact: true }).click();
  await expect.poll(() => requests.length).toBe(1);
  expect(requests[0].filters.document_ids).toEqual([SECOND_ID]);
  expect(requests[0].filters.source).toBeUndefined();
});

test('failed library deletion remains retryable after page reload', async ({ page }) => {
  await prepareAuthenticatedPage(page);
  let deleting = false;
  let removed = false;
  await page.route('**/api/kbs', route => route.fulfill({ json: removed ? [] : [{
    kb_id: DEV_KB_ID, name: '待删除知识库', state: deleting ? 'deleting' : 'active', docs: 0,
  }] }));
  await page.route(`**/api/kbs/${DEV_KB_ID}`, route => {
    if (!deleting) {
      deleting = true;
      return route.fulfill({ status: 503, json: { detail: '资源清理失败，请重试' } });
    }
    removed = true;
    return route.fulfill({ json: { status: 'success' } });
  });
  await page.getByRole('button', { name: '知识库', exact: true }).click();
  await page.getByTitle('更多操作').click();
  await page.getByRole('button', { name: '删除', exact: true }).click();
  await page.getByRole('button', { name: '确定删除', exact: true }).click();
  await expect(page.getByRole('button', { name: '重试删除' })).toBeVisible();
  await page.reload();
  await page.getByRole('button', { name: '知识库', exact: true }).click();
  await expect(page.getByRole('button', { name: '管理文档' })).toHaveCount(0);
  await page.getByRole('button', { name: '重试删除' }).click();
  await expect(page.getByText('待删除知识库', { exact: true })).toHaveCount(0);
});
