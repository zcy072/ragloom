import { expect, test } from '@playwright/test';

import { prepareAuthenticatedPage } from './fixtures.js';

test('initialization blocks sending until history and models are ready', async ({ page }) => {
  await prepareAuthenticatedPage(page);
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  await page.route('**/api/models', async route => {
    await gate;
    await route.fulfill({ json: [{ provider: 'mock-model', configured: true, is_active: 1 }] });
  });
  await page.reload();
  await expect(page.getByText('历史回答')).toBeVisible();
  const input = page.getByPlaceholder('输入问题，Enter 发送，Shift+Enter 换行');
  await expect(input).toBeDisabled();
  release();
  await input.fill('模拟异常断流');
  await page.getByRole('button', { name: '发送', exact: true }).click();
  await expect(page.getByText('⚠️ 回答连接意外中断，请重试。')).toBeVisible();
});

test('copy preserves original text and reports clipboard failures', async ({ page }) => {
  await prepareAuthenticatedPage(page);
  await expect(page.getByText('历史回答')).toBeVisible();
  await page.evaluate(() => {
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: {
      writeText: async text => { window.copiedAnswer = text; }
    } });
  });
  await page.getByRole('button', { name: '复制回答' }).click();
  await expect(page.getByText('回答已复制')).toBeVisible();
  expect(await page.evaluate(() => window.copiedAnswer)).toBe('历史回答');
  await page.evaluate(() => {
    navigator.clipboard.writeText = async () => { throw new Error('denied'); };
  });
  await page.getByRole('button', { name: '复制回答' }).click();
  await expect(page.getByText('复制失败，请检查浏览器剪贴板权限')).toBeVisible();
  await page.getByTitle('新对话').click();
  await expect(page.getByRole('button', { name: '导出当前会话 Markdown' })).toBeDisabled();
  await page.getByText('已有会话', { exact: true }).click();
  await expect(page.getByText('历史回答')).toBeVisible();
  await expect(page.getByRole('button', { name: '导出当前会话 Markdown' })).toBeEnabled();
});

test('stopping a partial answer enables copying and export without diagnostics', async ({ page }) => {
  await prepareAuthenticatedPage(page);
  await expect(page.getByText('历史回答')).toBeVisible();
  await page.evaluate(() => {
    const originalFetch = window.fetch;
    window.fetch = (url, options) => {
      if (url !== '/api/chat') return originalFetch(url, options);
      return Promise.resolve(new Response(new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('data: {"chunk":"尚未完成的回答"}\n\n'));
          options.signal.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')));
        }
      }), { headers: { 'Content-Type': 'text/event-stream' } }));
    };
  });
  await page.getByPlaceholder('输入问题，Enter 发送，Shift+Enter 换行').fill('等待停止');
  await page.getByRole('button', { name: '发送', exact: true }).click();
  await expect(page.getByText('尚未完成的回答')).toBeVisible();
  await expect(page.getByRole('button', { name: '复制回答' }).last()).toBeDisabled();
  await expect(page.getByRole('button', { name: '导出当前会话 Markdown' })).toBeDisabled();
  await page.getByTitle('停止生成').click();
  await expect(page.getByRole('button', { name: '复制回答' }).last()).toBeEnabled();
  await expect(page.getByRole('button', { name: '导出当前会话 Markdown' })).toBeEnabled();
  await expect(page.getByText('尚未完成的回答')).toBeVisible();
  await expect(page.getByText('检索诊断', { exact: true })).toHaveCount(0);
});

test('sessions can be searched and renamed from the sidebar', async ({ page }) => {
  const renames = [];
  await prepareAuthenticatedPage(page, { onSessionRename: (id, payload) => renames.push([id, payload]) });
  const search = page.getByPlaceholder('搜索标题或知识库');
  await expect(page.getByText('研发排期')).toBeVisible();

  await search.fill('研发');
  await expect(page.getByText('研发排期')).toBeVisible();
  await expect(page.getByText('已有会话')).toHaveCount(0);
  await search.fill('不存在的会话');
  await expect(page.getByText('没有匹配的对话')).toBeVisible();
  await search.fill('');

  const card = page.getByText('研发排期').locator('xpath=ancestor::div[contains(@class, "group")][1]');
  await card.hover();
  await card.getByRole('button', { name: '重命名对话' }).click();
  const titleInput = page.getByLabel('会话标题');
  await expect(titleInput).toBeFocused();
  await titleInput.fill('新的排期');
  await titleInput.press('Enter');
  await expect(page.getByText('新的排期')).toBeVisible();
  expect(renames).toEqual([['session-2', { title: '新的排期' }]]);

  // Esc 取消编辑：不发请求，标题保持不变。
  await page.getByText('新的排期').dblclick();
  await page.getByLabel('会话标题').fill('不会保存');
  await page.getByLabel('会话标题').press('Escape');
  await expect(page.getByText('新的排期')).toBeVisible();
  expect(renames).toHaveLength(1);
});

test('Markdown citations and session export preserve evidence', async ({ page }) => {
  await prepareAuthenticatedPage(page);
  await page.route('**/api/chat/sessions/session-1', route => route.fulfill({
    json: [{ role: 'assistant', content: '**结论** [来源 2]\n\n| 项目 | 值 |\n| --- | --- |\n| A | B |', sources: [{
      id: 2, citation_ids: [2], source: '制度.pdf', page: '第 8 页', content: '证据二',
      evidence: [{ id: 2, source: '制度.pdf', kb_name: '通用知识库', page: '第 8 页', page_numbers: [8], content: '证据二' }]
    }] }]
  }));
  await page.reload();
  await expect(page.locator('.chat-markdown strong')).toHaveText('结论');
  await expect(page.locator('.chat-markdown table')).toBeVisible();
  await page.getByRole('button', { name: '查看来源 2' }).click();
  await expect(page.getByTitle('document-preview')).toHaveAttribute('src', /#page=8$/);
  await page.getByTitle('关闭', { exact: true }).click();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: '导出当前会话 Markdown' }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe('已有会话.md');
});

test('desktop chat and knowledge-base workflows', async ({ page }) => {
  const errors = await prepareAuthenticatedPage(page);

  await expect(page.getByText('历史回答')).toBeVisible();
  await page.getByTitle('检索筛选').click();
  await page.locator('select').filter({ has: page.locator('option[value="制度"]') }).selectOption('制度');
  await page.getByPlaceholder('输入问题，Enter 发送，Shift+Enter 换行').fill('验证流式回答');
  await page.getByRole('button', { name: '发送' }).click();
  await expect(page.getByText('冒烟回答')).toBeVisible();
  await page.getByText('[1] 制度.pdf').click();
  const previewFrame = page.getByTitle('document-preview');
  await expect(previewFrame).toBeVisible();
  const previewBox = await previewFrame.locator('..').boundingBox();
  expect(previewBox.height).toBeGreaterThan(page.viewportSize().height * 0.9);
  await previewFrame.locator('..').getByTitle('关闭').click();

  await page.getByRole('button', { name: '知识库' }).click();
  await expect(page.getByRole('heading', { name: '知识库管理' })).toBeVisible();
  await page.getByRole('button', { name: '管理文档' }).first().click();
  await expect(page.getByText('管理文档 - 通用知识库')).toBeVisible();
  await expect(page.getByLabel('PDF / DOCX 解析方式')).toHaveValue('mineru_cloud');
  await expect(page.getByLabel('图片解析方式')).toHaveValue('vision_model');
  await expect(page.getByText('图片会发送给已配置的视觉模型解析，失败时改用本地 Paddle OCR。')).toBeVisible();
  await expect(page.getByText('制度.pdf')).toBeVisible();
  await expect(page.getByText('2026/7/23 16:00:00')).toBeVisible();
  await page.getByText('分段配置（当前：800 字符 / 重叠 150 字符）').click();
  await expect(page.getByText('恢复默认值')).toBeVisible();
  await page.getByText('制度.pdf').click();
  await expect(page.getByTitle('关闭').last()).toBeVisible();
  await page.getByTitle('关闭').last().click();

  expect(errors).toEqual([]);
});

test('chat reports an SSE connection that closes before its terminal event', async ({ page }) => {
  const errors = await prepareAuthenticatedPage(page);

  await page.getByPlaceholder('输入问题，Enter 发送，Shift+Enter 换行').fill('模拟异常断流');
  await page.getByRole('button', { name: '发送' }).click();

  await expect(page.getByText('⚠️ 回答连接意外中断，请重试。')).toBeVisible();
  expect(errors).toEqual([]);
});

test('explicit new chat does not send the previous session id', async ({ page }) => {
  const chatRequests = [];
  const errors = await prepareAuthenticatedPage(page, {
    onChatRequest: payload => chatRequests.push(payload)
  });

  await expect(page.getByText('历史回答')).toBeVisible();
  await page.getByTitle('新对话').click();
  await page.getByPlaceholder('输入问题，Enter 发送，Shift+Enter 换行').fill('全新的问题');
  await page.getByRole('button', { name: '发送' }).click();
  await expect(page.getByText('冒烟回答')).toBeVisible();

  expect(chatRequests).toHaveLength(1);
  expect(chatRequests[0].new_session).toBe(true);
  expect(chatRequests[0]).not.toHaveProperty('session_id');
  expect(errors).toEqual([]);
});

test('role deletion confirmation is not clipped by the role table', async ({ page }) => {
  const errors = await prepareAuthenticatedPage(page);

  await page.getByRole('button', { name: '角色管理' }).click();
  const roleRow = page.getByRole('row', { name: /研发管理员/ });
  await roleRow.getByRole('button', { name: '删除' }).click();

  const dialog = page.getByRole('dialog', { name: '删除角色' });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText('研发管理员');

  const dialogBox = await dialog.boundingBox();
  const viewport = page.viewportSize();
  expect(dialogBox.x).toBeGreaterThan(0);
  expect(dialogBox.y).toBeGreaterThan(0);
  expect(dialogBox.width).toBeLessThan(300);
  expect(dialogBox.x + dialogBox.width).toBeLessThan(viewport.width);
  expect(dialogBox.y + dialogBox.height).toBeLessThan(viewport.height);
  expect(errors).toEqual([]);
});

test('user and role management use server pagination', async ({ page }) => {
  const errors = await prepareAuthenticatedPage(page);

  await page.getByRole('button', { name: '用户管理' }).click();
  await expect(page.getByText('共 23 条')).toBeVisible();
  await page.getByTitle('下一页').click();
  await expect(page.getByText('用户 23')).toBeVisible();

  await page.getByRole('button', { name: '角色管理' }).click();
  await expect(page.getByText('共 25 条')).toBeVisible();
  await page.getByTitle('下一页').click();
  await expect(page.getByText('项目角色 21')).toBeVisible();
  expect(errors).toEqual([]);
});
