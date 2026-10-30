import { expect, test } from '@playwright/test';

import { prepareAuthenticatedPage } from './fixtures.js';

test('ops monitor shows dependency status, run statistics, filtered runs, failed jobs and log tail', async ({ page }) => {
  const errors = await prepareAuthenticatedPage(page);
  // 问答页的初始化请求未结束时 Suspense 仍停留在该页面，等它就绪后再切换菜单。
  await expect(page.getByText('历史回答')).toBeVisible();
  await page.waitForLoadState('networkidle');
  await page.getByRole('button', { name: '运行监控' }).click();

  await expect(page.getByRole('heading', { name: '运行监控' })).toBeVisible();
  // 开发服务器首次编译该页面较慢，Suspense 在组件就绪前保持新页面隐藏。
  await expect(page.getByText('33.3%')).toBeVisible({ timeout: 20000 });
  await expect(page.getByText('SQLite 正常')).toBeVisible();
  await expect(page.getByText('Milvus 异常')).toBeVisible();
  await expect(page.getByText('MilvusException: 连接超时')).toBeVisible();
  await expect(page.getByText('运行记录问题 2')).toBeVisible();
  await expect(page.getByText('RuntimeError: upstream down')).toBeVisible();
  await expect(page.getByText('坏文件.pdf')).toBeVisible();
  await expect(page.getByText('模拟日志行')).toBeVisible();

  await page.getByLabel('状态过滤').selectOption('error');
  await expect(page.getByText('运行记录问题 2')).toHaveCount(0);
  await expect(page.getByText('运行记录问题 3')).toBeVisible();
  expect(errors).toEqual([]);
});
