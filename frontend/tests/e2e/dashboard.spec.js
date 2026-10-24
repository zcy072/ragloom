import { expect, test } from '@playwright/test';

import { prepareAuthenticatedPage } from './fixtures.js';

test('dashboard renders summary data and an interactive trend chart', async ({ page }) => {
  const errors = await prepareAuthenticatedPage(page);
  await page.getByRole('button', { name: '仪表盘统计' }).click();

  await expect(page.getByRole('heading', { name: '知识库运行概况' })).toBeVisible();
  await expect(page.getByText('12.5 MB').first()).toBeVisible();
  await expect(page.getByText('80%')).toBeVisible();
  await expect(page.locator('svg path[stroke="#3b82f6"]')).toHaveCount(1);

  const dayTarget = page.locator('button[aria-label*="07-22"]').first();
  await dayTarget.focus();
  await expect(page.getByText('07-22').last()).toBeVisible();
  await expect(page.getByText('会话数').last()).toBeVisible();
  await expect(page.getByText('回答数').last()).toBeVisible();

  const resources = await page.evaluate(() => performance.getEntriesByType('resource').map(item => item.name));
  expect(resources.some(url => /echarts/i.test(url))).toBe(false);
  expect(errors).toEqual([]);
});

test('mobile navigation keeps dashboard and knowledge-base content contained', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  const errors = await prepareAuthenticatedPage(page);

  await page.getByTitle('打开导航').click();
  await page.getByRole('button', { name: '仪表盘统计' }).click();
  await expect(page.getByRole('heading', { name: '知识库运行概况' })).toBeVisible();
  const chart = page.locator('[role="group"][aria-label^="最近30天问答趋势"]');
  await chart.scrollIntoViewIfNeeded();
  const chartBounds = await chart.boundingBox();
  expect(chartBounds.x).toBeGreaterThanOrEqual(0);
  expect(chartBounds.x + chartBounds.width).toBeLessThanOrEqual(390);

  await page.getByTitle('打开导航').click();
  await page.getByRole('button', { name: '知识库' }).click();
  await expect(page.getByRole('heading', { name: '知识库管理' })).toBeVisible();
  await page.getByRole('button', { name: '管理文档' }).first().click();
  await expect(page.getByText('制度.pdf')).toBeVisible();
  expect(errors).toEqual([]);
});
