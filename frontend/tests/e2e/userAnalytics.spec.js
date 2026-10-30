import { expect, test } from '@playwright/test';

import { prepareAuthenticatedPage } from './fixtures.js';

test('user analytics shows overview, sortable users, user detail and audited conversation view', async ({ page }) => {
  const errors = await prepareAuthenticatedPage(page);
  await expect(page.getByText('历史回答')).toBeVisible();
  await page.waitForLoadState('networkidle');
  await page.getByRole('button', { name: '用户分析' }).click();

  await expect(page.getByRole('heading', { name: '用户分析' })).toBeVisible();
  const overview = page.getByLabel('用户概览');
  await expect(overview.getByText('2 个待审核')).toBeVisible({ timeout: 20000 });
  await expect(overview.getByText('人均 7.5 次')).toBeVisible();
  await expect(overview.getByText('75%')).toBeVisible();
  await expect(page.getByLabel('知识库使用情况').getByText('研发知识库')).toBeVisible();
  await expect(page.getByRole('img', { name: /最近 30 天提问数/ }).locator('path')).toHaveCount(3);

  await expect(page.getByRole('cell', { name: /用户 01/ })).toBeVisible();
  await page.getByRole('button', { name: /^提问/ }).click();
  await expect(page.getByRole('cell', { name: /用户 23/ })).toBeVisible();
  await page.getByRole('button', { name: /^提问/ }).click();

  await page.getByRole('cell', { name: /用户 01/ }).click();
  const detail = page.getByRole('region', { name: '用户详情' });
  await expect(detail.getByText('年假可以分几次休')).toBeVisible();
  await expect(detail.getByText('用户名或密码错误')).toBeVisible();

  await expect(page.getByText('还没有查看记录')).toBeVisible();
  await detail.getByRole('button', { name: '查看对话' }).click();
  const dialog = page.getByRole('dialog', { name: '用户对话' });
  await expect(dialog.getByText('年假最多可分三次休完。')).toBeVisible();
  await expect(dialog.getByText('来源：员工手册.pdf')).toBeVisible();
  await dialog.getByRole('button', { name: '关闭' }).click();
  await expect(page.getByLabel('对话查看审计').getByText('通用知识库 / 年假怎么请')).toBeVisible();
  expect(errors).toEqual([]);
});
