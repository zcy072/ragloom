import { expect, test } from '@playwright/test';

import { collectPageErrors, installApiMocks } from './fixtures.js';

test('administrator login submits credentials and persists the session', async ({ page }) => {
  const errors = collectPageErrors(page);
  let loginPayload = null;
  await installApiMocks(page, { onAdminLogin: payload => { loginPayload = payload; } });
  await page.goto('/');

  await page.getByPlaceholder('输入账号名称', { exact: true }).fill('admin');
  await page.locator('input[type="password"]').fill('admin888');
  await page.getByRole('button', { name: '登录系统' }).click();

  await expect(page.getByRole('heading', { name: '问答对话' })).toBeVisible();
  expect(loginPayload).toMatchObject({ username: 'admin' });
  expect(loginPayload.password).toBe('admin888');
  const session = await page.evaluate(() => JSON.parse(localStorage.getItem('pymodel.auth')));
  expect(session).toMatchObject({ token: 'e2e-token', user: { username: 'e2e-admin', role: '超级管理员' } });
  expect(errors).toEqual([]);
});
