import { expect, test } from '@playwright/test';

import { collectPageErrors, installApiMocks, prepareAuthenticatedPage } from './fixtures.js';

const WORKSPACE_HEADING = { name: '问答对话' };

/** 记录页面主框架的导航次数，用来断言是否发生了整页重载。 */
const countNavigations = page => {
  const counter = { value: 0 };
  page.on('framenavigated', frame => {
    if (frame === page.mainFrame()) counter.value += 1;
  });
  return counter;
};

/** 未登录时启动 /me 返回 401、退出失败返回 503 都是预期的，浏览器会为此打印资源加载错误。 */
const scriptErrors = errors => errors.filter(message => !message.startsWith('Failed to load resource'));

test('login uses the session cookie contract and reloads into the workspace', async ({ page }) => {
  const errors = collectPageErrors(page);
  let loginPayload = null;
  await page.addInitScript(() => localStorage.setItem('pymodel.auth', JSON.stringify({ token: 'legacy', user: {} })));
  const api = await installApiMocks(page, { onAdminLogin: payload => { loginPayload = payload; } });
  await page.goto('/');

  await page.getByPlaceholder('输入账号名称', { exact: true }).fill('admin');
  await page.locator('input[type="password"]').fill('admin888');
  await page.getByRole('button', { name: '登录系统' }).click();

  await expect(page.getByRole('heading', WORKSPACE_HEADING)).toBeVisible();
  expect(loginPayload).toEqual({ username: 'admin', password: 'admin888' });
  expect(api.writeRequests.find(item => item.path === '/api/login')).toMatchObject({ method: 'POST', header: '1' });
  const storage = await page.evaluate(() => ({ ...localStorage }));
  // 旧令牌被删除；只留下一个不含身份信息的随机标记。
  expect(Object.keys(storage)).toEqual(['ragloom.auth-event']);
  expect(storage['ragloom.auth-event']).not.toMatch(/admin|token|e2e/i);
  expect(scriptErrors(errors)).toEqual([]);
});

test('wrong credentials show an error without reloading', async ({ page }) => {
  await installApiMocks(page);
  await page.goto('/');
  const navigations = countNavigations(page);

  await page.getByPlaceholder('输入账号名称', { exact: true }).fill('admin');
  await page.locator('input[type="password"]').fill('wrong-password');
  await page.getByRole('button', { name: '登录系统' }).click();

  await expect(page.getByText('用户名或密码错误')).toBeVisible();
  expect(navigations.value).toBe(0);
});

test('password rules are checked before registration is submitted', async ({ page }) => {
  const api = await installApiMocks(page);
  await page.goto('/');
  await page.getByRole('button', { name: '申请权限' }).click();
  await page.getByPlaceholder('作为登录账号').fill('new-user');
  await page.getByPlaceholder('如：雷勇').fill('新用户');
  const passwords = page.locator('input[type="password"]');
  await passwords.nth(0).fill('密'.repeat(25));
  await passwords.nth(1).fill('密'.repeat(25));
  await page.getByRole('button', { name: '提交申请' }).click();
  await expect(page.getByText('密码不能超过 72 字节')).toBeVisible();
  await passwords.nth(0).fill(' 1234 ');
  await passwords.nth(1).fill(' 1234 ');
  await page.getByRole('button', { name: '提交申请' }).click();
  await expect(page.getByText('密码至少需要 8 个字符')).toBeVisible();
  expect(api.writeRequests).toEqual([]);
});

test('startup failures show a retry screen instead of the login page', async ({ page }) => {
  await installApiMocks(page, { authenticated: true });
  let failures = 1;
  await page.route('**/api/me', route => {
    if (failures-- > 0) return route.fulfill({ status: 503, json: { detail: '数据库不可用' } });
    return route.fallback();
  });
  await page.goto('/');

  await expect(page.getByText('服务暂时不可用（503），请稍后重试')).toBeVisible();
  await expect(page.getByRole('button', { name: '登录系统' })).toHaveCount(0);
  await page.getByRole('button', { name: '重试' }).click();
  await expect(page.getByRole('heading', WORKSPACE_HEADING)).toBeVisible();
});

test('logout calls the server, keeps the page on failure and reloads on success', async ({ page }) => {
  const errors = collectPageErrors(page);
  const api = await installApiMocks(page, { authenticated: true });
  await page.goto('/');
  // 等工作区首屏数据加载完再退出，避免在途请求在退出后收到 401。
  await expect(page.getByText('历史回答')).toBeVisible();

  api.logoutStatus = 503;
  await page.getByTitle('退出登录').click();
  await expect(page.getByText('退出未完成，请重试')).toBeVisible();
  await expect(page.getByRole('heading', WORKSPACE_HEADING)).toBeVisible();

  api.logoutStatus = 204;
  await page.getByTitle('退出登录').click();
  await expect(page.getByRole('button', { name: '登录系统' })).toBeVisible();
  expect(api.logouts).toBe(2);
  expect(api.writeRequests.filter(item => item.path === '/api/logout').every(item => item.header === '1')).toBe(true);
  expect(scriptErrors(errors)).toEqual([]);
});

test('password change submits raw passwords and returns to login', async ({ page }) => {
  const api = await installApiMocks(page, { authenticated: true });
  await page.goto('/');
  await page.getByTitle('修改密码').click();
  await page.getByPlaceholder('请输入当前密码').fill('admin888');
  await page.getByPlaceholder('至少 8 个字符，最多 72 字节').fill(' new pass ');
  await page.getByPlaceholder('请再次输入新密码').fill(' new pass ');
  const request = page.waitForRequest('**/api/change_password');
  await page.getByRole('button', { name: '确认修改' }).click();
  expect((await request).postDataJSON()).toEqual({ old_password: 'admin888', new_password: ' new pass ' });
  await expect(page.getByRole('button', { name: '登录系统' })).toBeVisible();
  expect(api.writeRequests.find(item => item.path === '/api/change_password')?.header).toBe('1');
});

test('a protected 401 reloads once and lands on the login page', async ({ page }) => {
  await prepareAuthenticatedPage(page);
  await page.route('**/api/dashboard/stats', route => route.fulfill({ status: 401, json: { detail: '登录状态已失效' } }));
  await page.route('**/api/me', route => route.fulfill({ status: 401, json: { detail: '未登录' } }));
  const navigations = countNavigations(page);
  await page.getByRole('button', { name: '仪表盘统计' }).click();

  await expect(page.getByRole('button', { name: '登录系统' })).toBeVisible();
  // 等待一段时间确认启动时 /me 的 401 没有再次触发重载。
  await page.waitForTimeout(500);
  expect(navigations.value).toBe(1);
});

test('another tab announcing an auth change reloads this tab', async ({ context, page }) => {
  const api = await installApiMocks(page, { authenticated: true });
  await page.goto('/');
  await expect(page.getByRole('heading', WORKSPACE_HEADING)).toBeVisible();

  const other = await context.newPage();
  await installApiMocks(other);
  await other.goto('/');
  await expect(other.getByRole('button', { name: '登录系统' })).toBeVisible();
  // 另一个标签页退出；本页的 storage 监听收到随机标记后重载并回到登录页。
  api.authenticated = false;
  await other.evaluate(() => localStorage.setItem('ragloom.auth-event', crypto.randomUUID()));
  await expect(page.getByRole('button', { name: '登录系统' })).toBeVisible();
});
