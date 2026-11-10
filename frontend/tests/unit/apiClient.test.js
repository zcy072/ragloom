import assert from 'node:assert/strict';
import test from 'node:test';

import { ApiError, apiDelete, apiGet, apiPost, apiPut, apiRequest, authorizedFetch } from '../../src/apiClient.js';

const ORIGIN = 'http://ragloom.test';

/** 安装假的 location 与 fetch，记录每次请求和重载次数。 */
const installBrowser = (respond = () => ({ status: 200, body: '{}' })) => {
  const original = { fetch: globalThis.fetch, location: globalThis.location, localStorage: globalThis.localStorage };
  const calls = [];
  const state = { reloads: 0, calls };
  const storage = new Map();
  globalThis.localStorage = {
    getItem: key => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: key => storage.delete(key)
  };
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    writable: true,
    value: { origin: ORIGIN, reload: () => { state.reloads += 1; } }
  });
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    const { status, body = '' } = respond(url, options);
    return {
      ok: status >= 200 && status < 300,
      status,
      statusText: '',
      async text() { return body; }
    };
  };
  state.storage = storage;
  state.restore = () => {
    globalThis.fetch = original.fetch;
    globalThis.localStorage = original.localStorage;
    Object.defineProperty(globalThis, 'location', { configurable: true, writable: true, value: original.location });
  };
  return state;
};

/** 每个用例加载独立的模块实例，避免“至多重载一次”的内存标记互相影响。 */
const freshClient = tag => import(`../../src/apiClient.js?case=${tag}`);

test('requests use the session cookie and never send or store a bearer token', async () => {
  const browser = installBrowser();
  try {
    await apiGet('/api/kbs');
    const { options } = browser.calls[0];
    assert.equal(options.credentials, 'same-origin');
    assert.equal(options.headers.get('Authorization'), null);
    assert.equal(browser.storage.size, 0);
  } finally {
    browser.restore();
  }
});

test('write requests carry the fixed request header; safe methods do not', async () => {
  const browser = installBrowser();
  try {
    await apiGet('/api/kbs');
    await apiPost('/api/kbs', { name: 'a' });
    await apiPut('/api/kbs/1', { name: 'b' });
    await apiDelete('/api/kbs/1');
    const form = new FormData();
    form.append('file', new Blob(['x']), 'a.txt');
    await apiRequest('/api/documents/kb', { method: 'POST', body: form });
    await authorizedFetch('/api/chat', { method: 'post', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    await authorizedFetch('/api/kbs', { method: 'HEAD' });

    const header = index => browser.calls[index].options.headers.get('X-RAGLoom-Request');
    assert.equal(header(0), null);
    assert.equal(header(1), '1');
    assert.equal(header(2), '1');
    assert.equal(header(3), '1');
    assert.equal(header(4), '1');
    // FormData 不能手工设置 Content-Type，否则会丢失 multipart 边界。
    assert.equal(browser.calls[4].options.headers.get('Content-Type'), null);
    assert.equal(browser.calls[4].options.body, form);
    assert.equal(header(5), '1');
    assert.equal(browser.calls[5].options.headers.get('Content-Type'), 'application/json');
    assert.equal(header(6), null);
  } finally {
    browser.restore();
  }
});

test('only same-origin /api paths are allowed', async () => {
  const browser = installBrowser();
  try {
    for (const url of ['https://evil.example/api/kbs', '//evil.example/api/kbs', '/static/a.js', '/apix/kbs']) {
      await assert.rejects(authorizedFetch(url), TypeError, url);
    }
    assert.equal(browser.calls.length, 0);
    await authorizedFetch(`${ORIGIN}/api/kbs`);
    await authorizedFetch('/api/ops/chat-traces/export?rating=down');
    assert.equal(browser.calls.length, 2);
  } finally {
    browser.restore();
  }
});

test('protected 401 responses reload the page once without retrying', async () => {
  const client = await freshClient('reload-once');
  const browser = installBrowser(() => ({ status: 401, body: JSON.stringify({ detail: '登录状态已失效' }) }));
  try {
    await assert.rejects(
      client.apiGet('/api/kbs'),
      error => error instanceof client.ApiError && error.status === 401 && error.message === '登录状态已失效'
    );
    await assert.rejects(client.apiPost('/api/chat/sessions', {}), error => error.status === 401);
    const response = await client.authorizedFetch('/api/documents/kb/doc');
    assert.equal(response.status, 401);
    assert.equal(browser.reloads, 1);
    assert.equal(browser.calls.length, 3);
  } finally {
    browser.restore();
  }
});

test('login and startup identity 401 responses do not reload', async () => {
  const client = await freshClient('no-reload');
  const browser = installBrowser(() => ({ status: 401, body: JSON.stringify({ detail: '用户名或密码错误' }) }));
  try {
    await assert.rejects(client.apiPost('/api/login', { username: 'a', password: 'b' }), error => error.status === 401);
    await assert.rejects(client.apiGet('/api/me', { authReload: false }), error => error.status === 401);
    assert.equal(browser.reloads, 0);
    // authReload 选项只在前端使用，不传给 fetch。
    assert.equal('authReload' in browser.calls[1].options, false);
  } finally {
    browser.restore();
  }
});

test('403 and 5xx responses keep the page', async () => {
  const client = await freshClient('keep-page');
  let status = 403;
  const browser = installBrowser(() => ({ status, body: JSON.stringify({ detail: '无权访问' }) }));
  try {
    await assert.rejects(client.apiGet('/api/users'), error => error.status === 403);
    status = 503;
    await assert.rejects(client.apiGet('/api/users'), error => error.status === 503);
    assert.equal(browser.reloads, 0);
  } finally {
    browser.restore();
  }
});

test('a FastAPI validation error is shown as readable Chinese instead of [object Object]', async () => {
  const browser = installBrowser(() => ({
    status: 422,
    body: JSON.stringify({
      detail: [
        { type: 'string_too_long', loc: ['body', 'name'], msg: 'String should have at most 80 characters', ctx: { max_length: 80 } },
        { type: 'string_too_short', loc: ['body', 'username'], msg: 'String should have at least 1 character', ctx: { min_length: 1 } },
        { type: 'missing', loc: ['body', 'unknown_field'], msg: 'Field required' },
        { type: 'value_error', loc: ['body', 'name'], msg: 'Value error, 名称不合法' }
      ]
    })
  }));

  try {
    await assert.rejects(
      apiGet('/api/kbs'),
      error => error instanceof ApiError
        && error.status === 422
        && error.message === '名称不能超过 80 个字符；账号名称不能为空；输入不能为空；Value error, 名称不合法'
    );
  } finally {
    browser.restore();
  }
});
