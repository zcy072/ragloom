import assert from 'node:assert/strict';
import test from 'node:test';

import { ApiError, apiGet } from '../../src/apiClient.js';
import { loadAuthSession, saveAuthSession } from '../../src/authSession.js';

test('a 401 response clears the session and emits auth-expired', async () => {
  const values = new Map();
  const originalLocalStorage = globalThis.localStorage;
  const originalWindow = globalThis.window;
  const originalFetch = globalThis.fetch;
  const originalCustomEvent = globalThis.CustomEvent;
  globalThis.localStorage = {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: key => values.delete(key)
  };
  globalThis.window = new EventTarget();
  globalThis.CustomEvent = class CustomEvent extends Event {};
  let expiredEvents = 0;
  window.addEventListener('auth-expired', () => { expiredEvents += 1; });
  globalThis.fetch = async (_url, options) => {
    assert.equal(options.headers.get('Authorization'), 'Bearer expired-token');
    return {
      ok: false,
      status: 401,
      statusText: 'Unauthorized',
      async text() { return JSON.stringify({ detail: '登录凭证已过期' }); }
    };
  };

  try {
    saveAuthSession('expired-token', { username: 'admin', role: 'admin' });
    await assert.rejects(
      apiGet('/api/kbs'),
      error => error instanceof ApiError
        && error.status === 401
        && error.message === '登录凭证已过期'
    );
    assert.equal(loadAuthSession(), null);
    assert.equal(expiredEvents, 1);
  } finally {
    globalThis.localStorage = originalLocalStorage;
    globalThis.window = originalWindow;
    globalThis.fetch = originalFetch;
    globalThis.CustomEvent = originalCustomEvent;
  }
});

test('a FastAPI validation error shows its messages instead of [object Object]', async () => {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async () => ({
    ok: false,
    status: 422,
    statusText: 'Unprocessable Entity',
    async text() {
      return JSON.stringify({
        detail: [
          { type: 'string_too_long', loc: ['body', 'name'], msg: 'String should have at most 80 characters' },
          { type: 'missing', loc: ['body', 'tags'], msg: 'Field required' }
        ]
      });
    }
  });

  try {
    await assert.rejects(
      apiGet('/api/kbs'),
      error => error instanceof ApiError
        && error.status === 422
        && error.message === 'String should have at most 80 characters；Field required'
    );
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test('a stale 401 does not clear a session that another tab already replaced', async () => {
  const values = new Map();
  const originalLocalStorage = globalThis.localStorage;
  const originalWindow = globalThis.window;
  const originalFetch = globalThis.fetch;
  const originalCustomEvent = globalThis.CustomEvent;
  globalThis.localStorage = {
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: key => values.delete(key)
  };
  globalThis.window = new EventTarget();
  globalThis.CustomEvent = class CustomEvent extends Event {};
  let expiredEvents = 0;
  window.addEventListener('auth-expired', () => { expiredEvents += 1; });
  globalThis.fetch = async () => {
    // 请求在途中，另一个标签页写入了新的登录会话。
    saveAuthSession('new-token', { username: 'other' });
    return {
      ok: false,
      status: 401,
      statusText: 'Unauthorized',
      async text() { return JSON.stringify({ detail: '登录状态已失效' }); }
    };
  };

  try {
    saveAuthSession('old-token', { username: 'admin' });
    await assert.rejects(apiGet('/api/kbs'), error => error instanceof ApiError);
    assert.equal(loadAuthSession()?.token, 'new-token');
    assert.equal(expiredEvents, 0);
  } finally {
    globalThis.localStorage = originalLocalStorage;
    globalThis.window = originalWindow;
    globalThis.fetch = originalFetch;
    globalThis.CustomEvent = originalCustomEvent;
  }
});
