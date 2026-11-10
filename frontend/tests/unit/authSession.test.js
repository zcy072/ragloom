import assert from 'node:assert/strict';
import test from 'node:test';

import { AUTH_EVENT_CHANNEL, AUTH_EVENT_STORAGE_KEY, finishAuthChange, removeLegacyAuthSession } from '../../src/authSession.js';
import { newPasswordProblem, existingPasswordProblem } from '../../src/passwordPolicy.js';

/** 替换 localStorage 与 location，返回记录结果的对象。 */
const installBrowser = (storage) => {
  const original = { localStorage: globalThis.localStorage, location: globalThis.location };
  const state = { reloads: 0 };
  globalThis.localStorage = storage;
  Object.defineProperty(globalThis, 'location', {
    configurable: true,
    writable: true,
    value: { origin: 'http://ragloom.test', reload: () => { state.reloads += 1; } }
  });
  state.restore = () => {
    globalThis.localStorage = original.localStorage;
    Object.defineProperty(globalThis, 'location', { configurable: true, writable: true, value: original.location });
  };
  return state;
};

test('auth changes write a fresh marker without identity data and reload', () => {
  const values = new Map([['pymodel.auth', '{"token":"old"}']]);
  const browser = installBrowser({
    getItem: key => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: key => values.delete(key)
  });
  try {
    removeLegacyAuthSession();
    assert.equal(values.has('pymodel.auth'), false);
    finishAuthChange();
    const first = values.get(AUTH_EVENT_STORAGE_KEY);
    finishAuthChange();
    const second = values.get(AUTH_EVENT_STORAGE_KEY);
    assert.ok(first && second && first !== second);
    assert.doesNotMatch(second, /token|user|admin/i);
    assert.deepEqual([...values.keys()], [AUTH_EVENT_STORAGE_KEY]);
    assert.equal(browser.reloads, 2);
  } finally {
    browser.restore();
  }
});

test('unavailable localStorage does not break cleanup or auth changes', () => {
  const broken = () => { throw new Error('SecurityError'); };
  const browser = installBrowser({ getItem: broken, setItem: broken, removeItem: broken });
  try {
    assert.doesNotThrow(removeLegacyAuthSession);
    assert.doesNotThrow(finishAuthChange);
    assert.equal(browser.reloads, 1);
  } finally {
    browser.restore();
  }
});

test('auth changes are broadcast even when localStorage is unavailable', () => {
  const broken = () => { throw new Error('SecurityError'); };
  const browser = installBrowser({ getItem: broken, setItem: broken, removeItem: broken });
  const originalChannel = globalThis.BroadcastChannel;
  const posted = [];
  globalThis.BroadcastChannel = class {
    constructor(name) { this.name = name; this.closed = false; }
    postMessage(message) { posted.push([this.name, message]); }
    close() { this.closed = true; }
  };
  try {
    finishAuthChange();
    assert.deepEqual(posted, [[AUTH_EVENT_CHANNEL, 'changed']]);
    assert.equal(browser.reloads, 1);
  } finally {
    globalThis.BroadcastChannel = originalChannel;
    browser.restore();
  }
});

test('password rules count characters and UTF-8 bytes without trimming', () => {
  assert.match(newPasswordProblem('1234567'), /至少需要 8 个字符/);
  assert.equal(newPasswordProblem('12345678'), '');
  // 首尾空格属于密码本身，参与长度计算。
  assert.equal(newPasswordProblem(' 123456 '), '');
  assert.equal(newPasswordProblem('a'.repeat(72)), '');
  assert.match(newPasswordProblem('a'.repeat(73)), /72 字节/);
  // 24 个汉字 = 72 字节；再多一个字就超限。
  assert.equal(newPasswordProblem('密'.repeat(24)), '');
  assert.match(newPasswordProblem('密'.repeat(25)), /72 字节/);
  // 已有密码不受新的最小长度约束。
  assert.equal(existingPasswordProblem('short'), '');
  assert.match(existingPasswordProblem('a'.repeat(73), '原密码'), /^原密码不能超过 72 字节$/);
});
