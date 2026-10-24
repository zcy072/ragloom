import assert from 'node:assert/strict';
import test from 'node:test';

import { ref } from 'vue';

import { useChatSessions } from '../../src/composables/useChatSessions.js';

test('late history responses cannot replace a new conversation', async () => {
  let resolveHistory;
  const messages = ref([]);
  const composable = useChatSessions({ messages, selectedKb: ref('库') }, {
    store: { activeSessionId: null },
    apiGet: () => new Promise(resolve => { resolveHistory = resolve; })
  });
  const pending = composable.loadSessionMessages('old');
  assert.equal(composable.loadingSession.value, true);
  composable.startNewChat();
  resolveHistory([{ role: 'assistant', content: '旧回答' }]);
  await pending;
  assert.deepEqual(messages.value, []);
  assert.equal(composable.loadingSession.value, false);
});

test('deleting the active session resets chat state and refreshes the list', async () => {
  const messages = ref([{ role: 'user', content: '旧消息' }]);
  const selectedKb = ref('研发部');
  const sessionStore = { activeSessionId: 'session-1' };
  const deleted = [];
  const composable = useChatSessions(
    { messages, selectedKb },
    {
      store: sessionStore,
      apiDelete: async url => deleted.push(url),
      apiGet: async () => [{ id: 'session-2', kb_name: '财务部' }]
    }
  );

  await composable.deleteSession('session-1');

  assert.deepEqual(deleted, ['/api/chat/sessions/session-1']);
  assert.equal(sessionStore.activeSessionId, null);
  assert.equal(composable.newSessionPending.value, true);
  assert.deepEqual(messages.value, []);
  assert.deepEqual(composable.sessions.value, [{ id: 'session-2', kb_name: '财务部' }]);
  assert.equal(composable.deletingSessionId.value, null);
});

test('loading a session updates its knowledge base and clears new-session state', async () => {
  const messages = ref([]);
  const selectedKb = ref('研发部');
  const sessionStore = { activeSessionId: null };
  const composable = useChatSessions(
    { messages, selectedKb },
    {
      store: sessionStore,
      apiGet: async () => [{ role: 'assistant', content: '历史回答', sources: [] }]
    }
  );
  composable.sessions.value = [{ id: 'session/2', kb_name: '财务部' }];
  composable.newSessionPending.value = true;

  await composable.loadSessionMessages('session/2');

  assert.equal(sessionStore.activeSessionId, 'session/2');
  assert.equal(composable.newSessionPending.value, false);
  assert.equal(selectedKb.value, '财务部');
  assert.equal(messages.value[0].content, '历史回答');
});

const renameFixture = (overrides = {}) => {
  const calls = [];
  const errors = [];
  const composable = useChatSessions(
    { messages: ref([]), selectedKb: ref('研发部') },
    {
      store: { activeSessionId: null },
      toast: { error: message => errors.push(message) },
      apiPut: async (url, body) => { calls.push([url, body]); },
      apiGet: async () => [{ id: 'a/1', title: '新标题', kb_name: '研发部' }],
      ...overrides
    }
  );
  composable.sessions.value = [{ id: 'a/1', title: '旧标题', kb_name: '研发部' }];
  return { composable, calls, errors };
};

test('renaming a session sends the trimmed title and refreshes the list', async () => {
  const { composable, calls } = renameFixture();

  const renamed = await composable.renameSession('a/1', '  新标题  ');

  assert.equal(renamed, true);
  assert.deepEqual(calls, [['/api/chat/sessions/a%2F1/title', { title: '新标题' }]]);
  assert.equal(composable.sessions.value[0].title, '新标题');
});

test('renaming to an empty or unchanged title does not call the server', async () => {
  const { composable, calls } = renameFixture();

  assert.equal(await composable.renameSession('a/1', '   '), false);
  assert.equal(await composable.renameSession('a/1', '旧标题'), false);
  assert.equal(await composable.renameSession('missing', '标题'), false);
  assert.deepEqual(calls, []);
});

test('a failed rename keeps the old title and reports the error', async () => {
  const { composable, errors } = renameFixture({
    apiPut: async () => { throw new Error('会话不存在或无权访问'); }
  });

  const renamed = await composable.renameSession('a/1', '新标题');

  assert.equal(renamed, false);
  assert.equal(composable.sessions.value[0].title, '旧标题');
  assert.deepEqual(errors, ['会话不存在或无权访问']);
});

test('a failed session list load keeps the existing list and tells the user', async () => {
  const errors = [];
  const composable = useChatSessions({ messages: ref([]), selectedKb: ref('库') }, {
    store: { activeSessionId: null },
    toast: { error: message => errors.push(message) },
    apiGet: async () => { throw new Error('服务暂时不可用'); }
  });
  composable.sessions.value = [{ id: 's1', kb_name: '库' }];

  await composable.loadSessions();

  assert.deepEqual(composable.sessions.value, [{ id: 's1', kb_name: '库' }]);
  assert.deepEqual(errors, ['服务暂时不可用']);
});

test('an older session list response cannot overwrite a newer one', async () => {
  const resolvers = [];
  const composable = useChatSessions({ messages: ref([]), selectedKb: ref('库') }, {
    store: { activeSessionId: null },
    apiGet: () => new Promise(resolve => resolvers.push(resolve))
  });

  const first = composable.loadSessions();
  const second = composable.loadSessions();
  resolvers[1]([{ id: 'new' }]);
  await second;
  resolvers[0]([{ id: 'old' }]);
  await first;

  assert.deepEqual(composable.sessions.value, [{ id: 'new' }]);
});

test('a failed session history load tells the user', async () => {
  const errors = [];
  const composable = useChatSessions({ messages: ref([]), selectedKb: ref('库') }, {
    store: { activeSessionId: null },
    toast: { error: message => errors.push(message) },
    apiGet: async () => { throw new Error('会话不存在'); }
  });

  await composable.loadSessionMessages('missing');

  assert.deepEqual(errors, ['会话不存在']);
  assert.equal(composable.loadingSession.value, false);
});
