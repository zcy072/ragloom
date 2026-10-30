import assert from 'node:assert/strict';
import test from 'node:test';

import { ref } from 'vue';

import { useChatStream } from '../../src/composables/useChatStream.js';

const streamResponse = chunks => ({
  ok: true,
  body: {
    getReader() {
      let index = 0;
      return {
        async read() {
          if (index >= chunks.length) return { done: true };
          return { done: false, value: new TextEncoder().encode(chunks[index++]) };
        }
      };
    }
  }
});

const createComposable = ({
  chunks = [],
  chatStore = { activeSessionId: null },
  authorizedFetch = async () => streamResponse(chunks),
  idleTimeoutMs,
  webSearch
}) => {
  const messages = ref([]);
  const newSessionPending = ref(true);
  const modelConfigured = ref(true);
  let sessionRefreshes = 0;
  const errors = [];
  const composable = useChatStream(
    {
      messages,
      selectedKb: ref('研发部'),
      selectedModel: ref('OpenAI'),
      modelConfigured,
      hasRetrievalFilters: ref(false),
      retrievalFilterPayload: () => ({}),
      newSessionPending,
      loadSessions: async () => { sessionRefreshes += 1; },
      ...(webSearch ? { webSearch } : {})
    },
    {
      store: chatStore,
      toast: { error: message => errors.push(message) },
      authorizedFetch,
      ...(idleTimeoutMs ? { idleTimeoutMs } : {})
    }
  );
  return {
    composable,
    messages,
    newSessionPending,
    modelConfigured,
    chatStore,
    errors,
    refreshes: () => sessionRefreshes
  };
};

test('SSE events split across reads produce one terminal assistant message', async () => {
  const fixture = createComposable({
    chunks: [
      'data: {"chunk":"答"}\n',
      '\ndata: {"chunk":"案"}\n\ndata: {"sources":[{"id":1}],',
      '"session_id":"new-session","is_end":true,"final_answer":"最终答案"}\n\n'
    ]
  });
  fixture.composable.input.value = '问题';

  await fixture.composable.send();

  assert.equal(fixture.messages.value.length, 2);
  assert.equal(fixture.messages.value[1].content, '最终答案');
  assert.deepEqual(fixture.messages.value[1].sources, [{ id: 1 }]);
  assert.equal(fixture.chatStore.activeSessionId, 'new-session');
  assert.equal(fixture.newSessionPending.value, false);
  assert.equal(fixture.refreshes(), 1);
  assert.equal(fixture.composable.loading.value, false);
});

test('a stream ending without a terminal event reports an interrupted answer', async () => {
  const fixture = createComposable({ chunks: [] });
  fixture.composable.input.value = '问题';

  await fixture.composable.send();

  assert.match(fixture.messages.value[1].content, /连接意外中断/);
  assert.deepEqual(fixture.messages.value[1].sources, []);
  assert.equal(fixture.composable.loading.value, false);
});

test('stopping an active request aborts it without appending an error answer', async () => {
  let requestSignal;
  const fixture = createComposable({
    authorizedFetch: (_url, options) => {
      requestSignal = options.signal;
      return new Promise((_resolve, reject) => {
        options.signal.addEventListener('abort', () => {
          reject(new DOMException('Aborted', 'AbortError'));
        });
      });
    }
  });
  fixture.composable.input.value = '停止这个回答';

  const pendingSend = fixture.composable.send();
  await Promise.resolve();
  fixture.composable.stop();
  await pendingSend;

  assert.equal(requestSignal.aborted, true);
  assert.equal(fixture.composable.loading.value, false);
  assert.deepEqual(fixture.messages.value, [
    { role: 'user', content: '停止这个回答', imageUrl: null }
  ]);
});

test('a 412 response marks the model unavailable and shows backend detail', async () => {
  const fixture = createComposable({
    authorizedFetch: async () => ({
      ok: false,
      status: 412,
      async json() {
        return { detail: '请先配置模型密钥' };
      }
    })
  });
  fixture.composable.input.value = '问题';

  await fixture.composable.send();

  assert.equal(fixture.modelConfigured.value, false);
  assert.equal(fixture.messages.value[1].content, '⚠️ 请先配置模型密钥');
  assert.equal(fixture.composable.loading.value, false);
});

test('a validation error array is replaced by readable text instead of [object Object]', async () => {
  const fixture = createComposable({
    authorizedFetch: async () => ({
      ok: false,
      status: 422,
      async json() {
        return { detail: [{ type: 'string_too_long', loc: ['body', 'question'], msg: 'too long' }] };
      }
    })
  });
  fixture.composable.input.value = '问题';

  await fixture.composable.send();

  assert.match(fixture.messages.value[1].content, /^⚠️ 问题格式不符合要求.*4000/);
  assert.doesNotMatch(fixture.messages.value[1].content, /object/);
});

test('a generation error event is shown as a warning without sources', async () => {
  const fixture = createComposable({ chunks: [
    'data: {"error":"AI 服务暂时不可用，请稍后重试。","session_id":"s1","is_end":true}\n\n'
  ] });
  fixture.composable.input.value = '问题';

  await fixture.composable.send();

  assert.equal(fixture.messages.value.length, 2);
  assert.equal(fixture.messages.value[1].content, '⚠️ AI 服务暂时不可用，请稍后重试。');
  assert.deepEqual(fixture.messages.value[1].sources, []);
  assert.equal(fixture.chatStore.activeSessionId, 's1');
  assert.equal(fixture.errors.length, 0);
});

test('an interrupted terminal event keeps the partial answer and marks it', async () => {
  const fixture = createComposable({ chunks: [
    'data: {"chunk":"部分回答"}\n\n',
    'data: {"is_end":true,"final_answer":"部分回答","sources":[],"session_id":"s1","interrupted":true}\n\n'
  ] });
  fixture.composable.input.value = '问题';

  await fixture.composable.send();

  assert.equal(fixture.messages.value[1].content, '部分回答');
  assert.equal(fixture.messages.value[1].interrupted, true);
});

test('terminal diagnostics stay on the current message', async () => {
  const diagnostics = { skipped: true, rerank_status: 'not_run' };
  const fixture = createComposable({ chunks: [
    `data: ${JSON.stringify({ is_end: true, final_answer: '你好', sources: [], diagnostics })}\n\n`
  ] });
  fixture.composable.input.value = '你好';
  await fixture.composable.send();
  assert.deepEqual(fixture.messages.value[1].diagnostics, diagnostics);
});

test('partial stream preserves text without inventing diagnostics', async () => {
  const fixture = createComposable({ chunks: ['data: {"chunk":"部分回答"}\n\n'] });
  fixture.composable.input.value = '问题';
  await fixture.composable.send();
  assert.equal(fixture.messages.value[1].content, '部分回答');
  assert.equal(fixture.messages.value[1].diagnostics, undefined);
  assert.equal(fixture.errors.length, 1);
});

test('heartbeat comments are ignored and the streamed answer is complete', async () => {
  const fixture = createComposable({
    chunks: [
      ': ping\n\n',
      `data: ${JSON.stringify({ chunk: '你好' })}\n\n: ping\n\n`,
      `data: ${JSON.stringify({ sources: [], session_id: 's1', is_end: true, final_answer: '你好' })}\n\n`
    ]
  });
  fixture.composable.input.value = '问题';

  await fixture.composable.send();

  assert.equal(fixture.messages.value.length, 2);
  assert.equal(fixture.messages.value[1].content, '你好');
  assert.deepEqual(fixture.errors, []);
});

test('a stream that goes silent is aborted, keeps the partial answer and reports it', async () => {
  let requestSignal;
  const fixture = createComposable({
    idleTimeoutMs: 20,
    authorizedFetch: async (_url, options) => {
      requestSignal = options.signal;
      let sent = false;
      return {
        ok: true,
        body: {
          getReader: () => ({
            read: () => {
              if (!sent) {
                sent = true;
                return Promise.resolve({
                  done: false,
                  value: new TextEncoder().encode(`data: ${JSON.stringify({ chunk: '部分回答' })}\n\n`)
                });
              }
              return new Promise(() => {});
            },
            cancel: async () => {}
          })
        }
      };
    }
  });
  fixture.composable.input.value = '问题';

  await fixture.composable.send();

  assert.equal(requestSignal.aborted, true);
  assert.equal(fixture.composable.loading.value, false);
  assert.equal(fixture.messages.value[1].content, '部分回答');
  assert.equal(fixture.errors.length, 1);
});

test('answer text is flushed to the message even when updates are throttled', async () => {
  const tokens = Array.from({ length: 50 }, (_, index) => `词${index}`);
  const fixture = createComposable({
    chunks: [
      ...tokens.map(chunk => `data: ${JSON.stringify({ chunk })}\n\n`),
      `data: ${JSON.stringify({ sources: [], session_id: 's1', is_end: true, final_answer: tokens.join('') })}\n\n`
    ]
  });
  fixture.composable.input.value = '问题';

  await fixture.composable.send();

  assert.equal(fixture.messages.value[1].content, tokens.join(''));
});

test('stage events are collected live and attached to the answer with its trace ID', async () => {
  const fixture = createComposable({
    chunks: [
      'data: {"stage":"plan","status":"ok","elapsed_ms":5,"detail":{}}\n\n',
      'data: {"stage":"retrieval","status":"ok","elapsed_ms":20,"detail":{"candidates":3}}\n\n',
      'data: {"chunk":"答"}\n\n',
      'data: {"stage":"generate","status":"ok","elapsed_ms":30,"detail":{}}\n\n',
      'data: {"sources":[],"session_id":"s","is_end":true,"final_answer":"答","trace_id":"t1"}\n\n'
    ]
  });
  fixture.composable.input.value = '问题';

  await fixture.composable.send();

  const answer = fixture.messages.value[1];
  assert.equal(fixture.messages.value.length, 2);
  assert.deepEqual(answer.stages.map(stage => stage.stage), ['plan', 'retrieval', 'generate']);
  assert.equal(answer.trace_id, 't1');
  assert.equal(answer.feedback, 0);
  assert.equal(fixture.composable.liveStages.value.length, 3);
});

test('the web search toggle is sent only when enabled', async () => {
  const bodies = [];
  const webSearch = ref(true);
  const fixture = createComposable({
    webSearch,
    authorizedFetch: async (_path, options) => {
      bodies.push(JSON.parse(options.body));
      return streamResponse(['data: {"sources":[],"session_id":"s","is_end":true,"final_answer":"答"}\n\n']);
    }
  });
  fixture.composable.input.value = '问题';
  await fixture.composable.send();
  webSearch.value = false;
  fixture.composable.input.value = '问题二';
  await fixture.composable.send();

  assert.equal(bodies[0].web_search, true);
  assert.equal('web_search' in bodies[1], false);
});
