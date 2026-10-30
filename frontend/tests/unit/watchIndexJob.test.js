import assert from 'node:assert/strict';
import test from 'node:test';

import { isFailedIndexJob, waitForIndexJob } from '../../src/composables/watchIndexJob.js';

const noSleep = async () => {};

test('returns the job once it reaches a terminal state', async () => {
  const responses = [
    [{ id: 'j1', status: 'queued' }],
    [{ id: 'j1', status: 'running' }],
    [{ id: 'j0', status: 'completed' }, { id: 'j1', status: 'failed', error_message: 'boom' }]
  ];
  const urls = [];
  const job = await waitForIndexJob('研发 部', 'j1', {
    get: async url => { urls.push(url); return responses.shift(); },
    sleep: noSleep
  });

  assert.equal(job.status, 'failed');
  assert.equal(isFailedIndexJob(job), true);
  assert.equal(urls.length, 3);
  assert.equal(urls[0], '/api/kbs/%E7%A0%94%E5%8F%91%20%E9%83%A8/index_jobs');
});

test('completed and superseded jobs are not reported as failures', async () => {
  for (const status of ['completed', 'superseded']) {
    const job = await waitForIndexJob('库', 'j1', {
      get: async () => [{ id: 'j1', status }],
      sleep: noSleep
    });
    assert.equal(isFailedIndexJob(job), false);
  }
});

test('gives up quietly when the job never finishes or the lookup fails', async () => {
  const pending = await waitForIndexJob('库', 'j1', {
    get: async () => [{ id: 'j1', status: 'running' }],
    delays: [1, 1],
    sleep: noSleep
  });
  const broken = await waitForIndexJob('库', 'j1', {
    get: async () => { throw new Error('network'); },
    sleep: noSleep
  });

  assert.equal(pending, null);
  assert.equal(broken, null);
  assert.equal(isFailedIndexJob(null), false);
});
