import assert from 'node:assert/strict';
import test from 'node:test';

import { usePaginatedList } from '../../src/composables/usePaginatedList.js';

test('uses server page metadata and omits an empty search query', async () => {
  const requests = [];
  const list = usePaginatedList('/api/users', {
    apiGet: async url => {
      requests.push(url);
      return { items: [{ id: 1 }], page: 2, total: 21, total_pages: 3 };
    }
  });

  list.page.value = 2;
  await list.fetchPage('  ');

  assert.deepEqual(requests, ['/api/users?page=2&page_size=20']);
  assert.deepEqual(list.items.value, [{ id: 1 }]);
  assert.equal(list.total.value, 21);
  assert.equal(list.totalPages.value, 3);
});

test('ignores a stale response after a newer page request', async () => {
  let resolveFirst;
  const list = usePaginatedList('/api/roles', {
    apiGet: url => {
      if (url.includes('page=1')) {
        return new Promise(resolve => { resolveFirst = resolve; });
      }
      return Promise.resolve({ items: [{ id: 2 }], page: 2, total: 2, total_pages: 2 });
    }
  });

  const first = list.fetchPage();
  list.page.value = 2;
  await list.fetchPage();
  resolveFirst({ items: [{ id: 1 }], page: 1, total: 2, total_pages: 2 });
  await first;

  assert.deepEqual(list.items.value, [{ id: 2 }]);
  assert.equal(list.page.value, 2);
  assert.equal(list.loading.value, false);
});
