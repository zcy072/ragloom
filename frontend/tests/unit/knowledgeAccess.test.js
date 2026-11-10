import assert from 'node:assert/strict';
import { test } from 'node:test';

import { filterKnowledgeBases } from '../../src/knowledgeAccess.js';

test('chat options exclude deleting libraries; management retains only authorized ones', () => {
  const libraries = [
    { kb_id: 'a', state: 'active' },
    { kb_id: 'b', state: 'deleting' },
    { kb_id: 'c', state: 'deleting' },
  ];
  const user = { knowledgeBaseIds: ['a', 'b'] };
  assert.deepEqual(filterKnowledgeBases(libraries, user).map(kb => kb.kb_id), ['a']);
  assert.deepEqual(filterKnowledgeBases(libraries, user, { includeDeleting: true }).map(kb => kb.kb_id), ['a', 'b']);
});
