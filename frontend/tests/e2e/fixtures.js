import { expect } from '@playwright/test';

export const adminUser = {
  role: '超级管理员',
  username: 'e2e-admin',
  menuPermissions: ['仪表盘统计', '知识库', '知识库管理', '问答对话', '模型配置', '用户管理', '角色管理'],
  knowledgeBasePermissions: ['全公司']
};

const knowledgeBases = [
  { name: '通用知识库', description: '公共制度与流程', docs: 1, date: '2026-07-23' },
  { name: '研发知识库', description: '研发资料', docs: 0, date: '2026-07-22' }
];

const roles = [
  { id: 1, name: '超级管理员', description: '系统内置角色', menu_permissions: [], knowledge_base_permissions: ['全公司'] },
  { id: 2, name: '研发访客', description: '只读访问研发资料', menu_permissions: ['问答对话'], knowledge_base_permissions: ['研发知识库'] },
  { id: 3, name: '研发编辑', description: '维护研发资料', menu_permissions: ['知识库', '知识库管理'], knowledge_base_permissions: ['研发知识库'] },
  { id: 4, name: '研发管理员', description: '管理研发角色和资料', menu_permissions: ['角色管理'], knowledge_base_permissions: ['研发知识库'] },
  ...Array.from({ length: 21 }, (_, index) => ({
    id: index + 5,
    name: `项目角色 ${String(index + 1).padStart(2, '0')}`,
    description: `用于分页验证的角色 ${index + 1}`,
    menu_permissions: [],
    knowledge_base_permissions: ['研发知识库']
  }))
];

const users = Array.from({ length: 23 }, (_, index) => ({
  id: index + 1,
  username: `member-${String(index + 1).padStart(2, '0')}`,
  name: `用户 ${String(index + 1).padStart(2, '0')}`,
  phone: '13800000000',
  status: '正常',
  role: '研发访客',
  time: '2026-07-23 16:00:00'
}));

const dashboardStats = {
  kbs: 2,
  docs: 5,
  size: '12.5 MB',
  tasks: 5,
  kb_list: [
    {
      name: '通用知识库',
      docs: 5,
      size: '12.5 MB',
      success: 4,
      fail: 1,
      processing: 0,
      types: [{ ext: 'PDF', count: 5 }]
    }
  ],
  pie_data: [
    { name: '成功', value: 4 },
    { name: '失败', value: 1 },
    { name: '处理中', value: 0 }
  ],
  line_data: [
    { name: '07-21', 会话数: 2, 回答数: 2 },
    { name: '07-22', 会话数: 5, 回答数: 4 },
    { name: '07-23', 会话数: 3, 回答数: 3 }
  ]
};

const json = (route, body, status = 200) => route.fulfill({
  status,
  contentType: 'application/json',
  body: JSON.stringify(body)
});

const paginate = (items, url, fields) => {
  const params = new URL(url).searchParams;
  const pageSize = Number(params.get('page_size') || 20);
  const query = (params.get('query') || '').trim().toLowerCase();
  const matches = query
    ? items.filter(item => fields.some(field => String(item[field] || '').toLowerCase().includes(query)))
    : items;
  const totalPages = Math.max(1, Math.ceil(matches.length / pageSize));
  const page = Math.min(Math.max(Number(params.get('page') || 1), 1), totalPages);
  const start = (page - 1) * pageSize;
  return {
    items: matches.slice(start, start + pageSize),
    total: matches.length,
    page,
    page_size: pageSize,
    total_pages: totalPages
  };
};

export async function installApiMocks(page, { onAdminLogin, onChatRequest, onSessionRename } = {}) {
  // 重命名后列表接口要返回新标题，和真实后端一致。
  const sessionTitles = { 'session-1': '已有会话', 'session-2': '研发排期' };
  await page.route('**/api/**', async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();

    if (path === '/api/login' && method === 'POST') {
      onAdminLogin?.(request.postDataJSON());
      return json(route, { access_token: 'e2e-token', token_type: 'bearer', user: adminUser });
    }
    if (path === '/api/me' && method === 'GET') return json(route, adminUser);
    if (path === '/api/kbs' && method === 'GET') return json(route, knowledgeBases);
    if (path === '/api/users' && method === 'GET') return json(route, paginate(users, request.url(), ['username', 'name']));
    if (path === '/api/roles' && method === 'GET') return json(route, paginate(roles, request.url(), ['name', 'description']));
    if (path.match(/^\/api\/roles\/\d+$/) && method === 'DELETE') return json(route, { status: 'success' });
    if (path === '/api/models' && method === 'GET') {
      return json(route, [{ provider: 'mock-model', configured: true, is_active: 1 }]);
    }
    if (path === '/api/chat/sessions' && method === 'GET') {
      return json(route, [
        { id: 'session-1', title: sessionTitles['session-1'], kb_name: '通用知识库' },
        { id: 'session-2', title: sessionTitles['session-2'], kb_name: '研发知识库' }
      ]);
    }
    const renameMatch = path.match(/^\/api\/chat\/sessions\/([^/]+)\/title$/);
    if (renameMatch && method === 'PUT') {
      const payload = request.postDataJSON();
      onSessionRename?.(renameMatch[1], payload);
      sessionTitles[renameMatch[1]] = payload.title;
      return json(route, { status: 'success' });
    }
    if (path === '/api/chat/sessions/session-1' && method === 'GET') {
      return json(route, [{ role: 'assistant', content: '历史回答', sources: [] }]);
    }
    if (path.endsWith('/document_filters') && method === 'GET') {
      return json(route, { sources: ['制度.pdf'], document_types: ['制度'], tags: ['有效'] });
    }
    if (path === '/api/chat' && method === 'POST') {
      const payload = request.postDataJSON();
      onChatRequest?.(payload);
      if (payload.question === '模拟异常断流') {
        return route.fulfill({ status: 200, contentType: 'text/event-stream', body: '' });
      }
      const body = [
        'data: {"chunk":"冒烟"}\n\n',
        'data: {"chunk":"回答"}\n\n',
        'data: {"is_end":true,"final_answer":"冒烟回答 [来源 1]","sources":[{"id":1,"citation_ids":[1],"source":"制度.pdf","kb_name":"通用知识库","page":"第 1 页","content":"制度引用内容"}],"session_id":"session-1"}\n\n'
      ].join('');
      return route.fulfill({ status: 200, contentType: 'text/event-stream', body });
    }
    if (path === '/api/document_formats' && method === 'GET') {
      return json(route, [{ extension: '.pdf' }, { extension: '.txt' }]);
    }
    if (path === '/api/document_parsers' && method === 'GET') {
      return json(route, {
        document: [
          { id: 'native_text', label: '本地文本解析', available: true, extensions: ['.pdf', '.docx'] },
          { id: 'mineru_cloud', label: 'MinerU 云端解析', available: true, extensions: ['.pdf', '.docx'] },
          { id: 'paddle_ocr', label: 'Paddle 本地 OCR', available: true, extensions: ['.pdf', '.png', '.jpg', '.jpeg'] },
          { id: 'vision_model', label: '智能视觉解析（失败时 Paddle OCR）', available: true, extensions: ['.png', '.jpg', '.jpeg'] }
        ]
      });
    }
    if (path.endsWith('/chunk_config') && method === 'GET') {
      return json(route, { chunk_size: 800, chunk_overlap: 150, index_needs_rebuild: false });
    }
    if (path.endsWith('/retrieval_config') && method === 'GET') {
      return json(route, { query_rewrite_enabled: true });
    }
    if (path === '/api/documents/%E9%80%9A%E7%94%A8%E7%9F%A5%E8%AF%86%E5%BA%93' && method === 'GET') {
      return json(route, [{
        filename: '制度.pdf',
        title: '制度文件',
        document_type: '制度',
        tags: ['有效'],
        effective_date: '2026-01-01',
        parser_mode: 'native_text',
        status: 'ready',
        size: 2048,
        uploaded_at: '2026-07-23T08:00:00+00:00'
      }]);
    }
    if (path.includes('/api/documents/') && path.endsWith('%E5%88%B6%E5%BA%A6.pdf') && method === 'GET') {
      return route.fulfill({ status: 200, contentType: 'application/pdf', body: '%PDF-1.4\n%%EOF' });
    }
    if (path === '/api/dashboard/stats' && method === 'GET') return json(route, dashboardStats);

    return json(route, { detail: `Unhandled E2E request: ${method} ${path}` }, 501);
  });
}

export async function seedAuthSession(page) {
  await page.addInitScript(session => {
    localStorage.setItem('pymodel.auth', JSON.stringify(session));
  }, { token: 'e2e-token', user: adminUser });
}

export function collectPageErrors(page) {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => {
    if (message.type() === 'error') errors.push(message.text());
  });
  return errors;
}

export async function prepareAuthenticatedPage(page, options = {}) {
  const errors = collectPageErrors(page);
  await seedAuthSession(page);
  await installApiMocks(page, options);
  await page.goto('/');
  await expect(page.getByRole('heading', { name: '问答对话' })).toBeVisible();
  return errors;
}
