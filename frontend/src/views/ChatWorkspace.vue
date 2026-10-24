<template>
  <div class="flex h-full bg-white w-full">
    <ChatSessionSidebar
      :sessions="sessions"
      :active-session-id="store.activeSessionId"
      :deleting-session-id="deletingSessionId"
      @new-chat="handleStop(); handleNewChat()"
      @select="handleSelectSession"
      @delete="handleStop(); confirmDelete($event)"
      @rename="renameSession($event.id, $event.title)"
    />

    <!-- Right Area: Chat -->
    <div class="flex-1 flex flex-col relative bg-white min-w-0">
      <div class="px-6 py-4 border-b border-[#e5eded] flex items-center gap-4 flex-wrap bg-[#fbfdfd]">
        <div class="flex items-center gap-2">
          <span class="text-sm text-gray-500 w-20">知识库</span>
          <div class="relative">
            <select
              v-model="selectedKb"
              class="appearance-none bg-white border border-[#d8e4e4] text-[#385154] py-2 px-3 pr-8 rounded-md leading-tight focus:outline-none focus:border-[#68aaa7] text-sm"
            >
              <option v-for="kb in kbs" :key="kb.name" :value="kb.name">{{ kb.name }}</option>
            </select>
            <div class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-500">
              <ChevronDown class="h-4 w-4" />
            </div>
          </div>
        </div>
        <div v-if="models.length > 0" class="flex items-center gap-2">
          <span class="text-sm text-gray-500">模型</span>
          <div class="relative">
            <select
              v-model="selectedModel"
              class="appearance-none bg-white border border-[#d8e4e4] text-[#385154] py-2 px-3 pr-8 rounded-md leading-tight focus:outline-none focus:border-[#68aaa7] text-sm"
            >
              <option v-for="m in models.filter(md => md.configured)" :key="m.provider" :value="m.provider">{{ m.provider }}</option>
            </select>
            <div class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-500">
              <ChevronDown class="h-4 w-4" />
            </div>
          </div>
        </div>
        <button
          @click="showRetrievalFilters = !showRetrievalFilters"
          :class="['ml-auto inline-flex h-8 w-8 items-center justify-center rounded-md border transition-colors', hasRetrievalFilters ? 'border-[#7bbab6] bg-[#e8f5f3] text-[#176d70]' : 'border-[#d8e4e4] bg-white text-gray-500 hover:bg-[#f4f8f8]']"
          title="检索筛选"
        >
          <SlidersHorizontal class="h-4 w-4" />
        </button>
        <button @click="downloadSession" :disabled="loading || loadingSession || !messages.length" class="app-icon-button disabled:opacity-40" title="导出当前会话 Markdown" aria-label="导出当前会话 Markdown"><Download class="h-4 w-4" /></button>
      </div>

      <ChatRetrievalFilters
        v-if="showRetrievalFilters"
        v-model="retrievalFilters"
        :options="filterOptions"
        @reset="resetRetrievalFilters"
      />

      <div v-if="!modelConfigured" class="mx-6 mt-4 bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 flex items-start gap-3">
        <span class="text-amber-500 text-lg leading-none mt-0.5">⚠️</span>
        <div class="text-[13px] text-amber-800">
          <strong>尚未配置大模型 API Key</strong>
          <p class="mt-1 text-amber-700">问答功能需要配置 AI 大模型才能使用。请前往左侧菜单的「模型管理」页面，填写并保存您的 API Key。</p>
        </div>
      </div>

      <div ref="scrollRef" @scroll="updateStickToBottom" class="flex-1 overflow-y-auto p-6 space-y-6">
        <div v-for="(msg, i) in messages" :key="i" :class="['flex w-full', msg.role === 'user' ? 'justify-end' : 'justify-start']">
          <template v-if="msg.role === 'user'">
          <div class="bg-[#176d70] text-white px-5 py-3 rounded-lg rounded-tr-sm max-w-2xl text-[14px] leading-relaxed shadow-sm">
              <img v-if="msg.imageUrl" :src="msg.imageUrl" alt="附图" class="mb-2 max-w-xs max-h-48 rounded-lg object-contain border border-blue-300/40" />
              {{ msg.content }}
            </div>
          </template>
          <template v-else>
            <div class="flex flex-col w-full min-w-0 max-w-4xl">
              <div class="bg-white border border-[#dfe9e9] rounded-lg p-5 text-[14px] leading-relaxed text-[#263a3d] shadow-[0_1px_2px_rgba(18,53,57,0.04)]">
                <ChatAnswer :content="msg.content" :sources="msg.sources || []" @citation="openPreview" />
                <button @click="copyAnswer(msg.content)" :disabled="loading || loadingSession || !msg.content" class="app-icon-button mt-2 disabled:opacity-40" title="复制回答" aria-label="复制回答"><Copy class="h-4 w-4" /></button>
              </div>
              <div v-if="msg.interrupted" class="mt-2 text-[12px] text-amber-700">回答生成中断，以上内容可能不完整，请重试。</div>
              <RetrievalDiagnostics v-if="msg.diagnostics" :diagnostics="msg.diagnostics" />
              <div v-if="visibleSources(msg).length > 0" class="mt-3">
                <div class="flex items-center text-[12px] text-gray-500 mb-2">
                  <LinkIcon class="h-3 w-3 mr-1" />
                  <span>引用来源 ({{ visibleSources(msg).length }}个文件)</span>
                </div>
                <div class="space-y-2">
                  <div v-for="(src, idx) in visibleSources(msg)" :key="`${src.kb_name || selectedKb}:${src.source}:${src.id || idx}`" @click="openPreview(src)" class="border border-[#dce7e7] rounded-md bg-[#fbfdfd] p-3 text-[13px] hover:border-[#8fc9c4] transition-colors cursor-pointer">
                    <div class="flex items-center justify-between mb-1">
                      <div class="flex min-w-0 items-center gap-2 text-blue-600 font-medium w-3/4 hover:underline">
                        <span class="truncate">[{{ sourceCitationLabel(src) }}] {{ src.source }}</span>
                        <span v-if="src.kb_name && src.kb_name !== selectedKb" class="shrink-0 text-[10px] font-normal text-teal-700">{{ src.kb_name }}</span>
                      </div>
                      <div v-if="src.page" class="text-gray-400 text-[11px] whitespace-nowrap bg-gray-50 px-1.5 py-0.5 rounded">{{ src.page }}</div>
                    </div>
                    <div v-if="src.context_title" class="mb-1 truncate text-[11px] text-teal-700">{{ src.context_title }}</div>
                    <div v-if="src.content" class="text-gray-600">{{ src.content }}</div>
                    <div v-if="src.evidence?.length" class="mt-2 flex flex-wrap gap-2">
                      <button v-for="entry in src.evidence" :key="entry.id" @click.stop="openPreview(entry)" class="text-teal-700 underline" :title="entry.content">[来源 {{ entry.id }}] {{ entry.page }}</button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </template>
        </div>
        <div v-if="loading" class="flex justify-start w-full">
          <div class="bg-white border border-gray-200 rounded-lg p-4 text-[14px] text-gray-500 shadow-sm flex items-center">
            <div class="animate-pulse">正在检索知识库思考中...</div>
          </div>
        </div>
        <div ref="endRef"></div>
      </div>

      <ChatComposer
        v-model="input"
        :max-length="MAX_QUESTION_CHARS"
        :loading="loading"
        :disabled="initializing || loadingSession"
        :attached-image="attachedImage"
        @send="handleSend"
        @stop="handleStop"
        @image-select="handleImageSelect"
        @image-paste="handleImagePaste"
        @remove-image="removeAttachedImage"
      />
    </div>

    <DocumentPreview
      v-if="showPreview"
      :filename="previewDoc"
      :url="documentPreviewUrl"
      :evidence="previewEvidence"
      :context-label="previewKbName && previewKbName !== selectedKb ? previewKbName : ''"
      @close="showPreview = false"
    />
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue';
import { store } from '../store';
import { ChevronDown, Link as LinkIcon, SlidersHorizontal, Copy, Download } from 'lucide-vue-next';
import { sessionMarkdown, exportFilename } from '../chatExport';
import { toast } from '../toast';
import { filterKnowledgeBases } from '../knowledgeAccess';
import { apiGet } from '../apiClient';
import ChatComposer from '../components/ChatComposer.vue';
import ChatAnswer from '../components/ChatAnswer.vue';
import RetrievalDiagnostics from '../components/RetrievalDiagnostics.vue';
import ChatRetrievalFilters from '../components/ChatRetrievalFilters.vue';
import ChatSessionSidebar from '../components/ChatSessionSidebar.vue';
import DocumentPreview from '../components/DocumentPreview.vue';
import { useChatSessions } from '../composables/useChatSessions';
import { MAX_QUESTION_CHARS, useChatStream } from '../composables/useChatStream';

const messages = ref([]);
const initializing = ref(true);
const endRef = ref(null);
const scrollRef = ref(null);

const kbs = ref([]);
const selectedKb = ref('通用知识库');
const models = ref([]);
const selectedModel = ref('');
const modelConfigured = ref(true);
const showRetrievalFilters = ref(false);
const filterOptions = ref({ sources: [], document_types: [], tags: [] });
const retrievalFilters = ref({
  source: '',
  document_type: '',
  tag: '',
  effective_date_from: '',
  effective_date_to: ''
});

const previewDoc = ref(null);
const previewEvidence = ref(null);
const previewPage = ref(null);
const previewKbName = ref('');
const showPreview = ref(false);
const hasRetrievalFilters = computed(() => Object.values(retrievalFilters.value).some(Boolean));
const documentPreviewUrl = computed(() => {
  if (!previewDoc.value) return '';
  const sourceDept = previewKbName.value || selectedKb.value;
  const baseUrl = `/api/documents/${encodeURIComponent(sourceDept)}/${encodeURIComponent(previewDoc.value)}?preview=true`;
  if (!/\.pdf$/i.test(previewDoc.value) || !previewPage.value) return baseUrl;
  return `${baseUrl}#page=${previewPage.value}`;
});
// 距离底部多少像素以内仍视为“正在跟随最新内容”。
const STICK_TO_BOTTOM_THRESHOLD = 80;
/** 用户向上翻看历史时为 false，此时流式输出不再强制拉回底部。 */
let stickToBottom = true;
/** 根据滚动位置判断用户是否停留在底部附近。 */
const updateStickToBottom = () => {
  const el = scrollRef.value;
  if (!el) return;
  stickToBottom = el.scrollHeight - el.scrollTop - el.clientHeight <= STICK_TO_BOTTOM_THRESHOLD;
};
/** 等待消息渲染完成后滚动到对话底部；force 用于发送消息、切换会话等必须回到底部的场景。 */
const scrollToBottom = async (force = false) => {
  if (force) stickToBottom = true;
  if (!stickToBottom) return;
  await nextTick();
  // 流式输出每个 token 都会触发滚动，平滑动画会被反复打断而抖动，生成期间直接跳到底部。
  endRef.value?.scrollIntoView({ behavior: loading.value ? 'auto' : 'smooth' });
};
/** 清空所有文档元数据检索条件。 */
const resetRetrievalFilters = () => {
  retrievalFilters.value = {
    source: '',
    document_type: '',
    tag: '',
    effective_date_from: '',
    effective_date_to: ''
  };
};
// 筛选项请求序号：快速切换知识库时丢弃较早发出的响应。
let filterOptionsRequestVersion = 0;
/** 加载当前知识库可用于检索筛选的来源、类型和标签。 */
const loadFilterOptions = async () => {
  if (!selectedKb.value) return;
  const requestVersion = ++filterOptionsRequestVersion;
  try {
    const data = await apiGet(`/api/kbs/${encodeURIComponent(selectedKb.value)}/document_filters`);
    if (requestVersion !== filterOptionsRequestVersion) return;
    filterOptions.value = {
      sources: data.sources || [],
      document_types: data.document_types || [],
      tags: data.tags || []
    };
  } catch {
    if (requestVersion !== filterOptionsRequestVersion) return;
    filterOptions.value = { sources: [], document_types: [], tags: [] };
  }
};
/** 只把用户实际填写的检索条件转换为请求参数。 */
const retrievalFilterPayload = () => ({
  ...(retrievalFilters.value.source ? { source: retrievalFilters.value.source } : {}),
  ...(retrievalFilters.value.document_type ? { document_type: retrievalFilters.value.document_type } : {}),
  ...(retrievalFilters.value.tag ? { tags: [retrievalFilters.value.tag] } : {}),
  ...(retrievalFilters.value.effective_date_from ? { effective_date_from: retrievalFilters.value.effective_date_from } : {}),
  ...(retrievalFilters.value.effective_date_to ? { effective_date_to: retrievalFilters.value.effective_date_to } : {})
});

const {
  sessions,
  loadingSession,
  deletingSessionId,
  newSessionPending,
  loadSessions,
  loadSessionMessages,
  startNewChat: handleNewChat,
  deleteSession: confirmDelete,
  renameSession
} = useChatSessions({ messages, selectedKb });

const {
  input,
  loading,
  attachedImage,
  handleImageSelect,
  handleImagePaste,
  removeAttachedImage,
  send: sendQuestion,
  stop: handleStop
} = useChatStream({
  messages,
  selectedKb,
  selectedModel,
  modelConfigured,
  hasRetrievalFilters,
  retrievalFilterPayload,
  newSessionPending,
  loadSessions
});

/** 切换会话；点击当前会话时什么都不做，避免误点中断正在生成的回答。 */
const handleSelectSession = (sessionId) => {
  if (sessionId === store.activeSessionId) return;
  handleStop();
  loadSessionMessages(sessionId);
};

const handleSend = () => {
  if (initializing.value || loadingSession.value) return;
  // 用户自己发出问题后应看到新消息，即使之前正在向上翻看。
  stickToBottom = true;
  return sendQuestion();
};

const copyAnswer = async (content) => {
  if (loading.value || loadingSession.value || !content) return;
  try {
    await navigator.clipboard.writeText(content);
    toast.success('回答已复制');
  } catch {
    toast.error('复制失败，请检查浏览器剪贴板权限');
  }
};
const downloadSession = () => {
  if (loading.value || loadingSession.value || !messages.value.length) return;
  const session = sessions.value.find(item => item.id === store.activeSessionId);
  const title = session?.title || '新会话';
  let url;
  try {
    const content = sessionMarkdown(messages.value, title, session?.kb_name || selectedKb.value);
    url = URL.createObjectURL(new Blob([content], { type: 'text/markdown;charset=utf-8' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = exportFilename(title);
    document.body.appendChild(link);
    link.click();
    link.remove();
  } catch {
    toast.error('会话导出失败');
  } finally {
    if (url) setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
};

watch([messages, loading], () => {
  scrollToBottom();
}, { deep: true });

// 历史会话加载完成后从底部开始阅读。不监听 activeSessionId：新会话的首个回答结束时
// 它才会被赋值，此时强制滚动会打断正在向上翻看的用户。
watch(loadingSession, (isLoading) => {
  if (!isLoading) scrollToBottom(true);
});

onMounted(async () => {
  try {
    const data = await apiGet('/api/kbs');
    const filtered = filterKnowledgeBases(data, store.user);
    kbs.value = filtered;
    
    if (store.chatDefaultKb && filtered.find(k => k.name === store.chatDefaultKb)) {
      selectedKb.value = store.chatDefaultKb;
    } else if (filtered.length > 0 && !filtered.find(k => k.name === selectedKb.value)) {
      selectedKb.value = filtered[0].name;
    }
  } catch (e) {
    console.error("Failed to fetch KBs", e);
  }

  await loadFilterOptions();
    
  await loadSessions();
  
  // If we navigated from KnowledgeBase, aggressively override active session
  if (store.chatDefaultKb) {
    const latestForKb = sessions.value.find(s => s.kb_name === selectedKb.value);
    if (latestForKb) {
      await loadSessionMessages(latestForKb.id);
    } else {
      handleNewChat();
    }
    store.chatDefaultKb = null; // Clear the flag after processing
  } else if (store.activeSessionId) {
    await loadSessionMessages(store.activeSessionId);
  } else if (sessions.value.length > 0 && !newSessionPending.value) {
    // Auto-resume latest session if none is active (normal reload)
    const latestForKb = sessions.value.find(s => s.kb_name === selectedKb.value);
    if (latestForKb) {
      await loadSessionMessages(latestForKb.id);
    } else {
      await loadSessionMessages(sessions.value[0].id);
    }
  }

  try {
    const data = await apiGet('/api/models');
    models.value = data;
    const active = data.find(m => m.is_active === 1);
    if (active) selectedModel.value = active.provider;
    syncModelConfigured();
  } catch {
    models.value = [];
    syncModelConfigured();
  } finally {
    initializing.value = false;
  }
});
/** 根据当前选中的模型同步“是否已配置密钥”状态。 */
const syncModelConfigured = () => {
  const model = models.value.find(item => item.provider === selectedModel.value);
  modelConfigured.value = Boolean(model?.configured);
};

watch(selectedModel, syncModelConfigured);

watch(selectedKb, async (newKb) => {
  if (!newKb || initializing.value) return;
  handleStop();
  resetRetrievalFilters();
  await loadFilterOptions();
  if (!sessions.value.length) return;
  const currentSessionInfo = sessions.value.find(s => s.id === store.activeSessionId);
  if (currentSessionInfo && currentSessionInfo.kb_name === newKb) return;
  
  const latestSessionForKb = sessions.value.find(s => s.kb_name === newKb);
  if (latestSessionForKb) {
    await loadSessionMessages(latestSessionForKb.id);
  } else {
    handleNewChat();
  }
});
/** 从回答文本中提取模型实际使用的来源编号。 */
const citationIdsFromAnswer = (content) => {
  const matches = String(content || '').matchAll(/(?:\[|【)\s*来源\s*(\d+)\s*(?:\]|】)/g);
  return [...new Set([...matches].map(match => Number(match[1])))];
};
/** 读取来源关联的全部引用编号。 */
const sourceCitationIds = (source) => (Array.isArray(source?.citation_ids) ? source.citation_ids : []);
/** 将来源编号整理为界面显示文本。 */
const sourceCitationLabel = (source) => sourceCitationIds(source).join(', ');
/** 筛选回答真正引用的来源，并按知识库和文件名合并展示；正文没有引用编号时不展示来源。 */
const visibleSources = (message) => {
  const sources = Array.isArray(message?.sources) ? message.sources : [];
  const citedIds = citationIdsFromAnswer(message?.content);
  const selected = sources.filter(source => sourceCitationIds(source).some(id => citedIds.includes(id)));
  const groups = new Map();

  selected.forEach((source, index) => {
    const key = `${source.kb_name || selectedKb.value}::${source.source || `source-${index}`}`;
    const group = groups.get(key) || {
      ...source,
      citation_ids: [],
      pages: [],
      titles: []
    };
    sourceCitationIds(source).forEach(id => {
      if (!group.citation_ids.includes(id)) group.citation_ids.push(id);
    });
    if (source.page && !group.pages.includes(source.page)) group.pages.push(source.page);
    if (source.context_title && !group.titles.includes(source.context_title)) group.titles.push(source.context_title);
    groups.set(key, group);
  });

  return [...groups.values()].map(({ pages, titles, ...source }) => ({
    ...source,
    page: pages.join('、'),
    context_title: titles.slice(0, 2).join(' / ')
  }));
};
/** 打开来源文档预览，并尽量定位到引用页码。 */
const openPreview = (src) => {
  if (src.evidence?.length) src = src.evidence[0];
  previewEvidence.value = src;
  previewDoc.value = src.source;
  previewKbName.value = src.kb_name || selectedKb.value;
  const pageMatch = String(src.page || '').match(/第\s*(\d+)/);
  previewPage.value = Array.isArray(src.page_numbers)
    ? src.page_numbers.find(page => Number.isInteger(page) && page > 0) || null
    : pageMatch ? Number(pageMatch[1]) : null;
  showPreview.value = true;
};

</script>
