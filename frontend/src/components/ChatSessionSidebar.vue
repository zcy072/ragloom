<template>
  <aside class="w-[260px] bg-[#f6f9f9] border-r border-[#e0e9e9] flex flex-col flex-shrink-0">
    <div class="p-4 flex items-center justify-between border-b border-[#e7eeee]">
      <div><p class="text-[11px] font-semibold text-[#799395]">会话</p><h2 class="font-semibold text-[#23383b] text-[15px] mt-0.5">历史对话</h2></div>
      <button @click="$emit('new-chat')" class="h-8 w-8 rounded-md border border-[#cfe0df] flex items-center justify-center text-[#21656a] hover:bg-[#e8f5f3] bg-white transition-colors" title="新对话">
        <Plus class="h-4 w-4" />
      </button>
    </div>
    <div v-if="sessions.length > 0" class="px-3 py-2">
      <label class="relative block">
        <span class="sr-only">搜索对话</span>
        <Search class="pointer-events-none absolute left-2.5 top-2.5 h-3.5 w-3.5 text-gray-400" />
        <input
          v-model="keyword"
          type="search"
          placeholder="搜索标题或知识库"
          class="w-full rounded-md border border-[#d8e4e4] bg-white py-2 pl-8 pr-2 text-[13px] text-[#385154] focus:border-[#68aaa7] focus:outline-none"
        />
      </label>
    </div>
    <div class="flex-1 overflow-y-auto px-2 pb-4 space-y-1">
      <div v-if="sessions.length === 0" class="text-gray-400 text-xs px-2 py-4 text-center">暂无对话记录</div>
      <div v-else-if="filteredSessions.length === 0" class="text-gray-400 text-xs px-2 py-4 text-center">没有匹配的对话</div>
      <div
        v-for="(session, index) in filteredSessions"
        :key="session.id"
        @click="$emit('select', session.id)"
        class="group rounded-md p-3 cursor-pointer border transition-all relative flex flex-col mb-2"
        :class="[
          colorClass(index).border,
          activeSessionId === session.id
            ? 'bg-[#eaf6f5] border-[#bfe2df] shadow-sm'
            : 'bg-white border-[#e4ebeb] hover:bg-[#f1f8f8]'
        ]"
      >
        <div class="flex items-start justify-between w-full">
          <div class="flex items-start flex-1 min-w-0 pr-6">
            <MessageSquare class="w-3 h-3 mr-3 flex-shrink-0 mt-0.5" :class="colorClass(index).iconText" />
            <div class="flex-1 min-w-0">
              <input
                v-if="editingId === session.id"
                ref="titleInput"
                v-model="draftTitle"
                :maxlength="TITLE_MAX_CHARS"
                aria-label="会话标题"
                class="mb-2 w-full rounded border border-[#68aaa7] bg-white px-1.5 py-1 text-[14px] font-bold text-gray-800 focus:outline-none"
                @click.stop
                @keydown.enter.prevent="!$event.isComposing && $event.keyCode !== 229 && commitRename(session)"
                @keydown.esc.prevent="cancelRename"
                @blur="commitRename(session)"
              />
              <div v-else class="text-[14px] text-gray-800 font-bold truncate mb-2.5 tracking-wide" :title="session.title" @dblclick.stop="startRename(session)">{{ session.title }}</div>
              <div class="text-[12px] text-gray-400 flex items-center gap-3 flex-wrap">
                <span class="flex items-center gap-1.5 truncate" title="所属知识库">
                  <Database class="w-3.5 h-3.5" />{{ session.kb_name }}
                </span>
                <span class="flex items-center gap-1.5 flex-shrink-0">
                  <Calendar class="w-3.5 h-3.5" />{{ session.date }}
                </span>
              </div>
            </div>
          </div>
          <button
            v-if="editingId !== session.id"
            @click.stop="startRename(session)"
            class="absolute right-10 top-4 p-1.5 rounded-lg text-gray-300 opacity-0 transition-all hover:bg-[#e8f5f3] hover:text-[#176d70] focus:opacity-100 group-hover:opacity-100"
            title="重命名对话"
            aria-label="重命名对话"
          >
            <Pencil class="w-3.5 h-3.5" />
          </button>
          <button
            @click.stop="deleteConfirmId = deleteConfirmId === session.id ? null : session.id"
            class="absolute right-3 top-4 transition-all p-1.5 rounded-lg"
            :class="deleteConfirmId === session.id ? 'text-red-500 bg-red-50' : 'text-gray-300 hover:text-red-500 hover:bg-red-50 opacity-0 group-hover:opacity-100'"
            :title="deleteConfirmId === session.id ? '取消删除' : '删除对话'"
          >
            <Trash2 class="w-3.5 h-3.5" />
          </button>
        </div>

        <div v-if="deleteConfirmId === session.id" class="mt-4 bg-white border border-gray-100 shadow-[0_4px_20px_rgba(0,0,0,0.08)] rounded-xl p-3 w-full flex flex-col gap-3 cursor-default" @click.stop>
          <div>
            <h4 class="text-[13px] font-bold text-gray-800">确认删除</h4>
            <p class="text-[11px] text-gray-500 mt-1">确定要删除该记录吗？此操作不可逆。</p>
          </div>
          <div class="flex justify-end gap-2">
            <button @click.stop="deleteConfirmId = null" :disabled="deletingSessionId === session.id" class="px-3 py-1.5 text-xs text-gray-600 bg-gray-50 hover:bg-gray-100 rounded-md disabled:opacity-50">取消</button>
            <button @click.stop="$emit('delete', session.id)" :disabled="deletingSessionId === session.id" class="px-3 py-1.5 text-xs font-bold text-white bg-red-500 hover:bg-red-600 rounded-md disabled:opacity-50">
              {{ deletingSessionId === session.id ? '删除中...' : '确定删除' }}
            </button>
          </div>
        </div>
      </div>
    </div>
  </aside>
</template>

<script setup>
import { computed, nextTick, ref } from 'vue';
import { Calendar, Database, MessageSquare, Pencil, Plus, Search, Trash2 } from 'lucide-vue-next';

// 与后端 SESSION_TITLE_RENAME_MAX_CHARS 保持一致。
const TITLE_MAX_CHARS = 50;

const props = defineProps({
  sessions: { type: Array, default: () => [] },
  activeSessionId: { type: [String, Number], default: null },
  deletingSessionId: { type: [String, Number], default: null }
});
const emit = defineEmits(['new-chat', 'select', 'delete', 'rename']);

const deleteConfirmId = ref(null);
const keyword = ref('');
const editingId = ref(null);
const draftTitle = ref('');
const titleInput = ref(null);

/** 按标题和知识库名过滤会话，忽略大小写与首尾空白。 */
const filteredSessions = computed(() => {
  const needle = keyword.value.trim().toLowerCase();
  if (!needle) return props.sessions;
  return props.sessions.filter(session =>
    `${session.title || ''}\n${session.kb_name || ''}`.toLowerCase().includes(needle)
  );
});

/** 进入行内重命名，并把焦点放到输入框。 */
const startRename = async (session) => {
  deleteConfirmId.value = null;
  editingId.value = session.id;
  draftTitle.value = session.title || '';
  await nextTick();
  const input = [].concat(titleInput.value || [])[0];
  input?.focus();
  input?.select();
};
/** 放弃本次编辑；Esc 之后的失焦不应再提交。 */
const cancelRename = () => {
  editingId.value = null;
};
/** 提交重命名：Enter 与失焦都会走到这里，只有仍处于编辑中且内容有变化时才通知父组件。 */
const commitRename = (session) => {
  if (editingId.value !== session.id) return;
  const title = draftTitle.value.trim();
  editingId.value = null;
  if (title && title !== session.title) emit('rename', { id: session.id, title });
};

const colors = [
  { border: 'border-l-blue-500', iconText: 'text-blue-500' },
  { border: 'border-l-red-500', iconText: 'text-red-500' },
  { border: 'border-l-orange-400', iconText: 'text-orange-400' },
  { border: 'border-l-green-500', iconText: 'text-green-500' }
];
/** 按会话序号循环选择侧边栏强调色。 */
const colorClass = (index) => colors[index % colors.length];
</script>
