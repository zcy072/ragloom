<template>
  <div class="flex-1 overflow-y-auto p-6 bg-[#f6f9f9] w-full text-slate-800">
    <div class="mb-6 flex items-end justify-between">
      <div><p class="text-[11px] font-semibold text-[#649092]">知识资产</p><h2 class="mt-1 text-lg font-semibold text-[#1e3437]">知识库管理</h2></div>
      <span class="text-xs text-[#819497]">{{ kbs.length }} 个知识库</span>
    </div>
    <div v-if="!readonly" class="mb-5 flex items-center space-x-3">
      <button @click="showNew = true" class="bg-[#176d70] hover:bg-[#10585c] text-white px-4 py-2 rounded-md shadow-sm text-sm font-medium flex items-center transition-colors">
        <Plus class="w-4 h-4 mr-1" /> 新建知识库
      </button>
      <div v-if="showNew" class="flex items-center space-x-2 animate-fade-in-right">
        <input v-model="newKbName" maxlength="80" @keydown.enter="!$event.isComposing && $event.keyCode !== 229 && handleCreate()" type="text" placeholder="知识库名称..." class="border border-[#d3e1e1] rounded-md px-3 py-1.5 text-sm focus:outline-none focus:border-[#65aaa6]" />
        <button @click="handleCreate" class="bg-[#176d70] text-white px-3 py-1.5 rounded-md text-sm font-medium">确认</button>
        <button @click="showNew = false" class="text-gray-400 text-sm hover:text-gray-600">取消</button>
      </div>
    </div>

    <div class="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
      <div v-for="kb in kbs" :key="kb.name" class="bg-white border border-[#e0e9e9] rounded-lg p-5 shadow-[0_1px_2px_rgba(18,53,57,0.04)] flex flex-col relative group hover:border-[#91c8c4] transition-colors">
        <div class="absolute top-4 right-4 bg-gray-100 text-gray-500 text-[10px] px-2 py-0.5 rounded">私有</div>
        <div class="mb-3"><div class="w-10 h-10 rounded-lg bg-[#eaf6f5] flex items-center justify-center"><FolderOpen class="text-[#278179] w-5 h-5" /></div></div>
        <div class="font-semibold text-[15px] text-gray-800 mb-1">{{ kb.name }}</div>
        <div class="text-xs text-gray-400 mb-6">{{ kb.description || '暂无描述' }}</div>
        <div class="mt-auto">
          <div class="flex items-center justify-between text-[11px] text-gray-400 mb-3">
            <span>{{ kb.docs }}个文档</span>
            <span>{{ kb.date }}</span>
          </div>
          <div class="flex items-center space-x-2 relative">
            <button @click="setManageKb(kb)" class="bg-white hover:bg-[#f3f8f8] text-[#4d6669] border border-[#d8e4e4] px-3 py-1.5 rounded-md text-[12px] font-medium transition-colors">{{ readonly ? '查看文档' : '管理文档' }}</button>
            <button @click="startChat(kb)" class="bg-[#176d70] hover:bg-[#10585c] text-white px-3 py-1.5 rounded-md text-[12px] font-medium transition-colors">开始问答</button>
            <div v-if="!readonly && kb.name !== '通用知识库'" class="relative ml-auto">
              <button @click="actionMenuId = actionMenuId === kb.name ? null : kb.name" class="h-[28px] w-[28px] flex items-center justify-center bg-gray-50 hover:bg-gray-100 text-gray-500 border border-gray-200 rounded transition-colors" title="更多操作"><MoreHorizontal class="w-3 h-3" /></button>
              <div v-if="actionMenuId === kb.name" class="absolute right-0 top-8 w-40 bg-white border border-gray-200 rounded-md shadow-lg py-1 z-10 text-[12px]">
                <div v-if="renameKbId === kb.name" class="px-3 py-2 flex flex-col space-y-2">
                  <input v-focus v-model="renameValue" maxlength="80" @keydown.enter="!$event.isComposing && $event.keyCode !== 229 && handleRenameKb(kb, renameValue)" @keydown.esc="renameKbId = null" type="text" class="w-full border border-gray-300 rounded px-2 py-1 text-xs focus:outline-none focus:border-blue-500" />
                  <div class="flex space-x-2">
                    <button @click.stop="handleRenameKb(kb, renameValue)" class="text-blue-600 font-bold hover:underline">保存</button>
                    <button @click.stop="renameKbId = null" class="text-gray-500 hover:underline">取消</button>
                  </div>
                </div>
                <button v-else @click.stop="renameKbId = kb.name; renameValue = kb.name" class="w-full text-left px-3 py-1.5 hover:bg-gray-50 text-gray-700">重命名</button>
                <div v-if="deleteKbConfirmId === kb.name" @click.stop class="absolute right-full top-0 mr-2 bg-white border border-gray-100 shadow-[0_10px_30px_rgba(0,0,0,0.1)] rounded-lg p-3 z-20 w-56 flex flex-col gap-3 cursor-default">
                  <div class="flex items-start gap-2.5">
                    <div class="bg-red-50 p-1.5 rounded-full text-red-500 flex-shrink-0 mt-0.5">
                      <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" /></svg>
                    </div>
                    <div><h4 class="text-[13px] font-semibold text-gray-800 leading-tight">删除知识库</h4><p class="text-[11px] text-gray-500 mt-1 leading-relaxed">确定要删除该知识库及其所有文档吗？</p></div>
                  </div>
                  <div class="flex justify-end gap-2 mt-1">
                    <button @click.stop="deleteKbConfirmId = null" :disabled="deletingKbId === kb.name" class="px-3 py-1.5 text-xs font-medium text-gray-600 bg-gray-50 hover:bg-gray-100 rounded-md transition-colors disabled:opacity-50">取消</button>
                    <button @click.stop="confirmDeleteKb(kb)" :disabled="deletingKbId === kb.name" :class="['px-3 py-1.5 text-xs font-medium text-white rounded-md transition-colors shadow-sm', deletingKbId === kb.name ? 'bg-red-400 cursor-not-allowed' : 'bg-red-500 hover:bg-red-600']">{{ deletingKbId === kb.name ? '删除中...' : '确定删除' }}</button>
                  </div>
                </div>
                <button v-else @click.stop="deleteKbConfirmId = kb.name" class="w-full text-left px-3 py-1.5 hover:bg-red-50 text-red-600">删除</button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div v-if="manageKb" class="fixed inset-0 bg-black/40 flex items-center justify-center z-50 animate-fade-in">
      <div class="bg-white rounded-xl shadow-xl w-[calc(100vw-2rem)] max-w-[800px] max-h-[90vh] h-[85vh] flex flex-col relative overflow-hidden">
        <div class="px-6 py-4 border-b border-gray-100 flex items-center justify-between relative z-10 bg-white">
          <h3 class="font-semibold text-gray-800 flex items-center"><FolderOpen class="w-5 h-5 text-blue-500 mr-2" />{{ readonly ? '查看文档' : '管理文档' }} - {{ manageKb.name }}</h3>
          <button @click="setManageKb(null)" class="text-gray-400 hover:text-gray-600 font-bold text-xl" title="关闭">&times;</button>
        </div>
        <div class="p-6 flex-1 overflow-y-auto relative z-10 bg-white">
          <KnowledgeBaseConfigPanel
            v-if="!readonly"
            v-model:chunk-config="chunkConfig"
            v-model:retrieval-config="retrievalConfig"
            v-model:show-chunk-config="showChunkConfig"
            v-model:show-retrieval-config="showRetrievalConfig"
            :saving-chunk="savingChunk"
            :rebuilding-index="rebuildingIndex"
            @rebuild="rebuildKnowledgeBase"
            @reset-chunk="resetChunkConfig"
            @save-chunk="saveChunkConfig"
            @save-retrieval="saveRetrievalConfig"
          />
          <KnowledgeBaseDocumentList
            v-model:upload-metadata="uploadMetadata"
            v-model:upload-parser-mode="uploadParserMode"
            v-model:upload-image-parser-mode="uploadImageParserMode"
            v-model:delete-confirm-id="deleteDocConfirmId"
            :readonly="readonly"
            :documents="docs"
            :is-uploading="isUploading"
            :deleting-document-id="deletingDocId"
            :parser-options="pdfParserOptions"
            :image-parser-options="imageParserOptions"
            :upload-accept="uploadAccept"
            :is-selectable-parser-document="isSelectableParserDocument"
            :parser-label="pdfParserLabel"
            @upload="handleUpload"
            @preview="openPreview"
            @edit-metadata="startMetadataEdit"
            @edit-parser="startParserEdit"
            @delete="confirmDeleteDoc"
          />
        </div>
        <div class="px-6 py-4 border-t border-gray-100 bg-gray-50 flex justify-end rounded-b-xl relative z-10">
          <button @click="setManageKb(null)" class="bg-white border border-gray-200 text-gray-600 px-4 py-2 rounded text-sm hover:bg-gray-100 transition-colors">关闭</button>
        </div>
      </div>
    </div>

    <KnowledgeBaseDocumentDialogs
      :metadata-document="editingMetadataDoc"
      v-model:metadata="editingMetadata"
      :saving-metadata="savingMetadata"
      :parser-document="editingParserDoc"
      v-model:parser-mode="editingParserMode"
      :parser-options="editingParserDoc ? parserOptionsFor(editingParserDoc.filename) : []"
      :saving-parser="savingParser"
      @cancel-metadata="cancelMetadataEdit"
      @save-metadata="saveDocumentMetadata"
      @cancel-parser="cancelParserEdit"
      @save-parser="saveDocumentParser"
    />

    <DocumentPreview
      v-if="showPreview && previewDoc && manageKb"
      :filename="previewDoc.filename"
      :url="`/api/documents/${encodeURIComponent(manageKb.name)}/${encodeURIComponent(previewDoc.filename)}?preview=true`"
      @close="closePreview"
    />
  </div>
</template>

<script setup>
import { onMounted, ref, watch } from 'vue';
import { FolderOpen, MoreHorizontal, Plus } from 'lucide-vue-next';

import DocumentPreview from '../components/DocumentPreview.vue';
import KnowledgeBaseConfigPanel from '../components/KnowledgeBaseConfigPanel.vue';
import KnowledgeBaseDocumentDialogs from '../components/KnowledgeBaseDocumentDialogs.vue';
import KnowledgeBaseDocumentList from '../components/KnowledgeBaseDocumentList.vue';
import { useDocuments } from '../composables/useDocuments';
import { useKnowledgeBases } from '../composables/useKnowledgeBases';
import { store } from '../store';

defineProps({
  readonly: { type: Boolean, default: false }
});
const vFocus = { mounted: element => element.focus() };
const showNew = ref(false);
const newKbName = ref('');
const deleteKbConfirmId = ref(null);
const renameKbId = ref(null);
const renameValue = ref('');
const deleteDocConfirmId = ref(null);
const actionMenuId = ref(null);
const previewDoc = ref(null);
const showPreview = ref(false);

const {
  knowledgeBases: kbs,
  deletingKnowledgeBaseId: deletingKbId,
  fetchKnowledgeBases: fetchKbs,
  createKnowledgeBase,
  deleteKnowledgeBase,
  renameKnowledgeBase
} = useKnowledgeBases();
/** 关闭文档预览并清空当前预览对象。 */
const closePreview = () => {
  showPreview.value = false;
  previewDoc.value = null;
};

const {
  manageKb,
  documents: docs,
  isUploading,
  deletingDocumentId: deletingDocId,
  showChunkConfig,
  chunkConfig,
  savingChunk,
  rebuildingIndex,
  showRetrievalConfig,
  retrievalConfig,
  uploadMetadata,
  documentParserOptions: pdfParserOptions,
  imageParserOptions,
  uploadAccept,
  uploadParserMode,
  uploadImageParserMode,
  editingMetadataDoc,
  editingMetadata,
  savingMetadata,
  editingParserDoc,
  editingParserMode,
  savingParser,
  initializeDocumentCapabilities,
  setManageKnowledgeBase: setManageKb,
  handleUpload,
  resetChunkConfig,
  saveChunkConfig,
  rebuildKnowledgeBase,
  saveRetrievalConfig,
  isSelectableParserDocument,
  parserOptionsFor,
  parserLabel: pdfParserLabel,
  startMetadataEdit,
  cancelMetadataEdit,
  saveDocumentMetadata,
  startParserEdit,
  cancelParserEdit,
  saveDocumentParser,
  deleteDocument
} = useDocuments({
  onKnowledgeBasesChanged: fetchKbs,
  onDocumentDeleted: filename => {
    if (previewDoc.value?.filename === filename) closePreview();
  }
});
/** 创建输入框中的知识库，成功后重置新增表单。 */
const handleCreate = async () => {
  if (await createKnowledgeBase(newKbName.value)) {
    showNew.value = false;
    newKbName.value = '';
  }
};
/** 确认删除知识库，并清理操作菜单状态。 */
const confirmDeleteKb = async knowledgeBase => {
  if (await deleteKnowledgeBase(knowledgeBase)) {
    deleteKbConfirmId.value = null;
    actionMenuId.value = null;
  }
};
/** 提交知识库重命名，并在成功后退出编辑状态。 */
const handleRenameKb = async (knowledgeBase, nextName) => {
  if (!nextName?.trim() || nextName.trim() === knowledgeBase.name) {
    renameKbId.value = null;
    return;
  }
  if (await renameKnowledgeBase(knowledgeBase, nextName)) {
    renameKbId.value = null;
    actionMenuId.value = null;
  }
};
/** 确认删除文档，并关闭对应确认提示。 */
const confirmDeleteDoc = async filename => {
  if (await deleteDocument(filename)) deleteDocConfirmId.value = null;
};
/** 进入问答页面，并把当前知识库设为聊天默认值。 */
const startChat = knowledgeBase => {
  store.chatDefaultKb = knowledgeBase.name;
  store.activeMenu = '问答对话';
};
/** 打开指定文档的预览面板。 */
const openPreview = document => {
  previewDoc.value = document;
  showPreview.value = true;
};
/** 响应仪表盘跳转请求，直接打开目标知识库。 */
const openTargetKnowledgeBase = target => {
  if (!target) return;
  setManageKb(target);
  store.targetKb = null;
};

onMounted(() => {
  fetchKbs();
  initializeDocumentCapabilities();
  openTargetKnowledgeBase(store.targetKb);
});

watch(() => store.targetKb, openTargetKnowledgeBase);
</script>
