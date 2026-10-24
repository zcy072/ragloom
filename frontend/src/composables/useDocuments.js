import { computed, onUnmounted, ref, watch } from 'vue';

import { apiDelete, apiGet, apiPost, apiPut, apiRequest } from '../apiClient';
import { toast } from '../toast';
// 与后端 services/documents/use_cases.py 的上传限制保持一致。
const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
const MAX_UPLOAD_FILENAME_BYTES = 200;
const POLL_BASE_MS = 2000;
const POLL_MAX_MS = 16000;
/** 判断文档是否还在后端索引中。 */
const isProcessing = document => document.status === 'processing';
/** 返回知识库默认的分块长度和重叠长度。 */
const defaultChunkConfig = () => ({ chunk_size: 800, chunk_overlap: 150 });
// 上传时优先选择 MinerU；能力探测失败时回退到本地文本解析。
const preferredUploadParserMode = 'mineru_cloud';
const fallbackUploadParserMode = 'native_text';
const preferredImageParserMode = 'vision_model';
const fallbackImageParserMode = 'paddle_ocr';
const defaultParserOptions = [
  { id: 'native_text', label: '本地文本解析', available: true, extensions: ['.pdf', '.docx'] },
  { id: 'mineru_cloud', label: 'MinerU 云端解析', available: false, extensions: ['.pdf', '.docx'] },
  { id: 'paddle_ocr', label: 'Paddle 本地 OCR', available: true, extensions: ['.pdf', '.png', '.jpg', '.jpeg'] },
  { id: 'vision_model', label: '智能视觉解析（失败时 Paddle OCR）', available: true, extensions: ['.png', '.jpg', '.jpeg'] }
];
/** 返回一份新的空文档元数据，避免多个表单共享同一对象。 */
const emptyMetadata = () => ({ document_type: '', tags: '', effective_date: '' });
/** 管理知识库文档、解析方式、元数据和索引配置等页面状态。 */
export function useDocuments({ onKnowledgeBasesChanged, onDocumentDeleted } = {}) {
  const manageKb = ref(null);
  const documents = ref([]);
  const isUploading = ref(false);
  const deletingDocumentId = ref(null);
  const showChunkConfig = ref(false);
  const chunkConfig = ref(defaultChunkConfig());
  const savingChunk = ref(false);
  const rebuildingIndex = ref(false);
  const showRetrievalConfig = ref(false);
  const retrievalConfig = ref({ query_rewrite_enabled: true });
  const uploadMetadata = ref(emptyMetadata());
  const parserOptions = ref([...defaultParserOptions]);
  const documentParserOptions = computed(() => parserOptions.value.filter(option =>
    option.extensions?.some(extension => ['.pdf', '.docx'].includes(extension))
  ));
  const imageParserOptions = computed(() => parserOptions.value.filter(option =>
    option.extensions?.some(extension => ['.png', '.jpg', '.jpeg'].includes(extension))
  ));
  const uploadAccept = ref('.pdf,.docx,.md,.txt,.png,.jpg,.jpeg');
  const uploadParserMode = ref(preferredUploadParserMode);
  const uploadImageParserMode = ref(preferredImageParserMode);
  const editingMetadataDoc = ref(null);
  const editingMetadata = ref(emptyMetadata());
  const savingMetadata = ref(false);
  const editingParserDoc = ref(null);
  const editingParserMode = ref('native_text');
  const savingParser = ref(false);
  // 仅在存在处理中文档时轮询，离开页面或任务结束后立即清除。
  let pollingTimer = null;
  let pollFailures = 0;
  // 请求版本用于丢弃切换知识库或写操作之前发出的过期响应。
  let documentRequestVersion = 0;
  // 轮询请求还没返回时不再叠加新的轮询，慢响应因此不会被后发的请求作废。
  let documentRequestInFlight = false;
  /** 将表单元数据整理为后端接口需要的格式。 */
  const metadataPayload = metadata => ({
    document_type: metadata.document_type?.trim() || '',
    tags: (metadata.tags || '').split(/[，,]/).map(tag => tag.trim()).filter(Boolean),
    effective_date: metadata.effective_date || ''
  });
  /** 从后端读取支持的文件格式，并更新文件选择器限制。 */
  const fetchDocumentFormats = async () => {
    try {
      const formats = await apiGet('/api/document_formats');
      const extensions = formats.map(format => format.extension).filter(Boolean);
      if (extensions.length) uploadAccept.value = extensions.join(',');
    } catch (_) {
      // Keep the core-format fallback when capability discovery is unavailable.
    }
  };
  /** 探测可用解析器，并选择当前环境中可用的默认方式。 */
  const fetchParserOptions = async () => {
    try {
      const data = await apiGet('/api/document_parsers');
      const options = data.document || data.pdf;
      if (!Array.isArray(options) || !options.length) {
        uploadParserMode.value = fallbackUploadParserMode;
        uploadImageParserMode.value = fallbackImageParserMode;
        return;
      }
      parserOptions.value = options;
      if (!options.some(option => option.id === uploadParserMode.value && option.available)) {
        uploadParserMode.value = options.some(
          option => option.id === preferredUploadParserMode && option.available
        )
          ? preferredUploadParserMode
          : options.find(option => option.id === fallbackUploadParserMode && option.available)?.id
            || options.find(option => option.available)?.id
            || fallbackUploadParserMode;
      }
      if (!options.some(option => option.id === uploadImageParserMode.value && option.available && option.extensions?.some(extension => ['.png', '.jpg', '.jpeg'].includes(extension)))) {
        uploadImageParserMode.value = options.some(option => option.id === preferredImageParserMode && option.available)
          ? preferredImageParserMode
          : options.find(option => option.id === fallbackImageParserMode && option.available)?.id
            || fallbackImageParserMode;
      }
    } catch (_) {
      // Native extraction and local Paddle remain available without discovery.
      uploadParserMode.value = fallbackUploadParserMode;
      uploadImageParserMode.value = fallbackImageParserMode;
    }
  };
  /** 并行触发文档格式和解析器能力探测。 */
  const initializeDocumentCapabilities = () => {
    fetchParserOptions();
    fetchDocumentFormats();
  };
  /**
   * 刷新当前知识库的文档列表。
   * 轮询调用（poll）在上一次请求未返回时直接跳过；显式刷新会作废仍在途中的请求，
   * 避免写操作之前发出的旧响应覆盖新数据。返回本次请求是否成功。
   */
  const fetchDocuments = async ({ poll = false } = {}) => {
    const kbName = manageKb.value?.name;
    if (!kbName) return true;
    if (poll && documentRequestInFlight) return true;
    const requestVersion = ++documentRequestVersion;
    documentRequestInFlight = true;
    try {
      const data = await apiGet(`/api/documents/${encodeURIComponent(kbName)}`);
      if (requestVersion === documentRequestVersion && manageKb.value?.name === kbName) {
        documents.value = data;
      }
      return true;
    } catch (error) {
      console.error('Failed to fetch documents', error);
      return false;
    } finally {
      if (requestVersion === documentRequestVersion) documentRequestInFlight = false;
    }
  };
  /** 切换当前管理的知识库，并加载其分块与检索配置。 */
  const setManageKnowledgeBase = (knowledgeBase) => {
    manageKb.value = knowledgeBase;
    documentRequestVersion += 1;
    documentRequestInFlight = false;
    if (!knowledgeBase) {
      documents.value = [];
      stopPolling();
      return;
    }

    fetchDocuments();
    apiGet(`/api/kbs/${encodeURIComponent(knowledgeBase.name)}/chunk_config`)
      .then(data => {
        if (manageKb.value?.name !== knowledgeBase.name) return;
        chunkConfig.value = {
          chunk_size: data.chunk_size,
          chunk_overlap: data.chunk_overlap,
          index_needs_rebuild: Boolean(data.index_needs_rebuild)
        };
      })
      .catch(() => {});
    apiGet(`/api/kbs/${encodeURIComponent(knowledgeBase.name)}/retrieval_config`)
      .then(data => {
        if (manageKb.value?.name === knowledgeBase.name) {
          retrievalConfig.value = { query_rewrite_enabled: Boolean(data.query_rewrite_enabled) };
        }
      })
      .catch(() => {});
  };

  /** 停止文档状态轮询并清零失败计数。 */
  const stopPolling = () => {
    clearTimeout(pollingTimer);
    pollingTimer = null;
    pollFailures = 0;
  };
  /** 按失败次数指数退避安排下一次轮询：2 秒起，最长 16 秒。 */
  const schedulePoll = () => {
    clearTimeout(pollingTimer);
    const delay = Math.min(POLL_BASE_MS * 2 ** pollFailures, POLL_MAX_MS);
    pollingTimer = setTimeout(async () => {
      const succeeded = await fetchDocuments({ poll: true });
      pollFailures = succeeded ? 0 : pollFailures + 1;
      // 回调返回后 documents 可能已经没有处理中文档，此时 watch 已停止轮询。
      if (pollingTimer !== null && documents.value.some(isProcessing)) schedulePoll();
    }, delay);
  };

  watch(documents, nextDocuments => {
    // Poll only while the backend has durable indexing work in progress.
    if (!nextDocuments.some(isProcessing)) {
      stopPolling();
    } else if (pollingTimer === null) {
      schedulePoll();
    }
  }, { deep: true });

  onUnmounted(stopPolling);
  /** 上传前在本地检查大小和文件名长度，返回不能上传的原因；可以上传则返回空字符串。 */
  const precheckUpload = file => {
    if (file.size > MAX_UPLOAD_BYTES) return '超过 50 MB';
    if (new TextEncoder().encode(file.name).length > MAX_UPLOAD_FILENAME_BYTES) {
      return `文件名超过 ${MAX_UPLOAD_FILENAME_BYTES} 字节（约 66 个汉字）`;
    }
    return '';
  };
  /** 逐个上传用户选择的文件；单个文件失败不会中断后面的文件，结束后汇总提示。 */
  const handleUpload = async (event) => {
    const files = [...(event.target.files || [])];
    if (!files.length || !manageKb.value) return;
    isUploading.value = true;
    const failures = [];
    let uploaded = 0;
    try {
      const kbName = encodeURIComponent(manageKb.value.name);
      const metadata = JSON.stringify(metadataPayload(uploadMetadata.value));
      for (const file of files) {
        const rejected = precheckUpload(file);
        if (rejected) {
          failures.push(`${file.name}：${rejected}`);
          continue;
        }
        const formData = new FormData();
        formData.append('file', file);
        formData.append('metadata', metadata);
        formData.append('parser_mode', parserModeForFile(file.name));
        try {
          await apiRequest(`/api/documents/${kbName}`, { method: 'POST', body: formData });
          uploaded += 1;
        } catch (error) {
          failures.push(`${file.name}：${error.message || '上传失败'}`);
        }
      }
      if (uploaded) {
        await fetchDocuments();
        await onKnowledgeBasesChanged?.();
        uploadMetadata.value = emptyMetadata();
      }
      if (!failures.length) {
        toast.success('上传成功');
      } else if (!uploaded) {
        toast.error(failures.length === 1 ? failures[0] : `全部 ${failures.length} 个文件上传失败：${failures.join('；')}`);
      } else {
        toast.error(`${uploaded} 个文件已上传，${failures.length} 个失败：${failures.join('；')}`);
      }
    } finally {
      isUploading.value = false;
      event.target.value = '';
    }
  };
  /** 将分块参数恢复为默认值，同时保留是否需要重建的状态。 */
  const resetChunkConfig = () => {
    chunkConfig.value = {
      ...defaultChunkConfig(),
      index_needs_rebuild: chunkConfig.value.index_needs_rebuild
    };
  };
  /** 保存分块参数；后端会按新配置创建完整索引重建任务。 */
  const saveChunkConfig = async () => {
    if (!manageKb.value) return;
    savingChunk.value = true;
    try {
      await apiPut(`/api/kbs/${encodeURIComponent(manageKb.value.name)}/chunk_config`, {
        chunk_size: Number(chunkConfig.value.chunk_size),
        chunk_overlap: Number(chunkConfig.value.chunk_overlap)
      });
      await fetchDocuments();
      chunkConfig.value.index_needs_rebuild = false;
      toast.success('分段配置已保存，正在按新配置重建全部索引');
      showChunkConfig.value = false;
    } catch (_) {
      toast.error('保存失败');
    } finally {
      savingChunk.value = false;
    }
  };
  /** 手动提交当前知识库的完整索引重建任务。 */
  const rebuildKnowledgeBase = async () => {
    if (!manageKb.value || rebuildingIndex.value) return;
    rebuildingIndex.value = true;
    try {
      await apiPost(`/api/kbs/${encodeURIComponent(manageKb.value.name)}/rebuild`);
      chunkConfig.value.index_needs_rebuild = false;
      await fetchDocuments();
      toast.success('索引重建已开始');
    } catch (error) {
      toast.error(error.message || '索引重建失败');
    } finally {
      rebuildingIndex.value = false;
    }
  };
  /** 保存查询改写开关，失败时恢复界面上的原值。 */
  const saveRetrievalConfig = async () => {
    if (!manageKb.value) return;
    const enabled = retrievalConfig.value.query_rewrite_enabled;
    try {
      await apiPut(`/api/kbs/${encodeURIComponent(manageKb.value.name)}/retrieval_config`, {
        query_rewrite_enabled: enabled
      });
    } catch (_) {
      retrievalConfig.value.query_rewrite_enabled = !enabled;
      toast.error('检索设置保存失败');
    }
  };
  /** 判断文档类型是否允许用户切换解析器。 */
  const isSelectableParserDocument = document => /\.(pdf|docx|png|jpe?g)$/i.test(document?.filename || '');
  /** 上传时按真实文件类型选择对应解析器，避免图片继承 PDF/DOCX 的设置。 */
  const parserModeForFile = filename => /\.(png|jpe?g)$/i.test(filename || '')
    ? uploadImageParserMode.value
    : uploadParserMode.value;
  /** 返回与指定文件扩展名匹配的解析器选项。 */
  const parserOptionsFor = filename => {
    const extension = `.${String(filename || '').split('.').pop().toLowerCase()}`;
    return parserOptions.value.filter(option => !option.extensions || option.extensions.includes(extension));
  };
  /** 将解析器内部标识转换为界面显示名称。 */
  const parserLabel = mode => parserOptions.value.find(option => option.id === mode)?.label || '本地文本解析';
  /** 打开元数据编辑状态，并复制当前文档值到表单。 */
  const startMetadataEdit = document => {
    editingMetadataDoc.value = document;
    editingMetadata.value = {
      document_type: document.document_type || '',
      tags: (document.tags || []).join(', '),
      effective_date: document.effective_date || ''
    };
  };
  /** 在没有保存操作进行时关闭元数据编辑状态。 */
  const cancelMetadataEdit = () => {
    if (!savingMetadata.value) editingMetadataDoc.value = null;
  };
  /** 保存由业务数据库维护的文档筛选元数据。 */
  const saveDocumentMetadata = async () => {
    if (!manageKb.value || !editingMetadataDoc.value || savingMetadata.value) return;
    savingMetadata.value = true;
    try {
      await apiPut(
        `/api/documents/${encodeURIComponent(manageKb.value.name)}/${encodeURIComponent(editingMetadataDoc.value.filename)}/metadata`,
        metadataPayload(editingMetadata.value)
      );
      editingMetadataDoc.value = null;
      await fetchDocuments();
      toast.success('文档元数据已保存');
    } catch (error) {
      toast.error(error.message || '文档元数据保存失败');
    } finally {
      savingMetadata.value = false;
    }
  };
  /** 打开解析器编辑状态，并带入文档当前解析方式。 */
  const startParserEdit = document => {
    if (!isSelectableParserDocument(document)) return;
    editingParserDoc.value = document;
    editingParserMode.value = document.parser_mode || 'native_text';
  };
  /** 在没有保存操作进行时关闭解析器编辑状态。 */
  const cancelParserEdit = () => {
    if (!savingParser.value) editingParserDoc.value = null;
  };
  /** 保存新的解析方式，并触发后端重新解析和索引该文档。 */
  const saveDocumentParser = async () => {
    if (!manageKb.value || !editingParserDoc.value || savingParser.value) return;
    savingParser.value = true;
    try {
      await apiPut(
        `/api/documents/${encodeURIComponent(manageKb.value.name)}/${encodeURIComponent(editingParserDoc.value.filename)}/parser`,
        { parser_mode: editingParserMode.value }
      );
      editingParserDoc.value = null;
      await fetchDocuments();
      toast.success('已提交重新解析任务');
    } catch (error) {
      toast.error(error.message || '重新解析失败');
    } finally {
      savingParser.value = false;
    }
  };
  /** 删除当前知识库中的文档，并同步刷新相关统计。 */
  const deleteDocument = async (filename) => {
    if (!manageKb.value || deletingDocumentId.value === filename) return false;
    deletingDocumentId.value = filename;
    try {
      await apiDelete(`/api/documents/${encodeURIComponent(manageKb.value.name)}/${encodeURIComponent(filename)}`);
      onDocumentDeleted?.(filename);
      await fetchDocuments();
      await onKnowledgeBasesChanged?.();
      toast.success('删除成功');
      return true;
    } catch (_) {
      toast.error('删除失败');
      return false;
    } finally {
      deletingDocumentId.value = null;
    }
  };

  return {
    manageKb,
    documents,
    isUploading,
    deletingDocumentId,
    showChunkConfig,
    chunkConfig,
    savingChunk,
    rebuildingIndex,
    showRetrievalConfig,
    retrievalConfig,
    uploadMetadata,
    parserOptions,
    documentParserOptions,
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
    setManageKnowledgeBase,
    fetchDocuments,
    handleUpload,
    resetChunkConfig,
    saveChunkConfig,
    rebuildKnowledgeBase,
    saveRetrievalConfig,
    isSelectableParserDocument,
    parserOptionsFor,
    parserLabel,
    startMetadataEdit,
    cancelMetadataEdit,
    saveDocumentMetadata,
    startParserEdit,
    cancelParserEdit,
    saveDocumentParser,
    deleteDocument
  };
}
