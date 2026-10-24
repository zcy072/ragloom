import { ref } from 'vue';

import { apiDelete, apiGet, apiPost, apiPut } from '../apiClient';
import { filterKnowledgeBases } from '../knowledgeAccess';
import { store } from '../store';
import { toast } from '../toast';
/** 封装知识库列表读取、创建、删除和重命名操作。 */
export function useKnowledgeBases() {
  const knowledgeBases = ref([]);
  const deletingKnowledgeBaseId = ref(null);
  /** 刷新知识库列表，并按当前用户权限过滤。 */
  const fetchKnowledgeBases = async () => {
    try {
      const data = await apiGet('/api/kbs');
      knowledgeBases.value = filterKnowledgeBases(data, store.user);
    } catch (error) {
      console.error('Failed to fetch knowledge bases', error);
    }
  };
  /** 创建知识库，成功后刷新列表并返回 true。 */
  const createKnowledgeBase = async (name) => {
    if (!name.trim()) return false;
    try {
      await apiPost('/api/kbs', { name });
      await fetchKnowledgeBases();
      toast.success('创建成功');
      return true;
    } catch (error) {
      toast.error(`创建失败: ${error.message}`);
      return false;
    }
  };
  /** 删除知识库，并用标识防止同一项目被重复提交。 */
  const deleteKnowledgeBase = async (knowledgeBase) => {
    if (deletingKnowledgeBaseId.value === knowledgeBase.name) return false;
    deletingKnowledgeBaseId.value = knowledgeBase.name;
    try {
      await apiDelete(`/api/kbs/${encodeURIComponent(knowledgeBase.name)}`);
      await fetchKnowledgeBases();
      toast.success('删除成功');
      return true;
    } catch {
      toast.error('删除失败');
      return false;
    } finally {
      deletingKnowledgeBaseId.value = null;
    }
  };
  /** 重命名知识库；名称未变化时直接视为成功。 */
  const renameKnowledgeBase = async (knowledgeBase, nextName) => {
    const name = nextName?.trim();
    if (!name || name === knowledgeBase.name) return true;
    try {
      await apiPut(`/api/kbs/${encodeURIComponent(knowledgeBase.name)}`, { name });
      await fetchKnowledgeBases();
      toast.success('重命名成功');
      return true;
    } catch (error) {
      toast.error(`重命名失败: ${error.message || '未知错误'}`);
      return false;
    }
  };

  return {
    knowledgeBases,
    deletingKnowledgeBaseId,
    fetchKnowledgeBases,
    createKnowledgeBase,
    deleteKnowledgeBase,
    renameKnowledgeBase
  };
}
