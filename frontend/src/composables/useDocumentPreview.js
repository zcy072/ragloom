import { computed, onBeforeUnmount, ref, watch } from 'vue';

import { authorizedFetch } from '../apiClient';
/** 将受保护的文档下载为本地对象 URL，供预览组件安全展示。 */
export function useDocumentPreview(url) {
  // 只缓存不带 #page 片段的 blob 地址；翻页或跳转引用只改片段，不重新下载文件。
  const blobUrl = ref('');
  const fragment = computed(() => {
    const value = String(url.value || '');
    return value.includes('#') ? `#${value.split('#', 2)[1]}` : '';
  });
  const baseUrl = computed(() => String(url.value || '').split('#')[0]);
  const resolvedUrl = computed(() => (blobUrl.value ? `${blobUrl.value}${fragment.value}` : ''));
  const loading = ref(false);
  const error = ref('');
  // 保存当前请求控制器；切换文档时用于取消上一条尚未结束的下载。
  let requestController = null;
  /** 释放浏览器创建的对象 URL，避免多次预览积累内存。 */
  const revokeResolvedUrl = () => {
    if (blobUrl.value.startsWith('blob:')) URL.revokeObjectURL(blobUrl.value);
    blobUrl.value = '';
  };
  /** 中止当前文档下载请求。 */
  const stopRequest = () => {
    requestController?.abort();
    requestController = null;
  };

  watch(baseUrl, async (nextUrl) => {
    stopRequest();
    revokeResolvedUrl();
    error.value = '';
    if (!nextUrl) return;

    const controller = new AbortController();
    requestController = controller;
    loading.value = true;
    try {
      const response = await authorizedFetch(nextUrl, { signal: controller.signal });
      if (!response.ok) throw new Error('文档加载失败');
      const objectUrl = URL.createObjectURL(await response.blob());
      if (requestController !== controller) {
        URL.revokeObjectURL(objectUrl);
        return;
      }
      blobUrl.value = objectUrl;
    } catch (requestError) {
      if (requestError.name !== 'AbortError') {
        error.value = requestError.message || '文档加载失败';
      }
    } finally {
      if (requestController === controller) {
        requestController = null;
        loading.value = false;
      }
    }
  }, { immediate: true });

  onBeforeUnmount(() => {
    stopRequest();
    revokeResolvedUrl();
  });

  return { resolvedUrl, loading, error };
}
