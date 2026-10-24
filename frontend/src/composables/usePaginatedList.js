import { ref } from 'vue';

import { apiGet } from '../apiClient.js';

/**
 * Load one server-paginated endpoint while ignoring stale search and page responses.
 * The optional dependencies make request sequencing testable without browser globals.
 */
export function usePaginatedList(path, dependencies = {}) {
  const get = dependencies.apiGet || apiGet;
  const debounceMs = dependencies.debounceMs ?? 250;
  const onError = dependencies.onError || (error => console.error(`Failed to fetch ${path}`, error));
  const items = ref([]);
  const page = ref(1);
  const pageSize = ref(20);
  const total = ref(0);
  const totalPages = ref(1);
  const loading = ref(false);

  let requestVersion = 0;
  let searchTimer = null;

  const requestUrl = query => {
    const params = new URLSearchParams({
      page: String(page.value),
      page_size: String(pageSize.value)
    });
    if (query?.trim()) params.set('query', query.trim());
    return `${path}?${params}`;
  };

  const fetchPage = async (query = '') => {
    const version = ++requestVersion;
    loading.value = true;
    try {
      const data = await get(requestUrl(query));
      if (version !== requestVersion) return;
      items.value = data.items;
      page.value = data.page;
      total.value = data.total;
      totalPages.value = data.total_pages;
    } catch (error) {
      if (version === requestVersion) onError(error);
    } finally {
      if (version === requestVersion) loading.value = false;
    }
  };

  const scheduleSearch = (query = '') => {
    page.value = 1;
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => fetchPage(query), debounceMs);
  };

  const changePage = (nextPage, query = '') => {
    if (nextPage === page.value || nextPage < 1 || nextPage > totalPages.value) return;
    clearTimeout(searchTimer);
    page.value = nextPage;
    fetchPage(query);
  };

  const changePageSize = (nextPageSize, query = '') => {
    if (nextPageSize === pageSize.value) return;
    clearTimeout(searchTimer);
    pageSize.value = nextPageSize;
    page.value = 1;
    fetchPage(query);
  };

  const invalidate = () => {
    requestVersion += 1;
    loading.value = false;
    clearTimeout(searchTimer);
  };

  return {
    items,
    page,
    pageSize,
    total,
    totalPages,
    loading,
    fetchPage,
    scheduleSearch,
    changePage,
    changePageSize,
    invalidate
  };
}
