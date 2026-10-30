import { apiGet } from '../apiClient.js';

const DEFAULT_DELAYS_MS = [500, 1000, 2000, 4000, 8000, 8000];
// 任务不再变化的终态；其中只有 failed 和 interrupted 需要提醒用户。
const FAILED_STATUSES = new Set(['failed', 'interrupted']);
const TERMINAL_STATUSES = new Set(['completed', 'superseded', ...FAILED_STATUSES]);

/**
 * 轮询知识库的索引任务，直到指定任务进入终态。
 * 返回最终的任务对象；超时或查询失败时返回 null，调用方不应把它当作失败。
 */
export async function waitForIndexJob(kbName, jobId, {
  get = apiGet,
  delays = DEFAULT_DELAYS_MS,
  sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
} = {}) {
  for (const delay of delays) {
    await sleep(delay);
    let jobs;
    try {
      jobs = await get(`/api/kbs/${encodeURIComponent(kbName)}/index_jobs`);
    } catch (_) {
      return null;
    }
    const job = jobs.find(item => item.id === jobId);
    if (job && TERMINAL_STATUSES.has(job.status)) return job;
  }
  return null;
}

/** 判断索引任务是否以失败告终。 */
export const isFailedIndexJob = job => Boolean(job) && FAILED_STATUSES.has(job.status);
