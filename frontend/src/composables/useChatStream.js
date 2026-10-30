import { getCurrentInstance, onBeforeUnmount, ref } from 'vue';

import { authorizedFetch } from '../apiClient.js';
import { store } from '../store.js';
import { toast } from '../toast.js';

const MAX_CHAT_IMAGE_BYTES = 5 * 1024 * 1024;
// 与后端 services/chat/image.py 的 SUPPORTED_CHAT_IMAGE_MIMES 保持一致。
const CHAT_IMAGE_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
// 与后端 ChatRequest.question 的 max_length 保持一致。
export const MAX_QUESTION_CHARS = 4000;
// 后端每 15 秒至少发一次心跳；连续这么久没有收到任何数据就认为连接已静默中断。
const STREAM_IDLE_TIMEOUT_MS = 45000;
// 流式回答写入界面的最小间隔，避免每个 token 都触发整段 Markdown 重新渲染。
const ANSWER_RENDER_INTERVAL_MS = 60;
/** 读取下一个分片；超过空闲时间仍无数据时中止请求并抛出 StreamIdleError。 */
const readWithIdleTimeout = (reader, timeoutMs, controller) => new Promise((resolve, reject) => {
  const timer = setTimeout(() => {
    controller.abort();
    reject(new StreamIdleError());
  }, timeoutMs);
  reader.read().then(
    result => { clearTimeout(timer); resolve(result); },
    error => { clearTimeout(timer); reject(error); }
  );
});
/** 回答流长时间没有任何数据（含心跳）。 */
class StreamIdleError extends Error {}
/** 管理问题提交、图片附件和 SSE 流式回答的完整生命周期。 */
export function useChatStream({
  messages,
  selectedKb,
  selectedModel,
  modelConfigured,
  hasRetrievalFilters,
  retrievalFilterPayload,
  newSessionPending,
  loadSessions,
  webSearch = null
}, dependencies = {}) {
  const fetchChat = dependencies.authorizedFetch || authorizedFetch;
  const chatStore = dependencies.store || store;
  const notifications = dependencies.toast || toast;
  const idleTimeoutMs = dependencies.idleTimeoutMs ?? STREAM_IDLE_TIMEOUT_MS;
  const input = ref('');
  const loading = ref(false);
  // 本轮已完成的处理阶段；回答出现前显示进度，结束后随消息保存。
  const liveStages = ref([]);
  const attachedImage = ref(null);
  // 已发送消息仍在使用的预览地址；组件卸载时统一释放。
  const sentPreviewUrls = new Set();
  // 同一时间只允许一个回答请求；控制器同时承担“停止生成”的职责。
  let abortController = null;
  /** 清除聊天附件，并按需释放本地预览地址。 */
  const removeAttachedImage = (revokePreview = true) => {
    if (revokePreview && attachedImage.value?.previewUrl) {
      URL.revokeObjectURL(attachedImage.value.previewUrl);
    }
    attachedImage.value = null;
  };
  /** 校验并读取图片，将其转换为预览地址和接口所需的 Base64。 */
  const attachImage = (originalFile) => {
    if (!CHAT_IMAGE_MIME_TYPES.includes(originalFile?.type)) {
      notifications.error('仅支持 JPG、PNG、WebP 和 GIF 图片');
      return;
    }
    if (originalFile.size > MAX_CHAT_IMAGE_BYTES) {
      notifications.error('聊天图片不能超过 5 MB');
      return;
    }

    // Object URLs are used only for local preview; the API receives the base64 payload.
    removeAttachedImage();
    const file = originalFile.name
      ? originalFile
      : new File([originalFile], '粘贴图片.png', { type: originalFile.type });
    const previewUrl = URL.createObjectURL(file);
    const reader = new FileReader();
    reader.onload = event => {
      const [header, base64] = event.target.result.split(',');
      const mime = header.match(/:(.*?);/)?.[1] || 'image/jpeg';
      attachedImage.value = { file, previewUrl, base64, mime };
    };
    reader.onerror = () => {
      URL.revokeObjectURL(previewUrl);
      notifications.error('图片读取失败，请重新选择');
    };
    reader.readAsDataURL(file);
  };
  /** 处理文件选择框产生的图片附件。 */
  const handleImageSelect = (event) => {
    const file = event.target.files?.[0];
    if (file) attachImage(file);
    event.target.value = '';
  };
  /** 从剪贴板中提取第一张图片作为聊天附件。 */
  const handleImagePaste = (event) => {
    if (loading.value) return;
    const imageItem = [...(event.clipboardData?.items || [])]
      .find(item => item.type.startsWith('image/'));
    const image = imageItem?.getAsFile();
    if (!image) return;
    event.preventDefault();
    attachImage(image);
  };
  /** 主动中止当前流式回答。 */
  const stop = () => {
    abortController?.abort();
    abortController = null;
    loading.value = false;
  };
  /** 提交问题并逐个解析 SSE 事件，将回答和来源实时写入消息列表。 */
  const send = async () => {
    if ((!input.value.trim() && !attachedImage.value) || loading.value) return;

    const questionText = input.value.trim() || '请描述并分析这张图片。';
    messages.value.push({
      role: 'user',
      content: questionText,
      imageUrl: attachedImage.value?.previewUrl || null
    });

    const imageBase64 = attachedImage.value?.base64 || null;
    const imageMime = attachedImage.value?.mime || null;
    input.value = '';
    if (attachedImage.value?.previewUrl) sentPreviewUrls.add(attachedImage.value.previewUrl);
    removeAttachedImage(false);
    loading.value = true;

    const controller = new AbortController();
    abortController = controller;
    const requestMessages = messages.value;
    const isCurrent = () => abortController === controller && !controller.signal.aborted && messages.value === requestMessages;
    try {
      const payload = {
        question: questionText,
        kb_name: selectedKb.value,
        ...(chatStore.activeSessionId ? { session_id: chatStore.activeSessionId } : {}),
        ...(newSessionPending.value ? { new_session: true } : {}),
        ...(imageBase64 ? { image_base64: imageBase64, image_mime: imageMime } : {}),
        ...(selectedModel.value ? { model_provider: selectedModel.value } : {}),
        ...(hasRetrievalFilters.value ? { filters: retrievalFilterPayload() } : {}),
        ...(webSearch?.value ? { web_search: true } : {})
      };
      const response = await fetchChat('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      if (!isCurrent()) return;
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        if (!isCurrent()) return;
        const defaultMessage = response.status === 412
          ? '尚未配置大模型 API Key，请管理员前往「模型管理」页面配置后再使用。'
          : '请求失败，请稍后重试。';
        if (response.status === 412) modelConfigured.value = false;
        // FastAPI 校验失败时 detail 是错误对象数组，不能直接展示。
        const detail = typeof errorData.detail === 'string' && errorData.detail
          ? errorData.detail
          : response.status === 422 ? `问题格式不符合要求，请检查长度（最多 ${MAX_QUESTION_CHARS} 字）后重试。` : defaultMessage;
        messages.value.push({ role: 'assistant', content: `⚠️ ${detail}` });
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let answer = '';
      const stages = [];
      liveStages.value = stages;
      let sessionCreated = false;
      let streamCompleted = false;
      // 每个 token 都重新渲染整段 Markdown 会让长回答越来越卡：这里先累积，
      // 按最小间隔写入消息；终态事件和流结束时一定会补写最终内容。
      let answerDirty = false;
      let lastFlushAt = 0;
      const flushAnswer = (force = false) => {
        // 会话已切换（消息数组被替换）时不能再写入，否则会污染新会话。
        if (!answerDirty || streamCompleted || messages.value !== requestMessages) return;
        const now = Date.now();
        if (!force && now - lastFlushAt < ANSWER_RENDER_INTERVAL_MS) return;
        messages.value[messages.value.length - 1].content = answer;
        answerDirty = false;
        lastFlushAt = now;
      };
      // 保存尚未形成完整 SSE 事件的网络分片。
      let buffer = '';

      // A network read may end in the middle of an SSE event. Keep incomplete text in
      // buffer until the protocol's blank-line boundary arrives.
      try {
        while (true) {
          const { value, done } = await readWithIdleTimeout(reader, idleTimeoutMs, controller);
          if (!isCurrent()) {
            await reader.cancel?.();
            return;
          }
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          let boundary = buffer.indexOf('\n\n');
          while (boundary !== -1) {
            const chunk = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + 2);
            boundary = buffer.indexOf('\n\n');
            if (!chunk.startsWith('data: ')) continue;

            try {
              const data = JSON.parse(chunk.substring(6));
              if (data.stage) {
                stages.push(data);
                liveStages.value = [...stages];
                continue;
              }
              if (data.chunk) {
                // Reuse one assistant message while tokens stream to avoid reactive list churn.
                if (answer === '') messages.value.push({ role: 'assistant', content: '', sources: [] });
                answer += data.chunk;
                answerDirty = true;
              }
              if (data.is_end) {
                streamCompleted = true;
                // A generation failure is not an answer: show it as a warning that is never persisted.
                if (data.error) {
                  if (answer === '') messages.value.push({ role: 'assistant', content: '', sources: [] });
                  const failedMessage = messages.value[messages.value.length - 1];
                  failedMessage.content = `⚠️ ${data.error}`;
                  failedMessage.sources = [];
                  failedMessage.stages = stages;
                  if (data.trace_id) failedMessage.trace_id = data.trace_id;
                } else {
                  // Sources and session ID are authoritative only in the terminal event.
                  if (answer === '') messages.value.push({ role: 'assistant', content: '', sources: [] });
                  const lastMessage = messages.value[messages.value.length - 1];
                  lastMessage.content = data.final_answer || answer;
                  lastMessage.sources = data.sources || [];
                  if (data.interrupted) lastMessage.interrupted = true;
                  if (data.diagnostics && Object.keys(data.diagnostics).length) lastMessage.diagnostics = data.diagnostics;
                  lastMessage.stages = stages;
                  if (data.trace_id) {
                    lastMessage.trace_id = data.trace_id;
                    lastMessage.feedback = 0;
                  }
                }
                if (!chatStore.activeSessionId && data.session_id && !sessionCreated) {
                  chatStore.activeSessionId = data.session_id;
                  await loadSessions();
                  if (!isCurrent()) return;
                  sessionCreated = true;
                }
                if (data.session_id) newSessionPending.value = false;
              }
            } catch (error) {
              console.error('Error parsing stream:', error);
            }
          }
          flushAnswer();
        }
      } finally {
        // 停止、空闲超时或异常退出时也要把已收到的内容写进界面。
        flushAnswer(true);
      }

      if (!streamCompleted && !controller.signal.aborted) {
        const interruptedMessage = '回答连接意外中断，请重试。';
        if (answer === '') {
          messages.value.push({
            role: 'assistant',
            content: `⚠️ ${interruptedMessage}`,
            sources: []
          });
        } else {
          notifications.error(interruptedMessage);
        }
      }
    } catch (error) {
      if (error instanceof StreamIdleError) {
        // 超时由我们主动中止；保留已收到的部分回答，只提示连接中断。
        if (abortController === controller && messages.value === requestMessages) {
          notifications.error('回答连接长时间没有响应，已停止等待，请重试。');
        }
        return;
      }
      if (error.name === 'AbortError' || !isCurrent()) return;
      let errorMessage = '⚠️ 请求失败，请稍后重试。';
      if (error?.response?.status === 412 || error?.status === 412) {
        errorMessage = '⚠️ 尚未配置大模型 API Key，请管理员前往左侧菜单「模型管理」页面填写并保存 API Key 后再使用。';
        modelConfigured.value = false;
      }
      try {
        if (error instanceof Response || error?.json) {
          const data = await error.json();
          if (data?.detail) errorMessage = `⚠️ ${data.detail}`;
        }
      } catch (_) {
        // Keep the fallback error message when the response body is not JSON.
      }
      if (isCurrent()) messages.value.push({ role: 'assistant', content: errorMessage });
    } finally {
      if (abortController === controller) {
        abortController = null;
        loading.value = false;
      }
    }
  };

  const cleanup = () => {
    stop();
    removeAttachedImage();
    sentPreviewUrls.forEach(url => URL.revokeObjectURL(url));
    sentPreviewUrls.clear();
  };
  if (getCurrentInstance()) onBeforeUnmount(cleanup);

  return {
    input,
    loading,
    liveStages,
    attachedImage,
    handleImageSelect,
    handleImagePaste,
    removeAttachedImage,
    send,
    stop
  };
}
