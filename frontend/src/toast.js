import { reactive } from 'vue';

export const toasts = reactive([]);

// 为每条提示生成稳定标识，删除提示时不依赖数组位置。
let idCounter = 0;

export const toast = {
  /** 显示成功提示。 */
  success(message) {
    this.add(message, 'success');
  },
  /** 显示错误提示。 */
  error(message) {
    this.add(message, 'error');
  },
  /** 加入一条提示并自动移除；错误提示通常更长，停留更久。 */
  add(message, type = 'info') {
    const id = idCounter++;
    toasts.push({ id, message, type });
    setTimeout(() => {
      this.remove(id);
    }, type === 'error' ? 6000 : 3000);
  },
  /** 根据提示标识从响应式列表中移除对应项。 */
  remove(id) {
    const index = toasts.findIndex(t => t.id === id);
    if (index > -1) {
      toasts.splice(index, 1);
    }
  }
};
