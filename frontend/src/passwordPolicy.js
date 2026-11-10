// 与后端 services/core/config.py 的密码规则保持一致；后端仍是最终权威。
export const MIN_PASSWORD_LENGTH = 8;
export const MAX_PASSWORD_BYTES = 72;

/** 按 Unicode 码点计数，和后端 len() 的结果一致。 */
const charCount = password => [...password].length;
/** bcrypt 只使用前 72 个 UTF-8 字节，超出部分必须拒绝而不是截断。 */
const byteCount = password => new TextEncoder().encode(password).length;

/** 检查已有密码（登录、原密码）只限制字节上限，避免旧的短密码无法登录。 */
export const existingPasswordProblem = (password, label = '密码') => (
  byteCount(password) > MAX_PASSWORD_BYTES ? `${label}不能超过 ${MAX_PASSWORD_BYTES} 字节` : ''
);

/** 检查新密码长度；不做 trim，首尾空格属于密码本身。 */
export const newPasswordProblem = (password, label = '密码') => {
  if (charCount(password) < MIN_PASSWORD_LENGTH) return `${label}至少需要 ${MIN_PASSWORD_LENGTH} 个字符`;
  return existingPasswordProblem(password, label);
};
