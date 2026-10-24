/** 安全读取用户对象中的权限数组，避免旧会话字段缺失导致报错。 */
const permissionsFor = (user, key) => {
  const permissions = user?.[key];
  return Array.isArray(permissions) ? permissions : [];
};
/** 判断用户是否有权读取指定知识库。 */
const canAccessKnowledgeBase = (user, knowledgeBaseName) => {
  const permissions = permissionsFor(user, 'knowledgeBasePermissions').length
    ? permissionsFor(user, 'knowledgeBasePermissions')
    : permissionsFor(user, 'deptPermissions');
  return permissions.includes('全公司') || permissions.includes(knowledgeBaseName);
};
/** 从知识库列表中过滤出当前用户可访问的项目。 */
export const filterKnowledgeBases = (knowledgeBases, user) =>
  (knowledgeBases || []).filter((knowledgeBase) =>
    canAccessKnowledgeBase(user, knowledgeBase.name)
  );
/** 判断用户是否可以进入知识库页面。 */
export const canOpenKnowledgeBase = (user) => {
  const permissions = permissionsFor(user, 'menuPermissions');
  return permissions.includes('知识库') || permissions.includes('知识库管理');
};
/** 判断用户是否拥有知识库增删改等管理权限。 */
export const canManageKnowledgeBases = (user) =>
  permissionsFor(user, 'menuPermissions').includes('知识库管理');
