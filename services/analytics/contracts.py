"""用户分析接口返回的数据契约。"""

from typing import TypedDict


class DailyActivityPoint(TypedDict):
    """某一天（本地日期）的提问数、提问人数和登录人数。"""

    date: str
    questions: int
    active_users: int
    login_users: int


class KnowledgeBaseUsage(TypedDict):
    """某个知识库在统计窗口内被多少人、问了多少次、收到多少差评。"""

    kb_name: str
    users: int
    questions: int
    downvotes: int


class ActivityOverview(TypedDict):
    """用户分析总览。"""

    days: int
    total_users: int
    pending_users: int
    new_users: int
    active_users: int
    login_users: int
    login_failures: int
    questions: int
    upvotes: int
    downvotes: int
    daily: list[DailyActivityPoint]
    hourly: list[int]
    knowledge_bases: list[KnowledgeBaseUsage]


class UserActivityRow(TypedDict):
    """用户列表中的一行：账户信息加窗口内的使用统计。"""

    id: int
    username: str
    name: str
    status: str
    role: str
    created_at: str
    questions: int
    sessions: int
    errors: int
    upvotes: int
    downvotes: int
    logins: int
    login_failures: int
    last_question_at: str
    last_login_at: str


class UserActivityPage(TypedDict):
    items: list[UserActivityRow]
    total: int
    page: int
    page_size: int
    total_pages: int


class LoginEventRecord(TypedDict):
    created_at: str
    success: bool
    reason: str
    client: str


class UserSessionRecord(TypedDict):
    id: str
    title: str
    kb_name: str
    created_at: str
    updated_at: str
    questions: int


class UserQuestionRecord(TypedDict):
    created_at: str
    kb_name: str
    question_preview: str
    status: str
    total_ms: float
    feedback: int


class DailyQuestionPoint(TypedDict):
    date: str
    questions: int


class UserActivityDetail(TypedDict):
    """单个用户的详情：基本信息、窗口统计、趋势、最近登录、会话和提问。"""

    days: int
    user: UserActivityRow
    no_evidence: int
    daily: list[DailyQuestionPoint]
    knowledge_bases: list[KnowledgeBaseUsage]
    recent_logins: list[LoginEventRecord]
    recent_sessions: list[UserSessionRecord]
    recent_questions: list[UserQuestionRecord]


class ConversationMessage(TypedDict):
    role: str
    content: str
    created_at: str
    source_names: list[str]
    feedback: int


class UserConversation(TypedDict):
    session_id: str
    title: str
    kb_name: str
    username: str
    messages: list[ConversationMessage]


class AuditRecord(TypedDict):
    id: int
    created_at: str
    actor_username: str
    action: str
    target_username: str
    detail: str


class AuditPage(TypedDict):
    items: list[AuditRecord]
    total: int
    page: int
    page_size: int
    total_pages: int
