"""Typed request and response contracts shared by HTTP routers."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

from services.chat.sessions import SESSION_TITLE_RENAME_MAX_CHARS
from services.core.config import (
    DOCUMENT_METADATA_MAX_TAG_LENGTH,
    DOCUMENT_METADATA_MAX_TAGS,
    validate_kb_name,
)
from services.indexing.contracts import IndexJobOperation, IndexJobStatus


class ApiModel(BaseModel):
    """Base API contract that rejects unknown fields and trims strings."""
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class SuccessResponse(ApiModel):
    """Common response returned by successful mutation endpoints."""
    status: Literal["success"] = "success"


class LoginRequest(ApiModel):
    """Credentials submitted by an account."""
    username: str
    password: str


class RegistrationRequest(ApiModel):
    """New user registration awaiting administrator approval."""
    username: str
    name: str
    phone: str
    password: str


class PasswordChangeRequest(ApiModel):
    """Password-change payload for the current account."""
    old_password: str
    new_password: str


class JobResponse(SuccessResponse):
    """Successful mutation response exposing its durable index job ID."""
    job_id: str


class RebuildResponse(JobResponse):
    """Confirmation that a complete knowledge-base rebuild was queued."""
    rebuilding: bool


class KnowledgeBaseNameRequest(ApiModel):
    """Validated knowledge-base name used by create and rename operations."""
    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """拒绝可能造成路径穿越或文件系统不安全的名称。"""
        return validate_kb_name(value)


class KnowledgeBaseSummary(ApiModel):
    """Knowledge-base list item with its document count."""
    name: str
    date: str
    docs: int


class ChunkConfigRequest(ApiModel):
    """Requested parent chunk size and overlap for future index builds."""
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self):
        """确保重叠长度严格小于父分块长度。"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("重叠长度须小于分块长度")
        return self


class ChunkConfigResponse(ApiModel):
    """Effective chunk configuration and whether indexes are stale."""
    chunk_size: int
    chunk_overlap: int
    index_needs_rebuild: bool


class RetrievalConfigRequest(ApiModel):
    """Retrieval behavior that can change without rebuilding indexes."""
    query_rewrite_enabled: StrictBool


class RetrievalConfigResponse(ApiModel):
    """Current retrieval behavior for one knowledge base."""
    query_rewrite_enabled: bool


class DocumentFilterOptionsResponse(ApiModel):
    """Distinct metadata values available to retrieval filters."""
    sources: list[str]
    document_types: list[str]
    tags: list[str]


class IndexJobResponse(ApiModel):
    """Public state and timestamps for one durable indexing job."""
    id: str
    filename: str | None = None
    operation: IndexJobOperation
    status: IndexJobStatus
    error_message: str = ""
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class DashboardFileType(ApiModel):
    """Dashboard document count grouped by file type."""
    ext: str
    count: int


class DashboardKnowledgeBase(ApiModel):
    """Dashboard processing and size statistics for one knowledge base."""
    name: str
    docs: int
    size: str
    success: int
    fail: int
    processing: int
    types: list[DashboardFileType]


class DashboardPiePoint(ApiModel):
    """Named value consumed by dashboard pie charts."""
    name: str
    value: int


class DashboardLinePoint(ApiModel):
    """Dated session and answer counts consumed by trend charts."""
    name: str
    session_count: int = Field(serialization_alias="会话数", validation_alias="会话数")
    answer_count: int = Field(serialization_alias="回答数", validation_alias="回答数")


class DashboardStatsResponse(ApiModel):
    """Complete dashboard totals, breakdowns, and recent trends."""
    kbs: int
    docs: int
    size: str
    tasks: int
    kb_list: list[DashboardKnowledgeBase]
    pie_data: list[DashboardPiePoint]
    line_data: list[DashboardLinePoint]


class DocumentMetadataRequest(ApiModel):
    """Editable metadata used for document-level retrieval filters."""
    document_type: str = Field(default="", max_length=80)
    tags: list[str] | str = Field(default_factory=list)
    effective_date: str = Field(default="", max_length=10)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str] | str) -> list[str] | str:
        """在进入存储层前限制规范化标签的数量和长度。"""
        values = value.replace("，", ",").split(",") if isinstance(value, str) else value
        tags = [" ".join(tag.split()) for tag in values if tag.strip()]
        if len(dict.fromkeys(tags)) > DOCUMENT_METADATA_MAX_TAGS:
            raise ValueError(f"最多可填写 {DOCUMENT_METADATA_MAX_TAGS} 个标签")
        if any(len(tag) > DOCUMENT_METADATA_MAX_TAG_LENGTH for tag in tags):
            raise ValueError(
                f"单个标签不能超过 {DOCUMENT_METADATA_MAX_TAG_LENGTH} 个字符"
            )
        return value

    @field_validator("effective_date")
    @classmethod
    def validate_effective_date(cls, value: str) -> str:
        """只接受空日期或真实有效的 ISO 日历日期。"""
        if value:
            try:
                date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("生效日期必须使用 YYYY-MM-DD 格式") from exc
        return value


class DocumentMetadataResponse(ApiModel):
    """Normalized metadata persisted for one document."""
    document_type: str
    tags: list[str]
    effective_date: str


class DocumentResponse(ApiModel):
    """Document list item with processing state, parser, and metadata."""
    filename: str
    path: str
    size: int
    uploaded_at: str
    status: str
    error_message: str
    index_job_id: str
    document_type: str
    tags: list[str]
    effective_date: str
    parser_mode: str


class DocumentParserOption(ApiModel):
    """One parser mode exposed to the document upload UI."""
    id: str
    label: str
    available: bool
    unavailable_reason: str = ""
    extensions: list[str]


class DocumentParserOptionsResponse(ApiModel):
    """Default parser and all parser choices supported by the backend."""
    document: list[DocumentParserOption]
    pdf: list[DocumentParserOption]


class DocumentFormatResponse(ApiModel):
    """Supported extension and the parser modes it permits."""
    extension: str
    label: str


class DocumentUploadResponse(JobResponse):
    """Upload result with stored filename and parser mode."""
    metadata: DocumentMetadataResponse | None
    parser_mode: str


class DocumentMetadataUpdateResponse(ApiModel):
    """Metadata update result with normalized persisted values."""
    status: str
    metadata: DocumentMetadataResponse


class DocumentParserRequest(ApiModel):
    """Request to change the parser used for one document."""
    parser_mode: Literal["native_text", "mineru_cloud", "paddle_ocr", "vision_model"]


class DocumentParserUpdateResponse(JobResponse):
    """Parser update result confirming the queued refresh job."""
    parser_mode: str


class ModelConfigRequest(ApiModel):
    """Editable OpenAI-compatible provider configuration."""
    provider: str = Field(min_length=1, max_length=120)
    api_key: str = ""
    base_url: str = Field(min_length=1, max_length=500)
    default_model: str = Field(min_length=1, max_length=200)
    clear_api_key: bool = False


class ModelConfigResponse(ApiModel):
    """Public provider settings with only API-key presence exposed."""
    provider: str
    configured: bool
    base_url: str | None
    default_model: str | None
    is_active: Literal[0, 1]


class ModelCheckResponse(ApiModel):
    """Provider readiness result used by the configuration UI."""
    configured: bool
    provider: str | None
    model: str | None
    is_active: bool


class ModelSaveResponse(SuccessResponse):
    """Model save result indicating whether the provider became active."""
    is_active: bool


class ModelActivationResponse(SuccessResponse):
    """Response confirming the newly active provider."""
    provider: str


class UserRoleRequest(ApiModel):
    """Role assignment for one user account."""
    role_id: int = Field(gt=0)


class UserStatusRequest(ApiModel):
    """Approval or enabled status assigned to one user account."""
    status: Literal["审核中", "正常", "停用"]


class UserResponse(ApiModel):
    """User administration list item with its resolved role."""
    id: int
    name: str
    phone: str
    status: str
    role: str
    time: str
    username: str | None


class UserPageResponse(ApiModel):
    """One page of user administration results and its resolved page metadata."""

    items: list[UserResponse]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_pages: int = Field(ge=1)


class RoleCreateRequest(ApiModel):
    """Name and description for a new role."""
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)


class RolePermissionsRequest(ApiModel):
    """Complete replacement permission sets for one role."""
    menu_permissions: list[str] = Field(default_factory=list)
    knowledge_base_permissions: list[str] = Field(default_factory=list)


class RoleResponse(ApiModel):
    """Role details and its menu and knowledge-base permissions."""
    id: int
    name: str
    description: str
    menu_permissions: list[str]
    knowledge_base_permissions: list[str]


class ChatSessionTitleRequest(ApiModel):
    """New display title for one chat session."""
    title: str = Field(min_length=1, max_length=SESSION_TITLE_RENAME_MAX_CHARS)


class RolePageResponse(ApiModel):
    """One page of role administration results and its resolved page metadata."""

    items: list[RoleResponse]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_pages: int = Field(ge=1)
