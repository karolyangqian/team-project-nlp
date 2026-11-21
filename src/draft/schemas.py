"""Core domain schemas and canonical formatting helpers across all pipeline stages."""

from enum import StrEnum

from pydantic import BaseModel, Field


class IntentLabel(StrEnum):
    """Supported customer support email intent categories."""

    TECHNICAL_SUPPORT = "Technical Support"
    PRODUCT_SUPPORT = "Product Support"
    CUSTOMER_SERVICE = "Customer Service"
    IT_SUPPORT = "IT Support"
    BILLING_AND_PAYMENTS = "Billing and Payments"
    RETURNS_AND_EXCHANGES = "Returns and Exchanges"
    SERVICE_OUTAGES_AND_MAINTENANCE = "Service Outages and Maintenance"
    SALES_AND_PRE_SALES = "Sales and Pre-Sales"
    HUMAN_RESOURCES = "Human Resources"
    GENERAL_INQUIRY = "General Inquiry"


class UrgencyLevel(StrEnum):
    """Customer email priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DocCategory(StrEnum):
    """Organization policy document categories."""

    BILLING_AND_PAYMENTS = "Billing and Payments"
    ACCOUNT_AND_ACCESS = "Account and Access"
    INTEGRATIONS = "Integrations"
    PERFORMANCE = "Performance"
    OUTAGES_AND_MAINTENANCE = "Outages and Maintenance"
    SECURITY_AND_COMPLIANCE = "Security and Compliance"
    DATA_ANALYTICS = "Data Analytics"
    DIGITAL_MARKETING = "Digital Marketing"
    SALES_AND_PRE_SALES = "Sales and Pre-Sales"
    RETURNS_AND_EXCHANGES = "Returns and Exchanges"
    HUMAN_RESOURCES = "Human Resources"
    GENERAL_INQUIRY = "General Inquiry"


def format_email_text(subject: str, body: str) -> str:
    """Return canonical single-string representation of an email subject and body."""
    clean_subject = subject.strip()
    clean_body = body.strip()
    if clean_subject:
        if clean_body:
            return f"Subject: {clean_subject}\n\n{clean_body}"
        return f"Subject: {clean_subject}"
    return clean_body


class EmailRecord(BaseModel):
    """Cleaned customer support email record with labels and staff answer."""

    email_id: str
    subject: str = ""
    body: str
    email: str
    intent: IntentLabel
    urgency: UrgencyLevel
    answer: str
    ticket_type: str = ""
    tags: list[str] = Field(default_factory=list)


class OrgDocument(BaseModel):
    """Organizational policy or guideline document used for factual grounding."""

    doc_id: str
    category: DocCategory
    title: str
    content: str


class IncomingEmail(BaseModel):
    """Raw incoming customer email submitted to the inference pipeline."""

    subject: str = ""
    body: str
    sender_name: str | None = None

    @property
    def formatted_text(self) -> str:
        """Return canonical formatted text matching training records."""
        return format_email_text(self.subject, self.body)


class ClassificationResult(BaseModel):
    """Stage 1 output containing predicted intent, urgency, and score distributions."""

    intent: IntentLabel
    urgency: UrgencyLevel
    intent_confidence: float = Field(ge=0.0, le=1.0)
    urgency_confidence: float = Field(ge=0.0, le=1.0)
    intent_scores: dict[str, float] = Field(default_factory=dict)
    urgency_scores: dict[str, float] = Field(default_factory=dict)


class RetrievedDoc(BaseModel):
    """Single retrieved organization policy document with ranking score."""

    doc_id: str
    category: DocCategory
    title: str
    content: str
    score: float


class RetrievedResponse(BaseModel):
    """Single retrieved historical email-response reference pair."""

    email_id: str
    email: str
    answer: str
    intent: IntentLabel
    urgency: UrgencyLevel
    score: float


class RetrievedContext(BaseModel):
    """Stage 2 output bundling dual-source retrieval results."""

    top_k_docs: list[RetrievedDoc] = Field(default_factory=list)
    top_k_responses: list[RetrievedResponse] = Field(default_factory=list)


class DraftResponse(BaseModel):
    """Stage 3 final drafted reply with supporting metadata and citations."""

    subject: str
    body: str
    classification: ClassificationResult
    retrieved_context: RetrievedContext
    cited_doc_ids: list[str] = Field(default_factory=list)
    model_name: str = ""
