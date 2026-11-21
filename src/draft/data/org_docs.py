"""Organization policy documents synthesis, persistence, and loading."""

from pathlib import Path

import pandas as pd

from draft.schemas import DocCategory, OrgDocument

_RAW_ORG_DOCS: list[tuple[str, DocCategory, str, str]] = [
    # Billing and Payments (4)
    (
        "DOC001",
        DocCategory.BILLING_AND_PAYMENTS,
        "Billing Cycle and Invoices",
        (
            "Invoices are issued on the first day of each month for the previous month's usage "
            "and are emailed to the billing contact. Standard payment terms are Net 15 from the "
            "invoice issue date unless an enterprise Net 30 agreement is active. Copies of past "
            "PDF invoices and itemized usage breakdowns are downloadable directly from the "
            "Billing Portal. Unpaid balances beyond 10 days past due trigger automated reminder "
            "notices prior to any temporary account restriction."
        ),
    ),
    (
        "DOC002",
        DocCategory.BILLING_AND_PAYMENTS,
        "Disputed Charges and Credit Adjustments",
        (
            "Customers may dispute unexpected charges or duplicate invoice line items within 30 "
            "calendar days of the statement date. Billing analysts complete initial charge audits "
            "within 2 business days and suspend automated dunning while the dispute is under "
            "review. Verified overcharges are refunded to the original payment method within 5 to "
            "7 business days or issued immediately as an account credit upon customer request."
        ),
    ),
    (
        "DOC003",
        DocCategory.BILLING_AND_PAYMENTS,
        "Accepted Payment Methods and Tax Compliance",
        (
            "Supported payment methods include major corporate credit cards, ACH bank transfers, "
            "and SEPA direct debit for accounts billed in USD or EUR. Enterprise accounts with "
            "annual commitments exceeding 5,000 USD may pay via wire transfer against formal "
            "purchase orders. Valid VAT IDs and US tax exemption certificates must be uploaded in "
            "Billing Settings before invoice generation to apply zero-rated tax rules."
        ),
    ),
    (
        "DOC004",
        DocCategory.BILLING_AND_PAYMENTS,
        "Subscription Upgrades, Downgrades, and Proration",
        (
            "Plan upgrades and additional seat purchases take effect immediately with prorated "
            "charges calculated to the exact remaining days in the current billing cycle. "
            "Downgrades and seat reductions are scheduled to apply at the start of the next "
            "billing cycle so existing workflows are not interrupted mid-month. Annual plans may "
            "be adjusted upward at any time, while downward tier changes occur at renewal."
        ),
    ),
    # Account and Access (4)
    (
        "DOC005",
        DocCategory.ACCOUNT_AND_ACCESS,
        "Password Reset and Account Lockout Recovery",
        (
            "Accounts automatically lock for 15 minutes after five consecutive failed sign-in "
            "attempts to prevent credential stuffing. Users can initiate self-service password "
            "resets via the primary verified email address, with reset links expiring after 30 "
            "minutes. If an employee has lost mailbox access, an Organization Admin must submit a "
            "signed verification ticket to update the recovery address within 4 business hours."
        ),
    ),
    (
        "DOC006",
        DocCategory.ACCOUNT_AND_ACCESS,
        "Multi-Factor Authentication (MFA) Reset Policy",
        (
            "Multi-Factor Authentication is mandatory for all administrator and billing roles "
            "across the platform. When a user loses their authenticator device, they should first "
            "use one of the 10 single-use backup codes generated during enrollment. If backup "
            "codes are unavailable, Support will reset MFA only after identity verification from "
            "a designated Organization Admin via the verified support portal."
        ),
    ),
    (
        "DOC007",
        DocCategory.ACCOUNT_AND_ACCESS,
        "Role-Based Access Control and Workspace Permissions",
        (
            "Workspaces enforce four standard roles: Owner, Admin, Editor, and Viewer, following "
            "the principle of least privilege. Only Organization Owners can transfer workspace "
            "ownership, approve SSO configuration changes, or authorize permanent workspace "
            "deletion. Permission changes take effect on the user's next API call or within 60 "
            "seconds for active web sessions."
        ),
    ),
    (
        "DOC008",
        DocCategory.ACCOUNT_AND_ACCESS,
        "Single Sign-On (SSO) and SCIM Provisioning",
        (
            "SAML 2.0 and OIDC Single Sign-On are supported for Okta, Microsoft Entra ID, Google "
            "Workspace, and Ping Identity on Business and Enterprise plans. SCIM 2.0 user "
            "provisioning automatically suspends workspace seats within 5 minutes of deactivating "
            "a user in the identity provider. Testing SSO changes in a sandbox workspace is "
            "required before enforcing strict SSO login across production users."
        ),
    ),
    # Integrations (4)
    (
        "DOC009",
        DocCategory.INTEGRATIONS,
        "REST API Authentication and Rate Limits",
        (
            "All REST API requests require a scoped Bearer token or OAuth 2.0 access token over "
            "HTTPS TLS 1.2 or higher. Standard workspaces are allotted 120 requests per minute "
            "per token, while Enterprise workspaces support up to 600 requests per minute. Clients "
            "receiving HTTP 429 responses should honor the Retry-After header using exponential "
            "backoff rather than tight retry loops."
        ),
    ),
    (
        "DOC010",
        DocCategory.INTEGRATIONS,
        "Webhook Delivery, Signatures, and Retries",
        (
            "Outbound webhooks include an HMAC-SHA256 signature in the X-Signature header so "
            "receivers can verify payload authenticity before processing. Webhook endpoints must "
            "return a 2xx HTTP status within 5 seconds; otherwise the delivery is marked timed "
            "out. Failed webhook events are automatically retried up to 5 times with exponential "
            "backoff over a 6-hour window before being sent to the dead-letter log."
        ),
    ),
    (
        "DOC011",
        DocCategory.INTEGRATIONS,
        "Third-Party CRM and Helpdesk Connectors",
        (
            "Native bi-directional connectors are maintained for Salesforce, HubSpot, Zendesk, "
            "Jira, and Slack through the Integrations Marketplace. Incremental synchronization "
            "runs every 5 minutes, while manual full resyncs are throttled to once per hour to "
            "protect API quotas. When a connector reports an expired OAuth grant, re-authorizing "
            "with a service account preserves field mappings and historical sync state."
        ),
    ),
    (
        "DOC012",
        DocCategory.INTEGRATIONS,
        "SDK Versioning and Endpoint Deprecation Policy",
        (
            "Official Python, TypeScript, and Go SDKs follow semantic versioning, with active "
            "security patches backported to the previous major version for 12 months. Breaking "
            "REST API changes are introduced only in versioned endpoints with a minimum 180-day "
            "deprecation notice emailed to technical contacts. Deprecated endpoints emit Sunset "
            "response headers indicating the exact retirement date."
        ),
    ),
    # Performance (3)
    (
        "DOC013",
        DocCategory.PERFORMANCE,
        "Application Latency Targets and Troubleshooting",
        (
            "The platform targets a p95 web dashboard response time under 400 milliseconds and API "
            "read latency under 250 milliseconds under normal load. When troubleshooting slow "
            "client rendering, users should capture a browser HAR trace, disable conflicting "
            "extensions, and verify network latency to the regional edge node. Support engineers "
            "require a timestamp, workspace ID, and request correlation ID to inspect backend "
            "traces."
        ),
    ),
    (
        "DOC014",
        DocCategory.PERFORMANCE,
        "Large Dataset Import and Batch Processing Limits",
        (
            "CSV and JSON bulk imports are capped at 250 MB or 500,000 records per single upload "
            "job to maintain shared cluster stability. Imports exceeding 50,000 rows are queued "
            "asynchronously and emit a completion email or webhook upon finishing. Splitting "
            "multi-million-row migrations into sequential 100,000-row batches avoids memory "
            "pressure and speeds up validation error reporting."
        ),
    ),
    (
        "DOC015",
        DocCategory.PERFORMANCE,
        "Caching Behavior and Edge Content Propagation",
        (
            "Public configuration assets and static dashboard tiles are cached at edge locations "
            "with a default time-to-live of 300 seconds. Updating workspace branding or routing "
            "rules invalidates edge caches automatically, though global propagation may take up "
            "to 2 minutes. Query endpoints supporting the Cache-Control no-cache header allow "
            "real-time validation after urgent administrative updates."
        ),
    ),
    # Outages and Maintenance (4)
    (
        "DOC016",
        DocCategory.OUTAGES_AND_MAINTENANCE,
        "Service Outage Communication and Status Updates",
        (
            "During active service disruptions, incident commanders publish initial status "
            "updates on the public Status Page within 15 minutes of detection. Ongoing incidents "
            "receive structured progress updates every 30 minutes until full service restoration "
            "is confirmed. Enterprise customers receive automated email and webhook notifications "
            "and can request a formal Root Cause Analysis within 5 business days of resolution."
        ),
    ),
    (
        "DOC017",
        DocCategory.OUTAGES_AND_MAINTENANCE,
        "Scheduled Maintenance Windows and Notice Policy",
        (
            "Routine infrastructure maintenance is scheduled on Sundays between 01:00 and 04:00 "
            "UTC to minimize impact across global business hours. Customers receive advance email "
            "notice at least 72 hours prior to any maintenance window expected to cause brief "
            "read-only periods or failovers. Emergency security patches may be applied outside "
            "standard windows with immediate notification via the Status Page."
        ),
    ),
    (
        "DOC018",
        DocCategory.OUTAGES_AND_MAINTENANCE,
        "Service Level Agreement (SLA) and Credit Eligibility",
        (
            "Paid production workspaces are covered by a 99.9 percent monthly uptime SLA, while "
            "Enterprise plans carry a 99.95 percent availability commitment. Downtime caused by "
            "announced scheduled maintenance or customer-side network failures is excluded from "
            "SLA calculations. Eligible service credits must be requested through Support within "
            "15 calendar days after the end of the affected billing month."
        ),
    ),
    (
        "DOC019",
        DocCategory.OUTAGES_AND_MAINTENANCE,
        "Incident Severity Classification and Response SLAs",
        (
            "Priority 1 critical outages affecting all workspace users receive an initial human "
            "engineering response within 15 minutes, 24/7. Priority 2 degraded performance tickets "
            "carry a 2-hour response target, while Priority 3 general defects are triaged within "
            "1 business day. Support tickets tagged with active outage indicators are "
            "automatically escalated to the on-call reliability rotation."
        ),
    ),
    # Security and Compliance (4)
    (
        "DOC020",
        DocCategory.SECURITY_AND_COMPLIANCE,
        "Data Encryption Standards in Transit and at Rest",
        (
            "All external and internal service traffic is encrypted in transit using TLS 1.2 or "
            "TLS 1.3 with strong forward-secrecy cipher suites. Customer databases, file "
            "attachments, and automated backups are encrypted at rest using AES-256 managed via "
            "cloud KMS with annual key rotation. Enterprise plans optionally support "
            "customer-managed encryption keys (CMEK) for regulated workloads."
        ),
    ),
    (
        "DOC021",
        DocCategory.SECURITY_AND_COMPLIANCE,
        "GDPR Data Subject Requests and Retention Policy",
        (
            "Workspace Owners can submit GDPR access, export, or right-to-erasure requests "
            "through the Privacy Center or by emailing our Data Protection Officer. Verified "
            "deletion requests purge primary production records within 14 days and expire "
            "encrypted backup snapshots within 30 days. Audit logs required for statutory "
            "financial recordkeeping are retained in anonymized form for 7 years."
        ),
    ),
    (
        "DOC022",
        DocCategory.SECURITY_AND_COMPLIANCE,
        "Security Vulnerability Disclosure and Incident Reporting",
        (
            "Suspected account compromises, phishing attempts, or unauthorized API access must be "
            "reported immediately as High-urgency Security incidents. Upon confirmation of "
            "unauthorized credential use, Support immediately revokes active sessions and rotates "
            "workspace API secrets. Responsible security researchers may submit reproducible "
            "reports via our encrypted PGP intake channel for a 24-hour triage acknowledgment."
        ),
    ),
    (
        "DOC023",
        DocCategory.SECURITY_AND_COMPLIANCE,
        "SOC 2 Type II and Compliance Audit Documentation",
        (
            "The platform maintains annual SOC 2 Type II and ISO 27001 certifications covering "
            "security, availability, and confidentiality trust principles. Current audit reports, "
            "penetration test executive summaries, and subprocessor lists are accessible under "
            "NDA from the Trust Portal. Standard compliance questionnaires are processed by our "
            "Security Assurance team within 5 business days."
        ),
    ),
    # Data Analytics (3)
    (
        "DOC024",
        DocCategory.DATA_ANALYTICS,
        "Scheduled Reports and Data Export Formats",
        (
            "Analytics dashboards support on-demand and scheduled exports in CSV, JSON, and "
            "Parquet formats with UTF-8 encoding. Scheduled email reports can be configured daily, "
            "weekly, or monthly and delivered directly to up to 25 internal recipients. Exports "
            "exceeding 100,000 rows are generated asynchronously and delivered as a secure "
            "time-limited download link valid for 48 hours."
        ),
    ),
    (
        "DOC025",
        DocCategory.DATA_ANALYTICS,
        "Dashboard Refresh Cadence and Timezone Alignment",
        (
            "Operational telemetry dashboards refresh every 15 minutes, whereas historical trend "
            "aggregations update once per hour on the UTC boundary. Workspace Admins can set a "
            "default reporting timezone, which shifts daily rollups without mutating underlying "
            "UTC event timestamps. Discrepancies between UI charts and raw exports usually trace "
            "to mismatched date-range filters or unclosed rolling-window intervals."
        ),
    ),
    (
        "DOC026",
        DocCategory.DATA_ANALYTICS,
        "Warehouse Sync and SQL Query Quotas",
        (
            "Managed data warehouse connectors sync incrementally to Snowflake, BigQuery, and "
            "PostgreSQL every 60 minutes. Custom analytics API queries are limited to a 90-day "
            "lookback window per request and a 30-second execution timeout to prevent resource "
            "starvation. Multi-year historical backfills should be requested via Support to run "
            "on dedicated off-peak batch workers."
        ),
    ),
    # Digital Marketing (3)
    (
        "DOC027",
        DocCategory.DIGITAL_MARKETING,
        "Email Campaign Deliverability and DNS Authentication",
        (
            "Outbound marketing campaigns require verified SPF, DKIM, and DMARC DNS records on "
            "the sender domain before bulk dispatch is enabled. New sending domains undergo an "
            "automated 14-day warm-up schedule capping initial daily volume to protect sender "
            "reputation. Campaigns exceeding a 0.3 percent spam complaint threshold or 5 percent "
            "hard bounce rate are paused automatically pending list hygiene review."
        ),
    ),
    (
        "DOC028",
        DocCategory.DIGITAL_MARKETING,
        "Attribution Tracking, UTM Parameters, and Consent",
        (
            "Campaign attribution tracks standard UTM source, medium, campaign, and content "
            "parameters using a configurable 30-day last-touch window. Tracking pixels and "
            "first-party analytics cookies fire only after end-user consent is recorded when "
            "strict privacy mode is enabled. Unsubscribe headers must remain active in every "
            "promotional message to comply with CAN-SPAM and GDPR rules."
        ),
    ),
    (
        "DOC029",
        DocCategory.DIGITAL_MARKETING,
        "Audience Segmentation and Suppression Lists",
        (
            "Dynamic audience segments recalculate every 10 minutes based on behavioral events "
            "and CRM profile attributes. Global suppression lists override all active campaign "
            "automations so opted-out contacts never receive promotional follow-ups. Importing "
            "third-party purchased contact lists is strictly prohibited and results in immediate "
            "messaging suspension."
        ),
    ),
    # Sales and Pre-Sales (3)
    (
        "DOC030",
        DocCategory.SALES_AND_PRE_SALES,
        "Enterprise Quotes, Custom Pricing, and Validity",
        (
            "Formal price quotes issued by Account Executives remain valid for 30 calendar days "
            "from the date of issue unless marked as a limited promotional offer. Volume tier "
            "discounts apply starting at 50 licensed seats or multi-year annual commitments. "
            "Custom MSA redlines and procurement portal onboarding are coordinated directly "
            "through the assigned Sales representative."
        ),
    ),
    (
        "DOC031",
        DocCategory.SALES_AND_PRE_SALES,
        "Product Trials and Proof-of-Concept Evaluations",
        (
            "New organizations receive a self-serve 14-day full-feature trial without requiring a "
            "credit card upfront. Qualified enterprise prospects may request a 30-day guided "
            "Proof of Concept with dedicated Solutions Engineering support and sandbox data. "
            "Trial workspaces retain all configurations for 30 days after expiration so teams can "
            "convert smoothly to a paid subscription."
        ),
    ),
    (
        "DOC032",
        DocCategory.SALES_AND_PRE_SALES,
        "Plan Feature Comparison and Licensing Tiers",
        (
            "Starter plans cover core ticketing and standard analytics for up to 10 seats, while "
            "Business plans add SAML SSO, custom webhooks, and 99.9 percent SLA coverage. "
            "Enterprise plans unlock dedicated account management, SCIM provisioning, CMEK "
            "encryption, and 600 RPM API throughput. Pre-sales inquiries requiring architecture "
            "reviews are routed to a Solutions Architect within 1 business day."
        ),
    ),
    # Returns and Exchanges (4)
    (
        "DOC033",
        DocCategory.RETURNS_AND_EXCHANGES,
        "Hardware Return Eligibility and 30-Day Window",
        (
            "Physical hardware devices and smart accessories purchased directly from our store may "
            "be returned within 30 calendar days of confirmed delivery. Returned items must "
            "include original packaging, power adapters, and intact serial labels without "
            "physical misuse damage. Opened software license keys and custom-engraved units are "
            "ineligible for discretionary returns unless confirmed defective on arrival."
        ),
    ),
    (
        "DOC034",
        DocCategory.RETURNS_AND_EXCHANGES,
        "Return Merchandise Authorization (RMA) Workflow",
        (
            "Every physical return or warranty replacement requires an approved RMA number prior "
            "to shipping the item back to our fulfillment center. Customers must provide their "
            "order number, serial ID, and brief defect description to receive a prepaid return "
            "shipping label within 1 business day. Packages arriving without a valid RMA label "
            "may face processing delays of up to 10 business days."
        ),
    ),
    (
        "DOC035",
        DocCategory.RETURNS_AND_EXCHANGES,
        "Defective Item Exchanges and Advance Replacements",
        (
            "Hardware verified as dead-on-arrival within 14 days of delivery qualifies for free "
            "expedited advance replacement once the RMA is approved. Standard warranty exchanges "
            "for manufacturing defects within 12 months ship within 2 business days of "
            "warehouse inspection. Customers are notified via email with carrier tracking numbers "
            "as soon as the replacement unit dispatches."
        ),
    ),
    (
        "DOC036",
        DocCategory.RETURNS_AND_EXCHANGES,
        "Refund Processing Timelines and Subscription Cancellations",
        (
            "Refunds for returned physical goods are initiated within 3 business days after the "
            "warehouse confirms item condition. Card and bank refunds typically settle on customer "
            "statements within 5 to 10 business days depending on the issuing bank. Annual SaaS "
            "subscriptions are eligible for a full refund only within 14 days of initial purchase "
            "or renewal."
        ),
    ),
    # Human Resources (3)
    (
        "DOC037",
        DocCategory.HUMAN_RESOURCES,
        "Recruitment Status and Candidate Inquiries",
        (
            "Job applicants inquiring about active applications receive status updates from the "
            "Talent Acquisition team within 3 business days. Interview scheduling, travel "
            "reimbursements, and offer letter revisions are handled exclusively through the "
            "designated Recruiting coordinator. To protect candidate privacy, application feedback "
            "and offer details are shared only with the verified applicant email on file."
        ),
    ),
    (
        "DOC038",
        DocCategory.HUMAN_RESOURCES,
        "Employee Onboarding and IT Equipment Provisioning",
        (
            "New hire IT access tickets and laptop provisioning requests must be submitted by the "
            "hiring manager at least 5 business days before the start date. Identity accounts, "
            "email aliases, and role-based tool licenses are activated on the morning of the "
            "employee's start date following HR verification. Urgent missing access during "
            "onboarding week is prioritized under an internal 4-hour resolution SLA."
        ),
    ),
    (
        "DOC039",
        DocCategory.HUMAN_RESOURCES,
        "Leave Requests, Benefits, and Payroll Support",
        (
            "Questions regarding payroll discrepancies, tax withholding forms, or medical benefits "
            "enrollment are routed securely to the People Operations queue. Urgent payroll issues "
            "affecting an upcoming pay cycle are reviewed within 1 business day of submission. "
            "Employees must never attach sensitive government ID numbers in plain email; HR "
            "provides a secure encrypted upload link upon triage."
        ),
    ),
    # General Inquiry (3)
    (
        "DOC040",
        DocCategory.GENERAL_INQUIRY,
        "Support Operating Hours and Response Targets",
        (
            "Standard customer support operates Monday through Friday from 08:00 to 20:00 UTC, "
            "excluding major regional public holidays. Standard-urgency general inquiries receive "
            "an initial reply within 1 business day, while high-urgency tickets are triaged "
            "within 2 hours. Enterprise accounts include 24/7 follow-the-sun coverage for "
            "production-blocking incidents."
        ),
    ),
    (
        "DOC041",
        DocCategory.GENERAL_INQUIRY,
        "Ticket Escalation and Supervisor Review Process",
        (
            "Customers dissatisfied with an ongoing resolution or experiencing missed SLA targets "
            "may request formal escalation by replying with their ticket reference ID. Escalated "
            "tickets are reviewed by a Tier-2 Lead or Support Manager within 4 business hours. "
            "When transferring tickets between billing, technical, and security teams, agents "
            "attach an internal summary so customers do not need to repeat prior context."
        ),
    ),
    (
        "DOC042",
        DocCategory.GENERAL_INQUIRY,
        "Customer Communication Style and Verification Standards",
        (
            "All drafted customer replies must greet the customer politely, acknowledge their "
            "specific issue clearly, and state concrete next steps or timelines. Agents must "
            "never fabricate ETA commitments, unpublished roadmap dates, or policy exceptions "
            "that contradict official organization documents. Before sharing account-specific "
            "details or making changes, confirm the requester matches an authorized contact."
        ),
    ),
]


def build_default_org_docs() -> list[OrgDocument]:
    """Return the 42 curated organization policy documents across all 12 categories."""
    return [
        OrgDocument(doc_id=doc_id, category=category, title=title, content=content)
        for doc_id, category, title, content in _RAW_ORG_DOCS
    ]


def save_org_docs(docs: list[OrgDocument], path: Path) -> None:
    """Persist organization policy documents to a UTF-8 CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([doc.model_dump(mode="json") for doc in docs])
    frame.to_csv(path, index=False, encoding="utf-8")


def load_org_docs(path: Path) -> list[OrgDocument]:
    """Load and validate organization policy documents from CSV."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    return [OrgDocument.model_validate(row) for row in frame.to_dict(orient="records")]
