"""Prompt templates, reply normalization, and citation helpers for Stage 3."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from draft.schemas import ClassificationResult, IncomingEmail, RetrievedContext

_BR_TAG_RE = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_DOC_ID_RE = re.compile(r"\b(DOC\d{3})\b")


@dataclass(frozen=True)
class PromptMessages:
    """Rendered system, optional few-shot turns, and user prompts for draft generation."""

    system: str
    user: str
    few_shot_examples: tuple[tuple[str, str], ...] = ()

    def to_chat_messages(self, assistant: str | None = None) -> list[dict[str, str]]:
        """Format prompt as standard role/content chat messages."""
        messages: list[dict[str, str]] = [{"role": "system", "content": self.system}]
        for shot_user, shot_assistant in self.few_shot_examples:
            messages.append({"role": "user", "content": shot_user})
            messages.append({"role": "assistant", "content": shot_assistant})
        messages.append({"role": "user", "content": self.user})
        if assistant is not None:
            messages.append({"role": "assistant", "content": assistant})
        return messages


def normalize_target_reply(text: str) -> str:
    """Replace HTML break tags and collapse redundant whitespace while keeping placeholders."""
    cleaned = _BR_TAG_RE.sub("\n", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.split("\n")]
    normalized = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", normalized).strip()


def extract_cited_doc_ids(text: str, valid_doc_ids: set[str] | None = None) -> list[str]:
    """Extract unique DOCxxx citations from generated text in appearance order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for doc_id in _DOC_ID_RE.findall(text):
        if valid_doc_ids is not None and doc_id not in valid_doc_ids:
            continue
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return ordered


def build_reply_subject(incoming_subject: str) -> str:
    """Build a standard Re: reply subject line without duplicating Re: prefixes."""
    stripped = incoming_subject.strip()
    if not stripped:
        return "Re: Your Support Inquiry"
    if stripped.lower().startswith("re:"):
        remainder = stripped[3:].lstrip()
        return f"Re: {remainder}" if remainder else "Re: Your Support Inquiry"
    return f"Re: {stripped}"


class PromptBuilder:
    """Renders Stage 3 system, few-shot, and user prompts from template files."""

    def __init__(self, prompts_dir: Path) -> None:
        env = Environment(
            loader=FileSystemLoader(str(prompts_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )
        self._system_tpl = env.get_template("system.j2")
        self._user_tpl = env.get_template("user_draft.j2")
        self._few_shot_examples = self._load_few_shot(prompts_dir / "few_shot.json")

    @staticmethod
    def _load_few_shot(path: Path) -> tuple[tuple[str, str], ...]:
        """Load optional few-shot chat turns from JSON."""
        if not path.exists():
            return ()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return tuple((str(item["user"]), str(item["assistant"])) for item in raw)

    def render(
        self,
        email: IncomingEmail,
        classification: ClassificationResult,
        context: RetrievedContext,
        use_few_shot: bool = True,
    ) -> PromptMessages:
        """Render complete chat messages from email, Stage 1 labels, and Stage 2 context."""
        docs_ctx = [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "category": str(doc.category),
                "content": doc.content,
            }
            for doc in context.top_k_docs
        ]
        responses_ctx = [
            {
                "intent": str(resp.intent),
                "urgency": str(resp.urgency),
                "answer": normalize_target_reply(resp.answer),
            }
            for resp in context.top_k_responses
        ]
        system_text = self._system_tpl.render().strip()
        user_text = self._user_tpl.render(
            intent=str(classification.intent),
            urgency=str(classification.urgency),
            docs=docs_ctx,
            responses=responses_ctx,
            email_text=email.formatted_text,
        ).strip()
        few_shot = self._few_shot_examples if (use_few_shot and context.top_k_docs) else ()
        return PromptMessages(system=system_text, user=user_text, few_shot_examples=few_shot)
