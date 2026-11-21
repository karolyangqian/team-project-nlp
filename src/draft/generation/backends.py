"""Modular LLM generation backends for local, remote API, and mock inference."""

import re
from typing import Protocol

import httpx

from draft.generation.prompts import PromptMessages, extract_cited_doc_ids

_URGENCY_LINE_RE = re.compile(
    r"^Urgency:\s*(low|medium|high)\b", flags=re.IGNORECASE | re.MULTILINE
)


class BaseGenerator(Protocol):
    """Protocol implemented by all Stage 3 LLM draft generators."""

    model_name: str

    def generate(
        self,
        messages: PromptMessages,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate an email draft body from rendered prompt messages."""
        ...


class OllamaGenerator:
    """Local GGUF inference generator backed by Ollama's /api/chat endpoint."""

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        default_max_new_tokens: int = 384,
        default_temperature: float = 0.2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_max_new_tokens = default_max_new_tokens
        self.default_temperature = default_temperature
        self.transport = transport

    def generate(
        self,
        messages: PromptMessages,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send chat prompt to Ollama and return the generated draft body."""
        endpoint = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages.to_chat_messages(),
            "stream": False,
            "options": {
                "temperature": (
                    self.default_temperature if temperature is None else temperature
                ),
                "num_predict": (
                    self.default_max_new_tokens if max_new_tokens is None else max_new_tokens
                ),
            },
        }
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.post(endpoint, json=payload)
                response.raise_for_status()
        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Start Ollama (`ollama serve`) and ensure `{self.model_name}` is created."
            ) from exc
        data = response.json()
        return str(data["message"]["content"]).strip()


class OpenAICompatibleGenerator:
    """Remote or local OpenAI-compatible /chat/completions generator."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        default_max_new_tokens: int = 384,
        default_temperature: float = 0.2,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.default_max_new_tokens = default_max_new_tokens
        self.default_temperature = default_temperature
        self.transport = transport

    def generate(
        self,
        messages: PromptMessages,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send chat completion request and return generated message content."""
        endpoint = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model_name,
            "messages": messages.to_chat_messages(),
            "max_tokens": (
                self.default_max_new_tokens if max_new_tokens is None else max_new_tokens
            ),
            "temperature": self.default_temperature if temperature is None else temperature,
        }
        with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
            response = client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
        data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()


class MockGenerator:
    """Deterministic offline generator for testing and dry-run pipeline verification."""

    def __init__(self, model_name: str = "mock-draft-generator") -> None:
        self.model_name = model_name

    def generate(
        self,
        messages: PromptMessages,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Produce a deterministic, urgency-aware, policy-citing mock reply."""
        del max_new_tokens, temperature
        match = _URGENCY_LINE_RE.search(messages.user)
        urgency = match.group(1).lower() if match else "medium"
        doc_ids = extract_cited_doc_ids(messages.user)

        if urgency == "high":
            opening = (
                "Thank you for contacting support, <name>. We recognize the urgency of your "
                "request and have prioritized it for immediate review."
            )
        elif urgency == "low":
            opening = (
                "Thank you for reaching out, <name>. We appreciate your patience and are happy "
                "to assist with your inquiry."
            )
        else:
            opening = (
                "Thank you for reaching out, <name>. Our support team is reviewing the details "
                "of your request and will follow up with concrete next steps."
            )

        policy_line = (
            f" In accordance with our policy guidelines [{doc_ids[0]}], our team will apply "
            "the standard verification and resolution steps for your account."
            if doc_ids
            else ""
        )
        closing = " Please share any additional reference details if needed."
        return f"{opening}{policy_line}{closing}"
