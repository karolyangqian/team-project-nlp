"""Stage 3 prompt formatting, generator backends, and SFT dataset builders."""

from draft.generation.backends import (
    BaseGenerator,
    MockGenerator,
    OllamaGenerator,
    OpenAICompatibleGenerator,
)
from draft.generation.prompts import (
    PromptBuilder,
    PromptMessages,
    build_reply_subject,
    extract_cited_doc_ids,
    normalize_target_reply,
)
from draft.generation.sft_dataset import (
    DOC_CATEGORY_TO_INTENT,
    SftDatasetBuilder,
    build_policy_grounded_examples,
)

__all__ = [
    "BaseGenerator",
    "DOC_CATEGORY_TO_INTENT",
    "MockGenerator",
    "OllamaGenerator",
    "OpenAICompatibleGenerator",
    "PromptBuilder",
    "PromptMessages",
    "SftDatasetBuilder",
    "build_policy_grounded_examples",
    "build_reply_subject",
    "extract_cited_doc_ids",
    "normalize_target_reply",
]
