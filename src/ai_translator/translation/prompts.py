from __future__ import annotations

import hashlib
from pathlib import Path
from string import Template
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import PromptVersion
from .errors import TranslationConfigurationError


_PROMPT_DIR = Path(__file__).resolve().parents[1] / "resources" / "prompts"


class PromptTemplateManager:
    def load_template(self, domain: str, *, task_kind: str = "translation",
                      prompt_file: Path | None = None) -> tuple[str, str]:
        if prompt_file is not None:
            try:
                return prompt_file.read_text(encoding="utf-8"), f"custom:{prompt_file.name}:{task_kind}"
            except OSError as exc:
                raise TranslationConfigurationError(f"Cannot read prompt file {prompt_file}: {exc}") from exc
        candidate = _PROMPT_DIR / f"{domain}-{task_kind}.txt"
        if not candidate.is_file() and task_kind in {"simplification", "simplify", "reading_level"}:
            candidate = _PROMPT_DIR / "simplification.txt"
        if not candidate.is_file():
            candidate = _PROMPT_DIR / f"{domain}.txt"
        if not candidate.is_file():
            candidate = _PROMPT_DIR / "default.txt"
        return candidate.read_text(encoding="utf-8"), f"translation:{domain}:{task_kind}"

    def render(self, template_text: str, values: dict[str, str]) -> str:
        try:
            return Template(template_text).substitute(values)
        except KeyError as exc:
            raise TranslationConfigurationError(f"Prompt template references unknown variable {exc.args[0]!r}") from exc
        except ValueError as exc:
            raise TranslationConfigurationError(f"Invalid prompt template: {exc}") from exc

    @staticmethod
    def get_or_create_version(session: Session, prompt_key: str, template_text: str) -> PromptVersion:
        digest = hashlib.sha256(template_text.encode("utf-8")).hexdigest()
        existing = session.scalar(select(PromptVersion).where(
            PromptVersion.prompt_key == prompt_key,
            PromptVersion.content_hash == digest,
        ))
        if existing is not None:
            return existing
        current_max = session.scalar(select(func.max(PromptVersion.version_number)).where(
            PromptVersion.prompt_key == prompt_key
        )) or 0
        version = PromptVersion(
            prompt_key=prompt_key,
            version_number=current_max + 1,
            template=template_text,
            content_hash=digest,
            metadata_json={"template_engine": "string.Template", "encoding": "utf-8"},
        )
        session.add(version)
        session.flush()
        return version


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

