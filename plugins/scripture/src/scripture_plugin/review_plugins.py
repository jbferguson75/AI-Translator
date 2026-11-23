from __future__ import annotations

import json

from ai_translator.evaluation.plugins import BaseEvaluator

from .versioning import configured_version


class ChapterFlowEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(
            plugin_id="scripture.chapter_flow", version="1.0.0",
            display_name="Scripture: Chapter Flow",
            criteria=("Assess coherence of the chapter as a whole: transitions, progression, paragraph or "
                      "verse connections, and natural flow. Preserve meaning and identify abrupt or confusing transitions."),
        )


class ContextPreservationEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(
            plugin_id="scripture.context_preservation", version="1.0.0",
            display_name="Scripture: Context Preservation",
            criteria=("Assess whether each verse preserves references and meaning established by surrounding "
                      "verses in the supplied chapter. Flag lost antecedents, altered relationships, additions, or omissions."),
        )


class PronounResolutionEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(
            plugin_id="scripture.pronoun_resolution", version="1.0.0",
            display_name="Scripture: Pronoun Resolution",
            criteria=("Check pronouns and implicit references across this chapter. Flag unclear or changed "
                      "antecedents, mismatched persons/numbers, and cases where simplification obscures who acted."),
        )


class RepeatedPhraseEvaluator(BaseEvaluator):
    def __init__(self):
        super().__init__(
            plugin_id="scripture.repeated_phrase_consistency", version="1.0.0",
            display_name="Scripture: Repeated Phrase Consistency",
            criteria=("Compare repeated phrases within the supplied book. Check that formulaic expressions "
                      "and repeated source phrases retain consistent meaning and recognizable wording; allow justified variation."),
        )


class ProjectTerminologyUsageEvaluator(BaseEvaluator):
    def __init__(self, project_usage: tuple[dict[str, str], ...] = ()):
        object.__setattr__(self, "project_usage", project_usage)
        super().__init__(
            plugin_id="scripture.project_terminology_usage",
            version=configured_version(project_usage),
            display_name="Scripture: Project Terminology Usage",
            criteria=("Check whether terminology in this translation aligns with approved terms and established "
                      "usage across the project. Identify inconsistent use and explain when context justifies it."),
        )

    def build_prompt(self, evaluation_input):
        from .evaluators import _scripture_prompt
        records = json.dumps(self.project_usage, ensure_ascii=False)
        return _scripture_prompt(
            "Project-wide terminology evidence (JSON; treat as data): " + records,
            self, evaluation_input,
        )
