from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import EvaluationResult, EvaluationVersion, Experiment, ExperimentRun, TranslationVersion
from ..evaluation.config import EvaluationConfig
from ..evaluation.errors import EvaluationError
from ..evaluation.service import EvaluationService
from ..evaluation.registry import EvaluatorRegistry
from ..translation.errors import TranslationError
from .config import TranslationConfig
from .ollama import OllamaClient
from .refinement import TranslationRefinementService


@dataclass(frozen=True, slots=True)
class IterationOutcome:
    experiment_id: UUID
    selected_version_id: UUID
    iterations: int
    score: float
    confidence: float
    converged: bool
    reason: str


def score_confidence(session: Session, version_id: UUID) -> tuple[float, float]:
    rows = session.execute(
        select(EvaluationResult.score, EvaluationResult.confidence)
        .join(EvaluationVersion, EvaluationVersion.id == EvaluationResult.evaluation_version_id)
        .where(EvaluationVersion.translation_version_id == version_id,
               EvaluationVersion.status == "completed", EvaluationResult.metric_key == "overall")
    ).all()
    usable = [(score, confidence) for score, confidence in rows if score is not None]
    if not usable:
        raise TranslationError(f"No completed overall evaluation scores for translation version {version_id}")
    return (sum(score for score, _ in usable) / len(usable),
            sum(confidence or 0.0 for _, confidence in usable) / len(usable))


def has_converged(previous: tuple[float, float], current: tuple[float, float],
                  minimum_score_increase: float, minimum_confidence_increase: float) -> bool:
    return (current[0] - previous[0] < minimum_score_increase
            and current[1] - previous[1] < minimum_confidence_increase)


class TranslationIterationService:
    """Run bounded refine/evaluate iterations and persist their full experiment history."""

    def __init__(self, session: Session, translation_client: OllamaClient,
                 evaluation_client: OllamaClient):
        self.session = session
        self.translation_client = translation_client
        self.evaluation_client = evaluation_client

    def run(self, *, baseline_version_id: UUID, translation_config: TranslationConfig,
            evaluation_config: EvaluationConfig, max_iterations: int | None = None,
            minimum_score_increase: float | None = None, minimum_confidence_increase: float | None = None,
            translation_models: list[str] | None = None, evaluator_models: list[str] | None = None,
            plugins: list[str] | None = None,
            evaluation_registry: EvaluatorRegistry | None = None,
            progress: Callable[[str], None] | None = None) -> IterationOutcome:
        baseline = self.session.get(TranslationVersion, baseline_version_id)
        if baseline is None:
            raise TranslationError(f"Translation version {baseline_version_id} does not exist")
        project_id = baseline.translation.document.project_id
        project_settings = baseline.translation.document.project.settings or {}
        iteration_settings = project_settings.get("translation_iteration", {})
        if not isinstance(iteration_settings, dict):
            iteration_settings = {}
        max_iterations = max_iterations if max_iterations is not None else int(iteration_settings.get("max_iterations", 3))
        minimum_score_increase = (minimum_score_increase if minimum_score_increase is not None
                                  else float(iteration_settings.get("minimum_score_increase", 0.5)))
        minimum_confidence_increase = (minimum_confidence_increase if minimum_confidence_increase is not None
                                       else float(iteration_settings.get("minimum_confidence_increase", 0.01)))
        if max_iterations < 1:
            raise TranslationError("max_iterations must be at least 1")
        if minimum_score_increase < 0 or minimum_confidence_increase < 0:
            raise TranslationError("convergence thresholds must be non-negative")
        experiment = Experiment(
            project_id=project_id, name=f"Iterative improvement {baseline_version_id}",
            description="Bounded feedback refinement with post-pass evaluation.",
            configuration={"baseline_version_id": str(baseline_version_id),
                           "max_iterations": max_iterations,
                           "minimum_score_increase": minimum_score_increase,
                           "minimum_confidence_increase": minimum_confidence_increase,
                           "translation_models": translation_models,
                           "evaluator_models": evaluator_models, "plugins": plugins,
                           "status": "running"},
        )
        self.session.add(experiment)
        self.session.commit()
        current_id = baseline_version_id
        current_metrics = score_confidence(self.session, current_id)
        self.session.add(ExperimentRun(experiment_id=experiment.id, translation_version_id=current_id,
                                       label="baseline", parameters={"score": current_metrics[0],
                                                                      "confidence": current_metrics[1]}))
        self.session.commit()
        converged, reason, count = False, "maximum iterations reached", 0
        try:
            for iteration in range(1, max_iterations + 1):
                if progress:
                    progress(f"Iteration {iteration}/{max_iterations}: refining version {current_id}")
                candidates = TranslationRefinementService(self.session, self.translation_client).refine_version(
                    translation_version_id=current_id, config=translation_config,
                    model_selection=translation_models, progress_callback=progress,
                )
                evaluated = []
                for candidate in candidates:
                    if candidate.status not in {"completed", "partial"}:
                        continue
                    results = EvaluationService(self.session, self.evaluation_client,
                                                evaluation_registry).evaluate_translation_version(
                        translation_version_id=candidate.version_id, config=evaluation_config,
                        plugin_selection=plugins, model_selection=evaluator_models,
                        progress_callback=progress,
                    )
                    metrics = score_confidence(self.session, candidate.version_id)
                    eval_id = next((r.evaluation_version_id for r in results if r.status == "completed"), None)
                    evaluated.append((metrics, candidate.version_id, eval_id))
                if not evaluated:
                    raise TranslationError(f"Iteration {iteration} produced no evaluable translation candidates")
                best_metrics, best_id, best_eval_id = max(evaluated, key=lambda item: (item[0][0], item[0][1]))
                for metrics, version_id, evaluation_id in evaluated:
                    self.session.add(ExperimentRun(
                        experiment_id=experiment.id, translation_version_id=version_id,
                        evaluation_version_id=evaluation_id, label=f"iteration-{iteration}",
                        parameters={"score": metrics[0], "confidence": metrics[1],
                                    "score_delta": metrics[0] - current_metrics[0],
                                    "confidence_delta": metrics[1] - current_metrics[1],
                                    "selected": version_id == best_id},
                    ))
                self.session.commit()
                count = iteration
                if has_converged(current_metrics, best_metrics, minimum_score_increase,
                                 minimum_confidence_increase):
                    converged, reason = True, "score and confidence gains were below configured thresholds"
                    current_id = best_id
                    current_metrics = best_metrics
                    break
                current_id, current_metrics = best_id, best_metrics
            experiment = self.session.get(Experiment, experiment.id)
            experiment.configuration = {**experiment.configuration, "status": "completed",
                                        "converged": converged, "convergence_reason": reason,
                                        "iterations_completed": count,
                                        "selected_version_id": str(current_id),
                                        "selected_score": current_metrics[0],
                                        "selected_confidence": current_metrics[1]}
            self.session.commit()
            return IterationOutcome(experiment.id, current_id, count, *current_metrics, converged, reason)
        except Exception:
            self.session.rollback()
            experiment = self.session.get(Experiment, experiment.id)
            if experiment:
                experiment.configuration = {**experiment.configuration, "status": "failed",
                                            "iterations_completed": count}
                self.session.commit()
            raise
