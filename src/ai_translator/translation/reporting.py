from __future__ import annotations

from collections import defaultdict
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import (Evaluation, EvaluationResult, EvaluationVersion, Experiment, ExperimentRun,
                         Chunk, ModelRun, TranslationSegment, TranslationVersion,
                         TranslationVersionParent)


def lineage_report(session: Session, version_id: UUID) -> dict:
    root = session.get(TranslationVersion, version_id)
    if root is None:
        raise ValueError(f"Translation version {version_id} does not exist")
    seen: set[UUID] = set()
    items: dict[str, dict] = {}
    edges: list[dict] = []
    stack = [version_id]
    while stack:
        current_id = stack.pop()
        if current_id in seen:
            continue
        seen.add(current_id)
        version = session.get(TranslationVersion, current_id)
        run = version.model_run
        items[str(current_id)] = {
            "version_number": version.version_number, "artifact_kind": version.artifact_kind,
            "created_at": version.created_at.isoformat(), "status": version.status,
            "model": run.model_name if run else None,
            "prompt_version_id": str(version.prompt_version_id) if version.prompt_version_id else None,
        }
        links = session.scalars(select(TranslationVersionParent).where(
            TranslationVersionParent.child_version_id == current_id
        ).order_by(TranslationVersionParent.ordinal)).all()
        has_primary_parent = any(link.parent_version_id == version.parent_version_id
                                 and link.relationship_kind == "parent" for link in links)
        if version.parent_version_id and not has_primary_parent:
            edges.append({"parent": str(version.parent_version_id), "child": str(current_id), "kind": "parent"})
            stack.append(version.parent_version_id)
        for link in links:
            edges.append({"parent": str(link.parent_version_id), "child": str(current_id),
                          "kind": link.relationship_kind})
            stack.append(link.parent_version_id)
    return {"root_version_id": str(version_id), "artifacts": items, "edges": edges}


def comparison_report(session: Session, version_ids: list[UUID]) -> dict:
    candidates = []
    for version_id in version_ids:
        version = session.get(TranslationVersion, version_id)
        if version is None:
            raise ValueError(f"Translation version {version_id} does not exist")
        rows = session.execute(
            select(Evaluation, EvaluationResult)
            .join(EvaluationVersion, EvaluationVersion.evaluation_id == Evaluation.id)
            .join(EvaluationResult, EvaluationResult.evaluation_version_id == EvaluationVersion.id)
            .where(EvaluationVersion.translation_version_id == version_id,
                   EvaluationVersion.status == "completed", EvaluationResult.metric_key == "overall")
            .order_by(EvaluationVersion.created_at.desc())
        ).all()
        latest: dict[str, tuple] = {}
        for evaluation, result in rows:
            latest.setdefault(evaluation.evaluator_name or result.plugin_id, (evaluation, result))
        categories = {name: {"score": result.score, "confidence": result.confidence,
                             "reasoning": result.reasoning, "strengths": result.strengths,
                             "issues": result.issue_details, "recommendations": result.recommendations}
                      for name, (_evaluation, result) in latest.items()}
        values = [r.score for _e, r in latest.values() if r.score is not None]
        candidates.append({"version_id": str(version_id), "version_number": version.version_number,
                           "artifact_kind": version.artifact_kind,
                           "overall_score": sum(values) / len(values) if values else None,
                           "categories": categories})
    scored = [item for item in candidates if item["overall_score"] is not None]
    top = max(scored, key=lambda item: item["overall_score"]) if scored else None
    category_scores: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for candidate in candidates:
        for category, value in candidate["categories"].items():
            if value["score"] is not None:
                category_scores[category].append((value["score"], candidate["version_id"]))
    return {"candidates": candidates, "highest_average_score_version_id": top["version_id"] if top else None,
            "best_version_by_evaluator": {
                category: max(values, key=lambda item: item[0])[1]
                for category, values in category_scores.items()
            },
            "selection_note": "This report compares saved evaluator evidence; it does not approve or select a translation."}


def improvement_report(session: Session, version_id: UUID, baseline_id: UUID | None = None) -> dict:
    version = session.get(TranslationVersion, version_id)
    if version is None:
        raise ValueError(f"Translation version {version_id} does not exist")
    parent_id = baseline_id or version.parent_version_id
    if parent_id is None:
        parents = session.scalars(select(TranslationVersionParent.parent_version_id).where(
            TranslationVersionParent.child_version_id == version_id
        ).order_by(TranslationVersionParent.ordinal)).all()
        parent_id = parents[0] if parents else None
    if parent_id is None:
        raise ValueError("No baseline parent is recorded; pass --baseline VERSION_UUID")
    comparison = comparison_report(session, [parent_id, version_id])
    before, after = comparison["candidates"]
    before_scores, after_scores = before["categories"], after["categories"]
    categories = {name: {"baseline": data["score"], "candidate": after_scores[name]["score"],
                         "delta": after_scores[name]["score"] - data["score"]}
                  for name, data in before_scores.items() if name in after_scores
                  and data["score"] is not None and after_scores[name]["score"] is not None}
    return {"baseline_version_id": str(parent_id), "candidate_version_id": str(version_id),
            "overall_score_delta": (after["overall_score"] - before["overall_score"]
                                    if after["overall_score"] is not None and before["overall_score"] is not None else None),
            "category_deltas": categories}


def convergence_report(session: Session, experiment_id: UUID) -> dict:
    experiment = session.get(Experiment, experiment_id)
    if experiment is None:
        raise ValueError(f"Experiment {experiment_id} does not exist")
    runs = session.scalars(select(ExperimentRun).where(
        ExperimentRun.experiment_id == experiment_id
    ).order_by(ExperimentRun.created_at, ExperimentRun.id)).all()
    return {"experiment_id": str(experiment_id), "configuration": experiment.configuration,
            "history": [{"label": row.label, "translation_version_id": str(row.translation_version_id)
                         if row.translation_version_id else None,
                         "evaluation_version_id": str(row.evaluation_version_id)
                         if row.evaluation_version_id else None,
                         "parameters": row.parameters} for row in runs]}


def side_by_side_report(session: Session, version_ids: list[UUID]) -> str:
    """Render aligned source and translation segments as a Markdown comparison table."""
    unique_ids = list(dict.fromkeys(version_ids))
    if not unique_ids:
        raise ValueError("Provide at least one translation version UUID")
    versions = []
    for version_id in unique_ids:
        version = session.get(TranslationVersion, version_id)
        if version is None:
            raise ValueError(f"Translation version {version_id} does not exist")
        versions.append(version)
    translation_id = versions[0].translation_id
    if any(version.translation_id != translation_id for version in versions):
        raise ValueError("Side-by-side versions must belong to the same translation task")

    aligned: dict[str, dict] = {}
    labels = []
    for version in versions:
        run = version.model_run
        manifest = version.run_manifest or {}
        model_label = (run.model_name if run else None) or manifest.get("model_name") or version.author_kind
        label = (f"v{version.version_number} · {version.artifact_kind} · {model_label} "
                 f"({str(version.id)[:8]})")
        labels.append(label)
        rows = session.execute(
            select(TranslationSegment.target_text, Chunk.stable_key, Chunk.source_text, Chunk.ordinal,
                   Chunk.locator)
            .join(Chunk, Chunk.id == TranslationSegment.chunk_id)
            .where(TranslationSegment.translation_version_id == version.id)
            .order_by(Chunk.ordinal)
        ).all()
        if not rows:
            raise ValueError(f"Translation version {version.id} has no translated segments")
        for target, stable_key, source, ordinal, locator in rows:
            item = aligned.setdefault(stable_key, {
                "source": source, "ordinal": ordinal, "locator": locator or {}, "translations": {},
            })
            if item["source"] != source:
                raise ValueError(f"Source text differs for aligned chunk {stable_key}")
            item["translations"][str(version.id)] = target

    def cell(value: str | None) -> str:
        if value is None:
            return "—"
        return value.replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")

    translation = versions[0].translation
    document = translation.document
    lines = [
        f"# Side-by-side translation report: {document.title}",
        "",
        f"- Target locale: {translation.target_locale}",
        f"- Task: {translation.task_kind}",
        f"- Reading level: {translation.target_reading_level or 'not specified'}",
        "",
        "| Source unit | Source text | " + " | ".join(cell(label) for label in labels) + " |",
        "|---|---|" + "---|" * len(labels),
    ]
    for stable_key, item in sorted(aligned.items(), key=lambda pair: (pair[1]["ordinal"], pair[0])):
        locator = item["locator"]
        reference = locator.get("reference") or stable_key
        values = [cell(item["source"])]
        values.extend(cell(item["translations"].get(str(version.id))) for version in versions)
        lines.append("| " + cell(str(reference)) + " | " + " | ".join(values) + " |")
    lines.extend(["", "— = no translation segment exists for this source unit.", ""])
    return "\n".join(lines)
