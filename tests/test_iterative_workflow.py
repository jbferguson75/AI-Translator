import json

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from ai_translator.db.base import Base
from ai_translator.db.models import Document, Project, Translation, TranslationVersion, TranslationVersionParent
from ai_translator.translation.iteration import has_converged
from ai_translator.translation.reporting import lineage_report
from ai_translator.evaluation.plugins import FidelityEvaluator


def test_convergence_requires_both_gains_below_threshold():
    assert has_converged((90.0, 0.90), (90.2, 0.905), 0.5, 0.01)
    assert not has_converged((90.0, 0.90), (90.8, 0.905), 0.5, 0.01)
    assert not has_converged((90.0, 0.90), (90.2, 0.92), 0.5, 0.01)


def test_evaluator_parses_structured_feedback_and_legacy_issue_strings():
    evaluator = FidelityEvaluator()
    structured = evaluator.parse_output(json.dumps({
        "score": 87, "confidence": 0.91, "reasoning": "Meaning is mostly preserved.",
        "strengths": ["Clear wording"],
        "issues": [{"type": "fidelity", "severity": "high", "description": "Meaning weakened"}],
        "recommendations": ["Restore key term"],
    }))
    assert structured.strengths == ("Clear wording",)
    assert structured.issue_details[0] == {
        "type": "fidelity", "severity": "high", "description": "Meaning weakened",
    }
    legacy = evaluator.parse_output(json.dumps({
        "score": 80, "confidence": 0.8, "reasoning": "Some omission.",
        "issues": ["An adjective was omitted"], "recommendations": [],
    }))
    assert legacy.issue_details[0]["description"] == "An adjective was omitted"


def test_lineage_report_follows_all_synthesis_candidates():
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    project = Project(name="lineage")
    session.add(project)
    session.flush()
    document = Document(project_id=project.id, title="source", domain="general", source_locale="en",
                        source_hash="a" * 64, source_text="source")
    session.add(document)
    session.flush()
    translation = Translation(document_id=document.id, target_locale="fr", task_kind="translation")
    session.add(translation)
    session.flush()
    originals = [TranslationVersion(translation_id=translation.id, version_number=i,
                                    artifact_kind="ORIGINAL_TRANSLATION") for i in (1, 2)]
    session.add_all(originals)
    session.flush()
    synthesis = TranslationVersion(translation_id=translation.id, version_number=3,
                                   artifact_kind="SYNTHESIZED_TRANSLATION",
                                   parent_version_id=originals[0].id)
    session.add(synthesis)
    session.flush()
    session.add_all([
        TranslationVersionParent(child_version_id=synthesis.id, parent_version_id=originals[0].id,
                                 relationship_kind="source_candidate", ordinal=0),
        TranslationVersionParent(child_version_id=synthesis.id, parent_version_id=originals[1].id,
                                 relationship_kind="source_candidate", ordinal=1),
    ])
    session.commit()

    report = lineage_report(session, synthesis.id)

    assert set(report["artifacts"]) == {str(row.id) for row in [*originals, synthesis]}
    assert {edge["kind"] for edge in report["edges"]} == {"parent", "source_candidate"}
    assert any(edge["parent"] == str(originals[1].id) for edge in report["edges"])
    session.close()
    engine.dispose()
