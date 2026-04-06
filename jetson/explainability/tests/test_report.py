"""Tests for explanation report generation module."""

import json
import pytest
from jetson.explainability.report import ExplanationSection, ExplanationReport
from jetson.explainability.attribution import FeatureImportance


# --- Fixtures ---

@pytest.fixture
def report():
    return ExplanationReport(title="Test Report", model_name="test_model")


@pytest.fixture
def sample_section():
    return ExplanationSection(
        title="Feature Analysis",
        content="The primary features driving the model are x1 and x2.",
        importance_level="high",
    )


# --- ExplanationSection tests ---

class TestExplanationSection:
    def test_creation(self):
        s = ExplanationSection("Title", "Content", "critical")
        assert s.title == "Title"
        assert s.content == "Content"
        assert s.importance_level == "critical"

    def test_default_importance(self):
        s = ExplanationSection("Title", "Content")
        assert s.importance_level == "normal"

    def test_invalid_importance_defaults_normal(self):
        s = ExplanationSection("Title", "Content", "invalid")
        assert s.importance_level == "normal"

    def test_all_importance_levels(self):
        for level in ["critical", "high", "normal", "low"]:
            s = ExplanationSection("T", "C", level)
            assert s.importance_level == level


# --- ExplanationReport tests ---

class TestExplanationReport:
    def test_init(self):
        r = ExplanationReport(title="My Report", model_name="my_model")
        assert r.title == "My Report"
        assert r.model_name == "my_model"

    def test_default_init(self):
        r = ExplanationReport()
        assert r.title == "NEXUS Explanation Report"
        assert r.model_name == "unknown"

    def test_add_section(self, report, sample_section):
        report.add_section(sample_section)
        assert report.section_count == 1

    def test_add_multiple_sections(self, report):
        for i in range(5):
            report.add_section(ExplanationSection(f"Section {i}", f"Content {i}"))
        assert report.section_count == 5

    def test_generate_text_report_basic(self, report, sample_section):
        report.add_section(sample_section)
        text = report.generate_text_report()
        assert "Test Report" in text
        assert "test_model" in text
        assert "Feature Analysis" in text
        assert "The primary features" in text

    def test_generate_text_report_empty(self, report):
        text = report.generate_text_report()
        assert "Test Report" in text

    def test_generate_text_report_importance_levels(self, report):
        for level in ["critical", "high", "normal", "low"]:
            report.add_section(ExplanationSection(f"{level} section", "content", level))
        text = report.generate_text_report()
        assert "CRITICAL" in text
        assert "HIGH" in text
        assert "NORMAL" in text
        assert "LOW" in text

    def test_generate_text_report_contains_markers(self, report):
        report.add_section(ExplanationSection("crit", "c", "critical"))
        report.add_section(ExplanationSection("high", "c", "high"))
        report.add_section(ExplanationSection("normal", "c", "normal"))
        report.add_section(ExplanationSection("low", "c", "low"))
        text = report.generate_text_report()
        assert "!!!" in text  # critical marker
        assert "!!" in text   # high marker

    def test_generate_text_report_with_visualization(self, report):
        report.add_section(ExplanationSection("test", "content"))
        report.add_visualization({"data": [1, 2, 3]}, "bar_chart", "Feature importance")
        text = report.generate_text_report()
        assert "bar_chart" in text
        assert "Feature importance" in text

    def test_generate_structured_report_basic(self, report, sample_section):
        report.add_section(sample_section)
        structured = report.generate_structured_report()
        assert structured["title"] == "Test Report"
        assert structured["model_name"] == "test_model"
        assert "created_at" in structured
        assert len(structured["sections"]) == 1
        assert structured["sections"][0]["title"] == "Feature Analysis"

    def test_generate_structured_report_empty(self, report):
        structured = report.generate_structured_report()
        assert structured["sections"] == []
        assert structured["visualizations"] == []

    def test_generate_structured_report_summary(self, report, sample_section):
        report.add_section(sample_section)
        structured = report.generate_structured_report()
        assert "summary" in structured
        assert isinstance(structured["summary"], str)

    def test_add_visualization(self, report):
        report.add_visualization([1, 2, 3], "bar", "test chart")
        assert len(report.visualizations) == 1
        assert report.visualizations[0]["type"] == "bar"
        assert report.visualizations[0]["description"] == "test chart"
        assert report.visualizations[0]["data"] == [1, 2, 3]

    def test_add_visualization_no_description(self, report):
        report.add_visualization({}, "scatter")
        assert report.visualizations[0]["description"] == ""

    def test_add_multiple_visualizations(self, report):
        report.add_visualization({}, "bar")
        report.add_visualization({}, "scatter")
        report.add_visualization({}, "line")
        assert len(report.visualizations) == 3

    def test_generate_summary_empty(self, report):
        summary = report.generate_summary()
        assert "No sections" in summary

    def test_generate_summary_basic(self, report):
        report.add_section(ExplanationSection("Sec 1", "content"))
        summary = report.generate_summary()
        assert "1 sections" in summary
        assert "test_model" in summary

    def test_generate_summary_with_critical(self, report):
        report.add_section(ExplanationSection("Critical Finding", "content", "critical"))
        summary = report.generate_summary()
        assert "1 critical finding" in summary

    def test_generate_summary_with_high(self, report):
        report.add_section(ExplanationSection("Important", "content", "high"))
        summary = report.generate_summary()
        assert "1 high-priority finding" in summary

    def test_generate_summary_key_findings(self, report):
        report.add_section(ExplanationSection("Bias detected", "content", "critical"))
        report.add_section(ExplanationSection("Low confidence", "content", "high"))
        summary = report.generate_summary()
        assert "Key findings" in summary
        assert "Bias detected" in summary

    def test_generate_summary_multiple_critical(self, report):
        report.add_section(ExplanationSection("C1", "c", "critical"))
        report.add_section(ExplanationSection("C2", "c", "critical"))
        summary = report.generate_summary()
        assert "2 critical finding" in summary

    def test_generate_technical_details_basic(self, report):
        attrs = [
            FeatureImportance("x1", 0.6, "positive", 0.3),
            FeatureImportance("x2", 0.4, "negative", -0.2),
        ]
        model_info = {"name": "nav_model", "version": "2.0", "type": "neural_net", "accuracy": 0.95}
        details = report.generate_technical_details(attrs, model_info)
        assert "nav_model" in details
        assert "2.0" in details
        assert "neural_net" in details
        assert "0.95" in details
        assert "x1" in details
        assert "x2" in details

    def test_generate_technical_details_empty(self, report):
        details = report.generate_technical_details([], {})
        assert "unknown" in details

    def test_generate_technical_details_sorted(self, report):
        attrs = [
            FeatureImportance("low", 0.1, "positive", 0.1),
            FeatureImportance("high", 0.9, "positive", 0.9),
        ]
        details = report.generate_technical_details(attrs, {"name": "m", "version": "1", "type": "t", "accuracy": "a"})
        lines = details.split("\n")
        # high importance should appear first
        high_idx = next(i for i, l in enumerate(lines) if "high" in l)
        low_idx = next(i for i, l in enumerate(lines) if "low" in l)
        assert high_idx < low_idx

    def test_generate_actionable_recommendations_no_issues(self, report):
        recs = report.generate_actionable_recommendations({})
        assert len(recs) == 1
        assert "No immediate actions" in recs[0]

    def test_generate_actionable_recommendations_bias(self, report):
        analysis = {"biased_features": ["x1", "x2"]}
        recs = report.generate_actionable_recommendations(analysis)
        assert any("bias" in r.lower() and "x1" in r for r in recs)

    def test_generate_actionable_recommendations_low_confidence(self, report):
        analysis = {"avg_confidence": 0.5}
        recs = report.generate_actionable_recommendations(analysis)
        assert any("confidence" in r.lower() and "0.50" in r for r in recs)

    def test_generate_actionable_recommendations_high_complexity(self, report):
        analysis = {"complexity_score": 0.9}
        recs = report.generate_actionable_recommendations(analysis)
        assert any("complexity" in r.lower() for r in recs)

    def test_generate_actionable_recommendations_anomalies(self, report):
        analysis = {"anomaly_count": 5}
        recs = report.generate_actionable_recommendations(analysis)
        assert any("5" in r and "anomal" in r.lower() for r in recs)

    def test_generate_actionable_recommendations_low_fairness(self, report):
        analysis = {"fairness_scores": {"x1": 0.5, "x2": 0.9}}
        recs = report.generate_actionable_recommendations(analysis)
        assert any("x1" in r and "fairness" in r.lower() for r in recs)

    def test_generate_actionable_recommendations_data_quality(self, report):
        analysis = {"data_quality_score": 0.6}
        recs = report.generate_actionable_recommendations(analysis)
        assert any("data quality" in r.lower() for r in recs)

    def test_generate_actionable_recommendations_multiple_issues(self, report):
        analysis = {
            "biased_features": ["x1"],
            "avg_confidence": 0.5,
            "complexity_score": 0.9,
            "anomaly_count": 3,
        }
        recs = report.generate_actionable_recommendations(analysis)
        assert len(recs) >= 3

    def test_export_report_json(self, report, sample_section):
        report.add_section(sample_section)
        exported = report.export_report("json")
        parsed = json.loads(exported)
        assert parsed["title"] == "Test Report"
        assert len(parsed["sections"]) == 1

    def test_export_report_text(self, report, sample_section):
        report.add_section(sample_section)
        exported = report.export_report("text")
        assert "Test Report" in exported

    def test_export_report_invalid_format(self, report):
        with pytest.raises(ValueError, match="Unsupported export format"):
            report.export_report("xml")

    def test_section_count(self, report):
        assert report.section_count == 0
        report.add_section(ExplanationSection("t", "c"))
        assert report.section_count == 1

    def test_has_critical_findings_true(self, report):
        report.add_section(ExplanationSection("t", "c", "critical"))
        assert report.has_critical_findings is True

    def test_has_critical_findings_false(self, report):
        report.add_section(ExplanationSection("t", "c", "normal"))
        assert report.has_critical_findings is False

    def test_has_critical_findings_empty(self, report):
        assert report.has_critical_findings is False

    def test_text_report_contains_equals_border(self, report):
        text = report.generate_text_report()
        assert "====" in text

    def test_structured_report_roundtrip(self, report, sample_section):
        report.add_section(sample_section)
        structured = report.generate_structured_report()
        # Re-serialize and verify
        reserialized = json.dumps(structured)
        reparsed = json.loads(reserialized)
        assert reparsed["title"] == "Test Report"
