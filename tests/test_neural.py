"""
Tests for the haven-engine neural pipeline.

Covers:
- PII masking correctness (regex and Presidio modes)
- Embedding pipeline vector normalization
- Score function range validation [0, 1]
- Ranking determinism for identical inputs
"""

import sys
import os
import numpy as np
import pytest

# Add src to path so we can import the neural package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from neural.pii_masking import PIIMasker, MaskedResult
from neural.embedding_pipeline import (
    EmbeddingPipeline,
    child_profile_to_text,
    family_profile_to_text,
)


# ---------------------------------------------------------------------------
# PII Masking Tests
# ---------------------------------------------------------------------------


class TestPIIMasker:
    """Tests for the PII masking module."""

    def test_ssn_masked(self):
        """SSNs should be detected and masked."""
        masker = PIIMasker(enable_presidio=False)
        result = masker.mask("SSN: 123-45-6789")
        assert "123-45-6789" not in result.masked_text
        assert result.entities_detected > 0
        assert "US_SSN" in result.entity_types

    def test_case_number_masked(self):
        """Child welfare case numbers should be detected."""
        masker = PIIMasker(enable_presidio=False)
        result = masker.mask("Case CW-2026-001234 filed today")
        assert "CW-2026-001234" not in result.masked_text
        assert "CASE_NUMBER" in result.entity_types

    def test_docket_id_masked(self):
        """Court docket IDs should be detected."""
        masker = PIIMasker(enable_presidio=False)
        result = masker.mask("Docket JUV-2026-0456")
        assert "JUV-2026-0456" not in result.masked_text
        assert "DOCKET_ID" in result.entity_types

    def test_placement_id_masked(self):
        """Placement IDs should be detected."""
        masker = PIIMasker(enable_presidio=False)
        result = masker.mask("Placement PLC-0012345 assigned")
        assert "PLC-0012345" not in result.masked_text
        assert "PLACEMENT_ID" in result.entity_types

    def test_phone_masked(self):
        """Phone numbers should be detected."""
        masker = PIIMasker(enable_presidio=False)
        result = masker.mask("Call (555) 123-4567 for details")
        assert "(555) 123-4567" not in result.masked_text
        assert "PHONE_NUMBER" in result.entity_types

    def test_date_masked(self):
        """Dates of birth should be detected."""
        masker = PIIMasker(enable_presidio=False)
        result = masker.mask("Born on 3/15/2018")
        assert "3/15/2018" not in result.masked_text
        assert "DATE_TIME" in result.entity_types

    def test_empty_string(self):
        """Empty strings should return clean result."""
        masker = PIIMasker(enable_presidio=False)
        result = masker.mask("")
        assert result.masked_text == ""
        assert result.entities_detected == 0

    def test_no_pii_passthrough(self):
        """Text without PII should pass through unchanged."""
        masker = PIIMasker(enable_presidio=False)
        text = "The child enjoys painting and reading."
        result = masker.mask(text)
        assert result.masked_text == text
        assert result.entities_detected == 0

    def test_multiple_pii_entities(self):
        """Multiple PII types should all be detected."""
        masker = PIIMasker(enable_presidio=False)
        result = masker.mask(
            "Case CW-2026-001234, SSN 123-45-6789, phone (555) 123-4567"
        )
        assert result.entities_detected >= 3
        assert "CASE_NUMBER" in result.entity_types
        assert "US_SSN" in result.entity_types
        assert "PHONE_NUMBER" in result.entity_types

    def test_mask_profile_recursive(self):
        """Profile masking should recursively process nested dicts."""
        masker = PIIMasker(enable_presidio=False)
        profile = {
            "name": "John Smith",
            "ssn": "123-45-6789",
            "case": "CW-2026-001234",
            "address": {
                "street": "123 Main St",
                "phone": "(555) 123-4567",
            },
            "age": 8,  # Non-string should pass through.
            "notes": ["Born on 3/15/2018", "No issues"],
        }
        masked = masker.mask_profile(profile)
        assert "123-45-6789" not in masked["ssn"]
        assert "CW-2026-001234" not in masked["case"]
        assert "(555) 123-4567" not in masked["address"]["phone"]
        assert masked["age"] == 8
        assert "3/15/2018" not in masked["notes"][0]

    def test_masked_result_metadata(self):
        """MaskedResult should contain correct metadata."""
        masker = PIIMasker(enable_presidio=False)
        text = "SSN: 123-45-6789"
        result = masker.mask(text)
        assert isinstance(result, MaskedResult)
        assert result.original_length == len(text)
        assert result.masked_length > 0
        assert isinstance(result.entity_types, list)


# ---------------------------------------------------------------------------
# PII Masking with Presidio Tests
# ---------------------------------------------------------------------------


class TestPIIMaskerPresidio:
    """Tests for Presidio-enabled PII masking."""

    def test_presidio_detects_person(self):
        """Presidio should detect person names."""
        masker = PIIMasker(enable_presidio=True)
        result = masker.mask("John Smith is the case worker.")
        assert result.entities_detected > 0

    def test_presidio_custom_recognizers(self):
        """Custom child welfare recognizers should be active."""
        masker = PIIMasker(enable_presidio=True)
        result = masker.mask("Case CW-2026-001234 assigned to docket JUV-2026-0456")
        # Should detect both custom entities.
        assert "CW-2026-001234" not in result.masked_text
        assert "JUV-2026-0456" not in result.masked_text


# ---------------------------------------------------------------------------
# Embedding Pipeline Tests
# ---------------------------------------------------------------------------


class TestEmbeddingPipeline:
    """Tests for the neural embedding pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance (may use fallback embeddings)."""
        return EmbeddingPipeline()

    def test_encode_child_returns_normalized_vector(self, pipeline):
        """Child embedding should be a unit vector."""
        profile = {
            "age": 8,
            "trauma_flags": ["neglect"],
            "required_care_level": "Moderate",
        }
        vec = pipeline.encode_child(profile)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5, "Vector should be normalized"

    def test_encode_family_returns_normalized_vector(self, pipeline):
        """Family embedding should be a unit vector."""
        profile = {
            "accepted_care_levels": ["Basic", "Moderate"],
            "safety_certifications": ["FirstAidCPR", "TraumaInformedCare"],
            "accepted_age_range": (5, 12),
        }
        vec = pipeline.encode_family(profile)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_score_range_zero_to_one(self, pipeline):
        """Score should always be in [0, 1]."""
        child_vec = pipeline.encode_child({"age": 8})
        family_vec = pipeline.encode_family({"accepted_care_levels": ["Basic"]})
        score = pipeline.score(child_vec, family_vec)
        assert 0.0 <= score <= 1.0, f"Score {score} outside [0, 1]"

    def test_identical_profiles_high_score(self, pipeline):
        """Identical text should produce a score near 1.0."""
        text = "Requires moderate care, history of neglect, age 8."
        vec = pipeline.encode_text(text)
        score = pipeline.score(vec, vec)
        assert score > 0.99, f"Self-similarity should be ~1.0, got {score}"

    def test_ranking_deterministic(self, pipeline):
        """Ranking the same inputs twice should produce identical results."""
        child = {"age": 10, "trauma_flags": ["neglect"], "required_care_level": "Basic"}
        families = [
            {"id": "F1", "accepted_care_levels": ["Basic"]},
            {"id": "F2", "accepted_care_levels": ["Moderate", "Treatment"]},
            {"id": "F3", "accepted_care_levels": ["Basic", "Moderate"]},
        ]

        ranking1 = pipeline.rank(child, families)
        ranking2 = pipeline.rank(child, families)

        ids1 = [m.family_id for m in ranking1]
        ids2 = [m.family_id for m in ranking2]
        assert ids1 == ids2, "Rankings should be deterministic"

        scores1 = [m.score for m in ranking1]
        scores2 = [m.score for m in ranking2]
        for s1, s2 in zip(scores1, scores2):
            assert abs(s1 - s2) < 1e-9, f"Scores should be identical: {s1} vs {s2}"

    def test_rank_returns_sorted_descending(self, pipeline):
        """Rankings should be sorted by score descending."""
        child = {"age": 8, "trauma_flags": ["neglect"]}
        families = [
            {"id": "F1", "accepted_care_levels": ["Basic"]},
            {"id": "F2", "accepted_care_levels": ["Treatment"]},
            {"id": "F3", "accepted_care_levels": ["Basic", "Moderate"]},
        ]

        ranking = pipeline.rank(child, families)
        scores = [m.score for m in ranking]
        assert scores == sorted(scores, reverse=True), "Rankings should be descending"

    def test_batch_encode(self, pipeline):
        """Batch encoding should produce correct matrix dimensions."""
        texts = [
            "Child needs moderate care.",
            "Family certified for treatment.",
            "Experienced with trauma recovery.",
        ]
        matrix = pipeline.batch_encode(texts)
        assert matrix.shape == (3, 384)
        # Each row should be normalized.
        for i in range(3):
            assert abs(np.linalg.norm(matrix[i]) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Profile-to-Text Template Tests
# ---------------------------------------------------------------------------


class TestProfileTemplates:
    """Tests for the profile-to-text conversion templates."""

    def test_child_profile_with_all_fields(self):
        """Child profile with all fields should produce rich text."""
        profile = {
            "age": 10,
            "trauma_flags": ["neglect", "abandonment"],
            "required_care_level": "Moderate",
            "behavioral_notes": "Responds well to structure",
            "educational_needs": "IEP for reading",
            "social_preferences": "Prefers small groups",
            "sibling_group_id": "SIB-001",
        }
        text = child_profile_to_text(profile)
        assert "neglect" in text.lower()
        assert "Moderate" in text
        assert "10" in text
        assert "sibling" in text.lower()

    def test_child_profile_empty(self):
        """Empty profile should produce default text."""
        text = child_profile_to_text({})
        assert text == "General foster care placement needed."

    def test_family_profile_with_all_fields(self):
        """Family profile with all fields should produce rich text."""
        profile = {
            "accepted_care_levels": ["Basic", "Moderate"],
            "safety_certifications": ["FirstAidCPR", "TraumaInformedCare"],
            "accepted_age_range": [5, 12],
            "capacity_max": 4,
            "capacity_current": 1,
            "family_strengths": "Experienced foster parents",
        }
        text = family_profile_to_text(profile)
        assert "Basic" in text
        assert "5-12" in text
        assert "1/4" in text

    def test_family_profile_empty(self):
        """Empty profile should produce default text."""
        text = family_profile_to_text({})
        assert text == "Licensed foster family."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
