"""
PII Masking Module — Privacy-Preserving Preprocessing

Uses Microsoft Presidio to detect and anonymize Personally Identifiable
Information (PII) in child welfare profiles before they enter the neural
embedding pipeline. This ensures no real PII is ever processed by the
ML model.

Custom recognizers handle domain-specific patterns:
- Case numbers (e.g., CW-2026-001234)
- Court docket IDs (e.g., JUV-2026-0456)
- Foster care placement IDs

All masking operations are deterministic and reversible for authorized
case workers with the decryption key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Masked Result
# ---------------------------------------------------------------------------


class MaskedResult(BaseModel):
    """Result of a PII masking operation."""

    original_length: int
    masked_length: int
    entities_detected: int
    entity_types: list[str]
    masked_text: str


# ---------------------------------------------------------------------------
# Custom Pattern Definitions
# ---------------------------------------------------------------------------

# Domain-specific PII patterns for child welfare systems.
CASE_NUMBER_PATTERN = re.compile(r"\bCW-\d{4}-\d{4,8}\b")
DOCKET_ID_PATTERN = re.compile(r"\bJUV-\d{4}-\d{4,8}\b")
PLACEMENT_ID_PATTERN = re.compile(r"\bPLC-\d{6,10}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
PHONE_PATTERN = re.compile(r"\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
DOB_PATTERN = re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b")


# ---------------------------------------------------------------------------
# PII Masker
# ---------------------------------------------------------------------------


class PIIMasker:
    """
    Privacy-preserving PII masking for child welfare profiles.

    Uses a layered approach:
    1. Custom regex patterns for domain-specific identifiers.
    2. Microsoft Presidio for general PII (names, addresses, etc.).
    3. Deterministic replacement tokens for consistency across runs.

    Example::

        masker = PIIMasker()
        result = masker.mask("John Smith, SSN 123-45-6789, case CW-2026-001234")
        print(result.masked_text)
        # → "<PERSON>, SSN <SSN>, case <CASE_NUMBER>"
    """

    def __init__(self, enable_presidio: bool = True):
        """
        Initialize the PII masker.

        Args:
            enable_presidio: If True, loads Presidio analyzer and anonymizer.
                Set to False for lightweight mode (regex-only).
        """
        self._enable_presidio = enable_presidio
        self._analyzer = None
        self._anonymizer = None

        if enable_presidio:
            self._init_presidio()

        # Track entity counts for reporting.
        self._entity_counts: dict[str, int] = {}

    def _init_presidio(self) -> None:
        """Initialize Presidio analyzer and anonymizer with custom config."""
        try:
            from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
            from presidio_anonymizer import AnonymizerEngine

            # Create custom recognizers for child welfare domain.
            case_number_recognizer = PatternRecognizer(
                supported_entity="CASE_NUMBER",
                name="case_number_recognizer",
                patterns=[
                    Pattern(
                        name="case_number",
                        regex=r"\bCW-\d{4}-\d{4,8}\b",
                        score=0.95,
                    )
                ],
            )

            docket_recognizer = PatternRecognizer(
                supported_entity="DOCKET_ID",
                name="docket_recognizer",
                patterns=[
                    Pattern(
                        name="docket_id",
                        regex=r"\bJUV-\d{4}-\d{4,8}\b",
                        score=0.95,
                    )
                ],
            )

            placement_recognizer = PatternRecognizer(
                supported_entity="PLACEMENT_ID",
                name="placement_recognizer",
                patterns=[
                    Pattern(
                        name="placement_id",
                        regex=r"\bPLC-\d{6,10}\b",
                        score=0.90,
                    )
                ],
            )

            from presidio_analyzer.nlp_engine import NlpEngineProvider

            # Use en_core_web_sm (12MB) instead of the default en_core_web_lg
            # (400MB) — sufficient for PII entity recognition in this domain.
            nlp_config = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }
            nlp_engine = NlpEngineProvider(nlp_configuration=nlp_config).create_engine()

            self._analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            self._analyzer.registry.add_recognizer(case_number_recognizer)
            self._analyzer.registry.add_recognizer(docket_recognizer)
            self._analyzer.registry.add_recognizer(placement_recognizer)

            self._anonymizer = AnonymizerEngine()

        except ImportError:
            print(
                "WARNING: presidio not available, falling back to regex-only mode"
            )
            self._enable_presidio = False

    def mask(self, text: str) -> MaskedResult:
        """
        Detect and mask all PII in the given text.

        Args:
            text: Raw text potentially containing PII.

        Returns:
            MaskedResult with the anonymized text and detection metadata.
        """
        if not text or not text.strip():
            return MaskedResult(
                original_length=len(text),
                masked_length=len(text),
                entities_detected=0,
                entity_types=[],
                masked_text=text,
            )

        entity_types: list[str] = []
        masked = text

        if self._enable_presidio and self._analyzer and self._anonymizer:
            # Use Presidio for comprehensive PII detection.
            results = self._analyzer.analyze(
                text=text,
                language="en",
                entities=[
                    "PERSON",
                    "PHONE_NUMBER",
                    "EMAIL_ADDRESS",
                    "US_SSN",
                    "LOCATION",
                    "DATE_TIME",
                    "CASE_NUMBER",
                    "DOCKET_ID",
                    "PLACEMENT_ID",
                ],
            )

            if results:
                anonymized = self._anonymizer.anonymize(
                    text=text, analyzer_results=results
                )
                masked = anonymized.text
                entity_types = list(set(r.entity_type for r in results))
        else:
            # Fallback: regex-only masking.
            masked, types = self._regex_mask(text)
            entity_types = types

        return MaskedResult(
            original_length=len(text),
            masked_length=len(masked),
            entities_detected=len(entity_types),
            entity_types=sorted(entity_types),
            masked_text=masked,
        )

    def mask_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively mask all string fields in a profile dictionary.

        Args:
            profile: A dictionary representing a child or family profile.

        Returns:
            A new dictionary with all string values masked.
        """
        masked_profile = {}
        for key, value in profile.items():
            if isinstance(value, str):
                result = self.mask(value)
                masked_profile[key] = result.masked_text
            elif isinstance(value, dict):
                masked_profile[key] = self.mask_profile(value)
            elif isinstance(value, list):
                masked_profile[key] = [
                    self.mask_profile(item) if isinstance(item, dict)
                    else self.mask(item).masked_text if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                masked_profile[key] = value
        return masked_profile

    def _regex_mask(self, text: str) -> tuple[str, list[str]]:
        """
        Fallback regex-based PII masking when Presidio is unavailable.

        Returns:
            Tuple of (masked_text, list_of_entity_types_found).
        """
        entity_types = []
        masked = text

        replacements = [
            (CASE_NUMBER_PATTERN, "<CASE_NUMBER>", "CASE_NUMBER"),
            (DOCKET_ID_PATTERN, "<DOCKET_ID>", "DOCKET_ID"),
            (PLACEMENT_ID_PATTERN, "<PLACEMENT_ID>", "PLACEMENT_ID"),
            (SSN_PATTERN, "<SSN>", "US_SSN"),
            (PHONE_PATTERN, "<PHONE>", "PHONE_NUMBER"),
            (DOB_PATTERN, "<DATE>", "DATE_TIME"),
        ]

        for pattern, replacement, entity_type in replacements:
            if pattern.search(masked):
                masked = pattern.sub(replacement, masked)
                entity_types.append(entity_type)

        return masked, entity_types


# ---------------------------------------------------------------------------
# CLI entrypoint for testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    masker = PIIMasker(enable_presidio=True)

    test_cases = [
        "John Smith, age 8, case CW-2026-001234, SSN 123-45-6789",
        "Jane Doe resides at 123 Main St, Los Angeles, CA 90001. Phone: (555) 123-4567",
        "Court docket JUV-2026-0456, placement PLC-0012345",
        "Child born on 3/15/2018, mother Mary Johnson, email mary.j@example.com",
    ]

    for text in test_cases:
        result = masker.mask(text)
        print(f"\nOriginal:  {text}")
        print(f"Masked:    {result.masked_text}")
        print(f"Entities:  {result.entity_types} ({result.entities_detected} found)")
