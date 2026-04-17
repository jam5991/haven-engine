//! # Symbolic Constraint Validator
//!
//! Implements the formal predicate for foster care placement validation:
//!
//! ```text
//! ∀c ∈ Children, ∀f ∈ Families:
//!   Haven(c, f) ⟺ Legal(f) ∧ Capacity(f) ∧ SafetyMatch(c, f)
//! ```
//!
//! All checks are O(1) boolean predicates — no neural inference, no LLM calls.
//! The search space is pruned symbolically before any scoring occurs.

use chrono::{NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Domain Types
// ---------------------------------------------------------------------------

/// Represents a child in the foster care system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Child {
    pub id: Uuid,
    pub name_redacted: String,
    pub date_of_birth: NaiveDate,
    pub age: u8,
    pub trauma_flags: Vec<TraumaFlag>,
    pub required_care_level: CareLevel,
    pub sibling_group_id: Option<Uuid>,
    pub county: String,
    pub state: String,
}

/// Trauma-informed flags derived from intake assessments.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TraumaFlag {
    PhysicalAbuse,
    Neglect,
    SexualAbuse,
    SubstanceExposure,
    DomesticViolence,
    MedicalTrauma,
    AbandonmentLoss,
}

/// Required level of care for the child.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CareLevel {
    /// Standard foster care.
    Basic,
    /// Moderate — requires additional support or training.
    Moderate,
    /// Treatment-level — therapeutic foster care.
    Treatment,
    /// Intensive — specialized medical or behavioral needs.
    Intensive,
}

/// Represents a licensed foster family.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Family {
    pub id: Uuid,
    pub name_redacted: String,
    pub license_status: LicenseStatus,
    pub license_expiry: NaiveDate,
    pub capacity_current: u8,
    pub capacity_max: u8,
    pub safety_certifications: Vec<SafetyCertification>,
    pub accepted_age_range: (u8, u8),
    pub accepted_care_levels: Vec<CareLevel>,
    pub county: String,
    pub state: String,
    pub sibling_group_capacity: u8,
}

/// License status for a foster family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LicenseStatus {
    Active,
    Provisional,
    Suspended,
    Expired,
    Revoked,
}

/// Safety certifications held by a foster family.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SafetyCertification {
    TraumaInformedCare,
    MedicationAdministration,
    CrisisIntervention,
    SubstanceAbuseAwareness,
    TherapeuticFosterCare,
    FirstAidCPR,
    FireSafety,
}

// ---------------------------------------------------------------------------
// Constraint Rules
// ---------------------------------------------------------------------------

/// State-specific placement constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementConstraints {
    /// Whether siblings must be co-placed (federal ASFA mandate).
    pub require_sibling_co_placement: bool,
    /// Whether the family must be in the same county as the child.
    pub require_same_county: bool,
    /// Whether the family must be in the same state as the child.
    pub require_same_state: bool,
    /// Minimum required safety certifications for a given care level.
    pub care_level_cert_requirements: Vec<(CareLevel, Vec<SafetyCertification>)>,
    /// Grace period in days for provisionally-licensed families.
    pub provisional_grace_days: i64,
}

impl Default for PlacementConstraints {
    fn default() -> Self {
        Self {
            require_sibling_co_placement: true,
            require_same_county: false,
            require_same_state: true,
            care_level_cert_requirements: vec![
                (CareLevel::Basic, vec![SafetyCertification::FirstAidCPR, SafetyCertification::FireSafety]),
                (CareLevel::Moderate, vec![
                    SafetyCertification::FirstAidCPR,
                    SafetyCertification::FireSafety,
                    SafetyCertification::TraumaInformedCare,
                ]),
                (CareLevel::Treatment, vec![
                    SafetyCertification::FirstAidCPR,
                    SafetyCertification::FireSafety,
                    SafetyCertification::TraumaInformedCare,
                    SafetyCertification::TherapeuticFosterCare,
                    SafetyCertification::CrisisIntervention,
                ]),
                (CareLevel::Intensive, vec![
                    SafetyCertification::FirstAidCPR,
                    SafetyCertification::FireSafety,
                    SafetyCertification::TraumaInformedCare,
                    SafetyCertification::TherapeuticFosterCare,
                    SafetyCertification::CrisisIntervention,
                    SafetyCertification::MedicationAdministration,
                ]),
            ],
            provisional_grace_days: 90,
        }
    }
}

// ---------------------------------------------------------------------------
// Validation Result
// ---------------------------------------------------------------------------

/// Categories of constraint violations for forensic reporting.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationKind {
    /// Family license is not active (suspended, expired, or revoked).
    InvalidLicense { status: LicenseStatus },
    /// Family license has expired past the grace period.
    ExpiredLicense { expiry: NaiveDate, today: NaiveDate },
    /// Family is at or over capacity.
    CapacityExceeded { current: u8, max: u8 },
    /// Child's age is outside the family's accepted range.
    AgeOutOfRange { child_age: u8, accepted: (u8, u8) },
    /// Missing required safety certifications for the child's care level.
    MissingSafetyCerts { missing: Vec<SafetyCertification> },
    /// Family cannot accommodate the child's care level.
    CareLevelMismatch { required: CareLevel, accepted: Vec<CareLevel> },
    /// Sibling co-placement mandate violated (family lacks capacity for siblings).
    SiblingCoPlacementViolation { required_capacity: u8, available: u8 },
    /// Geographic constraint violation.
    GeographicMismatch { child_location: String, family_location: String },
}

/// Result of a symbolic validation check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationResult {
    /// All predicates satisfied — placement is legally permissible.
    Valid {
        child_id: Uuid,
        family_id: Uuid,
    },
    /// One or more predicates failed — placement is prohibited.
    Violation {
        child_id: Uuid,
        family_id: Uuid,
        violations: Vec<ViolationKind>,
    },
}

// ---------------------------------------------------------------------------
// Validation Logic
// ---------------------------------------------------------------------------

/// Validate a child-family placement against all symbolic constraints.
///
/// This function implements O(1) boolean predicate evaluation. All checks
/// are constant-time lookups and comparisons — no iteration over unbounded
/// data structures.
///
/// # Returns
///
/// - `ValidationResult::Valid` if ALL predicates are satisfied.
/// - `ValidationResult::Violation` with a complete list of ALL violations
///   (not just the first failure) for forensic reporting.
pub fn validate(
    child: &Child,
    family: &Family,
    constraints: &PlacementConstraints,
) -> ValidationResult {
    let mut violations = Vec::new();

    // ── Predicate 1: Legal(f) — License validity ──────────────────────
    check_license(family, constraints, &mut violations);

    // ── Predicate 2: Capacity(f) — Available slots ────────────────────
    check_capacity(family, &mut violations);

    // ── Predicate 3: SafetyMatch(c, f) — Age, care level, certs ──────
    check_age_range(child, family, &mut violations);
    check_care_level(child, family, &mut violations);
    check_safety_certifications(child, family, constraints, &mut violations);

    // ── Predicate 4: Geographic constraints ───────────────────────────
    check_geographic(child, family, constraints, &mut violations);

    // ── Predicate 5: Sibling co-placement ─────────────────────────────
    check_sibling_placement(child, family, constraints, &mut violations);

    if violations.is_empty() {
        ValidationResult::Valid {
            child_id: child.id,
            family_id: family.id,
        }
    } else {
        ValidationResult::Violation {
            child_id: child.id,
            family_id: family.id,
            violations,
        }
    }
}

/// Check that the family's license is active and not expired.
fn check_license(
    family: &Family,
    constraints: &PlacementConstraints,
    violations: &mut Vec<ViolationKind>,
) {
    match family.license_status {
        LicenseStatus::Active => {}
        LicenseStatus::Provisional => {
            // Provisional licenses are allowed within the grace period.
            let today = Utc::now().date_naive();
            let days_until_expiry = (family.license_expiry - today).num_days();
            if days_until_expiry < -constraints.provisional_grace_days {
                violations.push(ViolationKind::ExpiredLicense {
                    expiry: family.license_expiry,
                    today,
                });
            }
        }
        ref status @ (LicenseStatus::Suspended | LicenseStatus::Expired | LicenseStatus::Revoked) => {
            violations.push(ViolationKind::InvalidLicense {
                status: status.clone(),
            });

            // Also check expiry for expired licenses.
            if *status == LicenseStatus::Expired {
                let today = Utc::now().date_naive();
                violations.push(ViolationKind::ExpiredLicense {
                    expiry: family.license_expiry,
                    today,
                });
            }
        }
    }
}

/// Check that the family has available capacity.
fn check_capacity(family: &Family, violations: &mut Vec<ViolationKind>) {
    if family.capacity_current >= family.capacity_max {
        violations.push(ViolationKind::CapacityExceeded {
            current: family.capacity_current,
            max: family.capacity_max,
        });
    }
}

/// Check that the child's age falls within the family's accepted range.
fn check_age_range(child: &Child, family: &Family, violations: &mut Vec<ViolationKind>) {
    let (min_age, max_age) = family.accepted_age_range;
    if child.age < min_age || child.age > max_age {
        violations.push(ViolationKind::AgeOutOfRange {
            child_age: child.age,
            accepted: family.accepted_age_range,
        });
    }
}

/// Check that the family accepts the child's required care level.
fn check_care_level(child: &Child, family: &Family, violations: &mut Vec<ViolationKind>) {
    if !family.accepted_care_levels.contains(&child.required_care_level) {
        violations.push(ViolationKind::CareLevelMismatch {
            required: child.required_care_level.clone(),
            accepted: family.accepted_care_levels.clone(),
        });
    }
}

/// Check that the family holds all required safety certifications for the
/// child's care level.
fn check_safety_certifications(
    child: &Child,
    family: &Family,
    constraints: &PlacementConstraints,
    violations: &mut Vec<ViolationKind>,
) {
    if let Some((_, required_certs)) = constraints
        .care_level_cert_requirements
        .iter()
        .find(|(level, _)| *level == child.required_care_level)
    {
        let missing: Vec<SafetyCertification> = required_certs
            .iter()
            .filter(|cert| !family.safety_certifications.contains(cert))
            .cloned()
            .collect();

        if !missing.is_empty() {
            violations.push(ViolationKind::MissingSafetyCerts { missing });
        }
    }
}

/// Check geographic placement constraints.
fn check_geographic(
    child: &Child,
    family: &Family,
    constraints: &PlacementConstraints,
    violations: &mut Vec<ViolationKind>,
) {
    if constraints.require_same_state && child.state != family.state {
        violations.push(ViolationKind::GeographicMismatch {
            child_location: format!("{}, {}", child.county, child.state),
            family_location: format!("{}, {}", family.county, family.state),
        });
    }

    if constraints.require_same_county && child.county != family.county {
        violations.push(ViolationKind::GeographicMismatch {
            child_location: child.county.clone(),
            family_location: family.county.clone(),
        });
    }
}

/// Check sibling co-placement requirements.
fn check_sibling_placement(
    child: &Child,
    family: &Family,
    constraints: &PlacementConstraints,
    violations: &mut Vec<ViolationKind>,
) {
    if constraints.require_sibling_co_placement && child.sibling_group_id.is_some() {
        // A child with a sibling group requires sufficient capacity for at
        // least 2 children (the child + at minimum one sibling).
        let available = family.capacity_max.saturating_sub(family.capacity_current);
        if available < 2 && family.sibling_group_capacity < 2 {
            violations.push(ViolationKind::SiblingCoPlacementViolation {
                required_capacity: 2,
                available,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_child() -> Child {
        Child {
            id: Uuid::new_v4(),
            name_redacted: "CHILD_001".into(),
            date_of_birth: NaiveDate::from_ymd_opt(2018, 3, 15).unwrap(),
            age: 8,
            trauma_flags: vec![TraumaFlag::Neglect],
            required_care_level: CareLevel::Basic,
            sibling_group_id: None,
            county: "Los Angeles".into(),
            state: "CA".into(),
        }
    }

    fn make_family() -> Family {
        Family {
            id: Uuid::new_v4(),
            name_redacted: "FAMILY_001".into(),
            license_status: LicenseStatus::Active,
            license_expiry: NaiveDate::from_ymd_opt(2027, 12, 31).unwrap(),
            capacity_current: 1,
            capacity_max: 4,
            safety_certifications: vec![
                SafetyCertification::FirstAidCPR,
                SafetyCertification::FireSafety,
                SafetyCertification::TraumaInformedCare,
            ],
            accepted_age_range: (5, 12),
            accepted_care_levels: vec![CareLevel::Basic, CareLevel::Moderate],
            county: "Los Angeles".into(),
            state: "CA".into(),
            sibling_group_capacity: 3,
        }
    }

    #[test]
    fn test_valid_placement() {
        let child = make_child();
        let family = make_family();
        let constraints = PlacementConstraints::default();
        let result = validate(&child, &family, &constraints);
        assert!(matches!(result, ValidationResult::Valid { .. }));
    }

    #[test]
    fn test_expired_license_violation() {
        let child = make_child();
        let mut family = make_family();
        family.license_status = LicenseStatus::Expired;
        family.license_expiry = NaiveDate::from_ymd_opt(2020, 1, 1).unwrap();
        let constraints = PlacementConstraints::default();

        let result = validate(&child, &family, &constraints);
        match result {
            ValidationResult::Violation { violations, .. } => {
                assert!(violations.iter().any(|v| matches!(v, ViolationKind::InvalidLicense { .. })));
            }
            _ => panic!("Expected violation for expired license"),
        }
    }

    #[test]
    fn test_capacity_exceeded_violation() {
        let child = make_child();
        let mut family = make_family();
        family.capacity_current = 4;
        family.capacity_max = 4;
        let constraints = PlacementConstraints::default();

        let result = validate(&child, &family, &constraints);
        match result {
            ValidationResult::Violation { violations, .. } => {
                assert!(violations.iter().any(|v| matches!(v, ViolationKind::CapacityExceeded { .. })));
            }
            _ => panic!("Expected violation for capacity exceeded"),
        }
    }

    #[test]
    fn test_age_out_of_range_violation() {
        let mut child = make_child();
        child.age = 16;
        let family = make_family();
        let constraints = PlacementConstraints::default();

        let result = validate(&child, &family, &constraints);
        match result {
            ValidationResult::Violation { violations, .. } => {
                assert!(violations.iter().any(|v| matches!(v, ViolationKind::AgeOutOfRange { .. })));
            }
            _ => panic!("Expected violation for age out of range"),
        }
    }

    #[test]
    fn test_care_level_mismatch_violation() {
        let mut child = make_child();
        child.required_care_level = CareLevel::Intensive;
        let family = make_family();
        let constraints = PlacementConstraints::default();

        let result = validate(&child, &family, &constraints);
        match result {
            ValidationResult::Violation { violations, .. } => {
                assert!(violations.iter().any(|v| matches!(v, ViolationKind::CareLevelMismatch { .. })));
            }
            _ => panic!("Expected violation for care level mismatch"),
        }
    }

    #[test]
    fn test_geographic_mismatch_violation() {
        let child = make_child();
        let mut family = make_family();
        family.state = "NY".into();
        let constraints = PlacementConstraints::default();

        let result = validate(&child, &family, &constraints);
        match result {
            ValidationResult::Violation { violations, .. } => {
                assert!(violations.iter().any(|v| matches!(v, ViolationKind::GeographicMismatch { .. })));
            }
            _ => panic!("Expected violation for geographic mismatch"),
        }
    }

    #[test]
    fn test_sibling_co_placement_violation() {
        let mut child = make_child();
        child.sibling_group_id = Some(Uuid::new_v4());
        let mut family = make_family();
        family.capacity_current = 3;
        family.capacity_max = 4;
        family.sibling_group_capacity = 1;
        let constraints = PlacementConstraints::default();

        let result = validate(&child, &family, &constraints);
        match result {
            ValidationResult::Violation { violations, .. } => {
                assert!(violations.iter().any(|v| matches!(v, ViolationKind::SiblingCoPlacementViolation { .. })));
            }
            _ => panic!("Expected violation for sibling co-placement"),
        }
    }

    #[test]
    fn test_multiple_violations_reported() {
        let mut child = make_child();
        child.age = 20;
        child.required_care_level = CareLevel::Intensive;

        let mut family = make_family();
        family.license_status = LicenseStatus::Revoked;
        family.capacity_current = 4;
        family.capacity_max = 4;
        family.state = "TX".into();

        let constraints = PlacementConstraints::default();
        let result = validate(&child, &family, &constraints);

        match result {
            ValidationResult::Violation { violations, .. } => {
                // Should catch: InvalidLicense, CapacityExceeded, AgeOutOfRange,
                // CareLevelMismatch, MissingSafetyCerts, GeographicMismatch
                assert!(violations.len() >= 4, "Expected multiple violations, got {}", violations.len());
            }
            _ => panic!("Expected multiple violations"),
        }
    }
}
