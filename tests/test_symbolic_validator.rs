//! Integration tests for the haven-engine symbolic validation pipeline.
//!
//! These tests verify end-to-end behavior of the constraint validator,
//! state machine, and graph layer working together.

use chrono::NaiveDate;
use uuid::Uuid;

use haven_engine::core::symbolic_validator::*;
use haven_engine::core::state_machine::*;
use haven_engine::graph::relational_topology::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn make_test_child(age: u8, care_level: CareLevel, county: &str, state: &str) -> Child {
    Child {
        id: Uuid::new_v4(),
        name_redacted: "TEST_CHILD".into(),
        date_of_birth: NaiveDate::from_ymd_opt(2018, 1, 1).unwrap(),
        age,
        trauma_flags: vec![TraumaFlag::Neglect, TraumaFlag::AbandonmentLoss],
        required_care_level: care_level,
        sibling_group_id: None,
        county: county.into(),
        state: state.into(),
    }
}

fn make_test_family(
    county: &str,
    state: &str,
    care_levels: Vec<CareLevel>,
    age_range: (u8, u8),
    capacity: (u8, u8),
) -> Family {
    Family {
        id: Uuid::new_v4(),
        name_redacted: format!("TEST_FAMILY_{}", county),
        license_status: LicenseStatus::Active,
        license_expiry: NaiveDate::from_ymd_opt(2027, 12, 31).unwrap(),
        capacity_current: capacity.0,
        capacity_max: capacity.1,
        safety_certifications: vec![
            SafetyCertification::FirstAidCPR,
            SafetyCertification::FireSafety,
            SafetyCertification::TraumaInformedCare,
            SafetyCertification::CrisisIntervention,
            SafetyCertification::TherapeuticFosterCare,
            SafetyCertification::MedicationAdministration,
        ],
        accepted_age_range: age_range,
        accepted_care_levels: care_levels,
        county: county.into(),
        state: state.into(),
        sibling_group_capacity: 3,
    }
}

// ---------------------------------------------------------------------------
// End-to-End: Validator + State Machine
// ---------------------------------------------------------------------------

#[test]
fn test_full_placement_pipeline() {
    let child = make_test_child(8, CareLevel::Moderate, "LA", "CA");
    let constraints = PlacementConstraints::default();

    let families = vec![
        make_test_family("LA", "CA", vec![CareLevel::Basic, CareLevel::Moderate], (5, 12), (1, 4)),
        make_test_family("SF", "CA", vec![CareLevel::Basic], (0, 5), (2, 2)), // age mismatch + full
        make_test_family("LA", "CA", vec![CareLevel::Moderate, CareLevel::Treatment], (6, 16), (0, 3)),
    ];

    // Step 1: Create case and begin screening.
    let mut case = PlacementCase::new(child.clone(), constraints.clone());
    case.candidate_families = families.clone();
    case.transition(Event::BeginScreening {
        candidate_count: families.len(),
    })
    .unwrap();

    // Step 2: Validate all families.
    let results: Vec<(Uuid, ValidationResult)> = families
        .iter()
        .map(|f| (f.id, validate(&child, f, &constraints)))
        .collect();

    let valid_count = results
        .iter()
        .filter(|(_, r)| matches!(r, ValidationResult::Valid { .. }))
        .count();

    // Family 0: Valid (Moderate, LA/CA, age 5-12, capacity 1/4)
    // Family 1: Invalid (Basic only, age 0-5, full capacity)
    // Family 2: Valid (Moderate, LA/CA, age 6-16, capacity 0/3)
    assert_eq!(valid_count, 2, "Expected 2 valid families");

    // Step 3: Transition to Validated state.
    let state = case
        .transition(Event::ValidationComplete { results })
        .unwrap();
    assert!(matches!(
        state,
        PlacementState::Validated {
            valid_family_count: 2,
            total_family_count: 3,
        }
    ));

    // Step 4: Neural ranking (simulated).
    let state = case
        .transition(Event::RankingComplete { top_score: 0.87 })
        .unwrap();
    assert!(matches!(state, PlacementState::Ranked { .. }));

    // Step 5: Finalize placement.
    let placed_family_id = families[0].id;
    let state = case
        .transition(Event::FinalizePlacement {
            family_id: placed_family_id,
        })
        .unwrap();
    assert!(matches!(state, PlacementState::Placed { .. }));

    // Verify audit trail.
    assert_eq!(case.audit_trail().len(), 4);
}

// ---------------------------------------------------------------------------
// End-to-End: No Valid Placements → Rejection
// ---------------------------------------------------------------------------

#[test]
fn test_rejection_when_no_valid_families() {
    let child = make_test_child(16, CareLevel::Intensive, "LA", "CA");
    let constraints = PlacementConstraints::default();

    // All families will fail: age range too narrow, care level mismatch.
    let families = vec![
        make_test_family("LA", "CA", vec![CareLevel::Basic], (0, 10), (1, 2)),
        make_test_family("SF", "CA", vec![CareLevel::Basic], (0, 8), (3, 3)),
    ];

    let mut case = PlacementCase::new(child.clone(), constraints.clone());
    case.transition(Event::BeginScreening {
        candidate_count: families.len(),
    })
    .unwrap();

    let results: Vec<(Uuid, ValidationResult)> = families
        .iter()
        .map(|f| (f.id, validate(&child, f, &constraints)))
        .collect();

    let state = case
        .transition(Event::ValidationComplete { results })
        .unwrap();
    assert!(matches!(state, PlacementState::Rejected { .. }));
    assert!(case.is_terminal());
}

// ---------------------------------------------------------------------------
// End-to-End: Graph + Validator Integration
// ---------------------------------------------------------------------------

#[test]
fn test_graph_validates_candidates() {
    let child = make_test_child(10, CareLevel::Basic, "LA", "CA");
    let constraints = PlacementConstraints::default();

    let families = vec![
        make_test_family("LA", "CA", vec![CareLevel::Basic], (5, 15), (1, 4)),
        make_test_family("SF", "CA", vec![CareLevel::Basic], (5, 15), (0, 3)),
        make_test_family("NYC", "NY", vec![CareLevel::Basic], (5, 15), (0, 3)),
    ];

    // Build graph.
    let mut graph = RelationalGraph::new();
    graph.load_families(&families);
    graph.add_child_with_edges(&child);

    // Query candidates from graph (only same-state families).
    let candidates = graph.query_candidates(&child.id);
    assert_eq!(candidates.len(), 2, "Graph should find 2 CA families");

    // Validate each graph candidate symbolically.
    for candidate in &candidates {
        let result = validate(&child, &candidate.family, &constraints);
        assert!(
            matches!(result, ValidationResult::Valid { .. }),
            "Graph candidate {:?} should be valid",
            candidate.family.name_redacted
        );
    }
}

// ---------------------------------------------------------------------------
// Compliance: 100% of returned results must be valid
// ---------------------------------------------------------------------------

#[test]
fn test_compliance_guarantee() {
    let child = make_test_child(8, CareLevel::Moderate, "LA", "CA");
    let constraints = PlacementConstraints::default();

    // Generate a mix of valid and invalid families.
    let families = vec![
        make_test_family("LA", "CA", vec![CareLevel::Moderate], (5, 12), (1, 4)),       // Valid
        make_test_family("LA", "CA", vec![CareLevel::Basic], (0, 5), (2, 2)),            // Invalid x3
        make_test_family("LA", "TX", vec![CareLevel::Moderate], (5, 12), (0, 3)),        // Wrong state
        make_test_family("LA", "CA", vec![CareLevel::Moderate, CareLevel::Treatment], (6, 14), (0, 5)), // Valid
    ];

    // Use the public API.
    let valid_placements = haven_engine::find_valid_placements(&child, &families, &constraints);

    // Every result MUST be valid — 100% compliance guarantee.
    for (_, result) in &valid_placements {
        assert!(
            matches!(result, ValidationResult::Valid { .. }),
            "find_valid_placements returned a violation — compliance breach!"
        );
    }

    // Should have exactly 2 valid families.
    assert_eq!(valid_placements.len(), 2);
}

// ---------------------------------------------------------------------------
// Forensic Reporting: All violations must be collected
// ---------------------------------------------------------------------------

#[test]
fn test_forensic_violation_reporting() {
    let child = make_test_child(8, CareLevel::Moderate, "LA", "CA");
    let constraints = PlacementConstraints::default();

    let families = vec![
        make_test_family("LA", "CA", vec![CareLevel::Basic], (0, 5), (3, 3)),
    ];

    let all_results = haven_engine::evaluate_all_placements(&child, &families, &constraints);

    for (_, result) in &all_results {
        if let ValidationResult::Violation { violations, .. } = result {
            // Should report: AgeOutOfRange, CapacityExceeded, CareLevelMismatch
            assert!(
                violations.len() >= 3,
                "Expected at least 3 violations for forensic report, got {}",
                violations.len()
            );

            let has_age = violations.iter().any(|v| matches!(v, ViolationKind::AgeOutOfRange { .. }));
            let has_capacity = violations.iter().any(|v| matches!(v, ViolationKind::CapacityExceeded { .. }));
            let has_care = violations.iter().any(|v| matches!(v, ViolationKind::CareLevelMismatch { .. }));

            assert!(has_age, "Missing AgeOutOfRange violation");
            assert!(has_capacity, "Missing CapacityExceeded violation");
            assert!(has_care, "Missing CareLevelMismatch violation");
        }
    }
}
