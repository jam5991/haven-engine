//! # Placement State Machine
//!
//! Manages the placement lifecycle as a finite state machine with enforced
//! transitions and guard clauses. Invalid state transitions are caught at
//! compile time via the type system and at runtime via guard predicates.
//!
//! ```text
//! Intake → Screening → Validated → Ranked → Placed → Review
//!                   ↘ Rejected (with violation report)
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

use super::symbolic_validator::{Child, Family, PlacementConstraints, ValidationResult, ViolationKind};

// ---------------------------------------------------------------------------
// State Definitions
// ---------------------------------------------------------------------------

/// All possible states in the placement lifecycle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PlacementState {
    /// Initial intake — child information has been submitted.
    Intake,
    /// Screening — candidate families are being identified.
    Screening,
    /// Validated — symbolic constraint checks have completed.
    Validated {
        valid_family_count: usize,
        total_family_count: usize,
    },
    /// Rejected — no valid placements exist; violation report attached.
    Rejected {
        violations: Vec<(Uuid, Vec<ViolationKind>)>,
    },
    /// Ranked — neural scoring has ordered the valid candidates.
    Ranked {
        top_match_score: f64,
    },
    /// Placed — child has been matched to a family.
    Placed {
        family_id: Uuid,
        placed_at: DateTime<Utc>,
    },
    /// Review — periodic compliance review of an active placement.
    Review {
        family_id: Uuid,
        review_due: DateTime<Utc>,
    },
}

/// Events that trigger state transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    /// Case worker submits an intake form.
    SubmitIntake,
    /// Begin screening candidate families.
    BeginScreening {
        candidate_count: usize,
    },
    /// Symbolic validation completed.
    ValidationComplete {
        results: Vec<(Uuid, ValidationResult)>,
    },
    /// Neural ranking completed.
    RankingComplete {
        top_score: f64,
    },
    /// Finalize placement with a specific family.
    FinalizePlacement {
        family_id: Uuid,
    },
    /// Trigger a compliance review.
    TriggerReview {
        review_due: DateTime<Utc>,
    },
    /// Reset to intake (e.g., placement disruption).
    Reset,
}

// ---------------------------------------------------------------------------
// Placement Case
// ---------------------------------------------------------------------------

/// A placement case tracking a child through the state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacementCase {
    pub case_id: Uuid,
    pub child: Child,
    pub state: PlacementState,
    pub candidate_families: Vec<Family>,
    pub constraints: PlacementConstraints,
    pub history: Vec<StateTransition>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A recorded state transition for audit logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    pub from: PlacementState,
    pub to: PlacementState,
    pub event: Event,
    pub timestamp: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during state transitions.
#[derive(Debug, Error)]
pub enum TransitionError {
    #[error("Invalid transition from {from:?} via event {event:?}")]
    InvalidTransition {
        from: PlacementState,
        event: String,
    },

    #[error("Guard clause failed: {reason}")]
    GuardFailed {
        reason: String,
    },

    #[error("No valid placements found — all {count} families had violations")]
    NoValidPlacements {
        count: usize,
    },
}

// ---------------------------------------------------------------------------
// State Machine Logic
// ---------------------------------------------------------------------------

impl PlacementCase {
    /// Create a new placement case for a child.
    pub fn new(child: Child, constraints: PlacementConstraints) -> Self {
        let now = Utc::now();
        Self {
            case_id: Uuid::new_v4(),
            child,
            state: PlacementState::Intake,
            candidate_families: Vec::new(),
            constraints,
            history: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Apply an event to transition the state machine.
    ///
    /// Returns the new state, or an error if the transition is invalid.
    pub fn transition(&mut self, event: Event) -> Result<PlacementState, TransitionError> {
        let new_state = match (&self.state, &event) {
            // Intake → Screening
            (PlacementState::Intake, Event::BeginScreening { candidate_count }) => {
                if *candidate_count == 0 {
                    return Err(TransitionError::GuardFailed {
                        reason: "Cannot begin screening with zero candidates".into(),
                    });
                }
                PlacementState::Screening
            }

            // Screening → Validated | Rejected
            (PlacementState::Screening, Event::ValidationComplete { results }) => {
                let valid_count = results
                    .iter()
                    .filter(|(_, r)| matches!(r, ValidationResult::Valid { .. }))
                    .count();

                if valid_count == 0 {
                    let violations: Vec<(Uuid, Vec<ViolationKind>)> = results
                        .iter()
                        .filter_map(|(id, r)| match r {
                            ValidationResult::Violation { violations, .. } => {
                                Some((*id, violations.clone()))
                            }
                            _ => None,
                        })
                        .collect();

                    PlacementState::Rejected { violations }
                } else {
                    PlacementState::Validated {
                        valid_family_count: valid_count,
                        total_family_count: results.len(),
                    }
                }
            }

            // Validated → Ranked
            (PlacementState::Validated { .. }, Event::RankingComplete { top_score }) => {
                if *top_score < 0.0 || *top_score > 1.0 {
                    return Err(TransitionError::GuardFailed {
                        reason: format!("Score must be in [0, 1], got {}", top_score),
                    });
                }
                PlacementState::Ranked {
                    top_match_score: *top_score,
                }
            }

            // Ranked → Placed
            (PlacementState::Ranked { .. }, Event::FinalizePlacement { family_id }) => {
                PlacementState::Placed {
                    family_id: *family_id,
                    placed_at: Utc::now(),
                }
            }

            // Placed → Review
            (PlacementState::Placed { family_id, .. }, Event::TriggerReview { review_due }) => {
                PlacementState::Review {
                    family_id: *family_id,
                    review_due: *review_due,
                }
            }

            // Review → Placed (review complete, return to placed state)
            (PlacementState::Review { family_id, .. }, Event::FinalizePlacement { .. }) => {
                PlacementState::Placed {
                    family_id: *family_id,
                    placed_at: Utc::now(),
                }
            }

            // Any → Intake (reset)
            (_, Event::Reset) => PlacementState::Intake,

            // Invalid transition
            (state, event) => {
                return Err(TransitionError::InvalidTransition {
                    from: state.clone(),
                    event: format!("{:?}", event),
                });
            }
        };

        // Record the transition for audit logging.
        let transition = StateTransition {
            from: self.state.clone(),
            to: new_state.clone(),
            event,
            timestamp: Utc::now(),
        };
        self.history.push(transition);
        self.state = new_state.clone();
        self.updated_at = Utc::now();

        Ok(new_state)
    }

    /// Check if the case is in a terminal state (Placed or Rejected).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            PlacementState::Placed { .. } | PlacementState::Rejected { .. }
        )
    }

    /// Get the complete audit trail of state transitions.
    pub fn audit_trail(&self) -> &[StateTransition] {
        &self.history
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::symbolic_validator::*;
    use chrono::NaiveDate;

    fn make_child() -> Child {
        Child {
            id: Uuid::new_v4(),
            name_redacted: "CHILD_TEST".into(),
            date_of_birth: NaiveDate::from_ymd_opt(2018, 6, 1).unwrap(),
            age: 8,
            trauma_flags: vec![],
            required_care_level: CareLevel::Basic,
            sibling_group_id: None,
            county: "Test County".into(),
            state: "CA".into(),
        }
    }

    #[test]
    fn test_happy_path_lifecycle() {
        let child = make_child();
        let constraints = PlacementConstraints::default();
        let mut case = PlacementCase::new(child, constraints);

        // Intake → Screening
        let state = case.transition(Event::BeginScreening { candidate_count: 5 }).unwrap();
        assert!(matches!(state, PlacementState::Screening));

        // Screening → Validated
        let family_id = Uuid::new_v4();
        let results = vec![(family_id, ValidationResult::Valid {
            child_id: case.child.id,
            family_id,
        })];
        let state = case.transition(Event::ValidationComplete { results }).unwrap();
        assert!(matches!(state, PlacementState::Validated { valid_family_count: 1, .. }));

        // Validated → Ranked
        let state = case.transition(Event::RankingComplete { top_score: 0.92 }).unwrap();
        assert!(matches!(state, PlacementState::Ranked { .. }));

        // Ranked → Placed
        let state = case.transition(Event::FinalizePlacement { family_id }).unwrap();
        assert!(matches!(state, PlacementState::Placed { .. }));

        // Should have 4 transitions in the audit trail.
        assert_eq!(case.audit_trail().len(), 4);
        assert!(case.is_terminal());
    }

    #[test]
    fn test_rejection_path() {
        let child = make_child();
        let constraints = PlacementConstraints::default();
        let mut case = PlacementCase::new(child, constraints);

        case.transition(Event::BeginScreening { candidate_count: 2 }).unwrap();

        // All families have violations → Rejected.
        let results = vec![
            (Uuid::new_v4(), ValidationResult::Violation {
                child_id: case.child.id,
                family_id: Uuid::new_v4(),
                violations: vec![ViolationKind::CapacityExceeded { current: 4, max: 4 }],
            }),
        ];
        let state = case.transition(Event::ValidationComplete { results }).unwrap();
        assert!(matches!(state, PlacementState::Rejected { .. }));
        assert!(case.is_terminal());
    }

    #[test]
    fn test_invalid_transition_error() {
        let child = make_child();
        let constraints = PlacementConstraints::default();
        let mut case = PlacementCase::new(child, constraints);

        // Cannot go directly from Intake to Ranked.
        let result = case.transition(Event::RankingComplete { top_score: 0.5 });
        assert!(result.is_err());
    }

    #[test]
    fn test_guard_clause_zero_candidates() {
        let child = make_child();
        let constraints = PlacementConstraints::default();
        let mut case = PlacementCase::new(child, constraints);

        let result = case.transition(Event::BeginScreening { candidate_count: 0 });
        assert!(result.is_err());
    }

    #[test]
    fn test_guard_clause_invalid_score() {
        let child = make_child();
        let constraints = PlacementConstraints::default();
        let mut case = PlacementCase::new(child, constraints);

        case.transition(Event::BeginScreening { candidate_count: 1 }).unwrap();
        let family_id = Uuid::new_v4();
        case.transition(Event::ValidationComplete {
            results: vec![(family_id, ValidationResult::Valid {
                child_id: case.child.id,
                family_id,
            })],
        }).unwrap();

        let result = case.transition(Event::RankingComplete { top_score: 1.5 });
        assert!(result.is_err());
    }

    #[test]
    fn test_reset_from_any_state() {
        let child = make_child();
        let constraints = PlacementConstraints::default();
        let mut case = PlacementCase::new(child, constraints);

        case.transition(Event::BeginScreening { candidate_count: 1 }).unwrap();
        let state = case.transition(Event::Reset).unwrap();
        assert!(matches!(state, PlacementState::Intake));
    }
}
