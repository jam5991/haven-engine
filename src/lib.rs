//! # Haven Engine
//!
//! A neuro-symbolic foster care placement engine that guarantees 100% legal
//! compliance by decoupling symbolic constraint checking from neural preference
//! scoring.
//!
//! ## Architecture
//!
//! The engine enforces a hard symbolic predicate before neural scoring is calculated:
//!
//! ```text
//! ∀c ∈ Children, ∀f ∈ Families:
//!   Haven(c, f) ⟺ Legal(f) ∧ Capacity(f) ∧ SafetyMatch(c, f)
//! ```
//!
//! Only if `Haven(c, f) = 1` does the system compute the neural preference score.

pub mod core;
pub mod graph;

use crate::core::symbolic_validator::{Child, Family, PlacementConstraints, ValidationResult};

/// A fully ranked placement recommendation that has passed all symbolic checks.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RankedPlacement {
    /// The candidate family.
    pub family: Family,
    /// Neural preference score ∈ [0, 1] (cosine similarity in latent space).
    pub score: f64,
    /// The symbolic validation result (always `Valid` for ranked placements).
    pub validation: ValidationResult,
}

/// Find all legally valid placements for a child, ranked by preference score.
///
/// This is the primary public API. It:
/// 1. Prunes the candidate set using O(1) symbolic constraint checks.
/// 2. Returns only families that pass ALL predicates.
/// 3. The caller is responsible for neural scoring (Python pipeline) on the
///    validated subset.
pub fn find_valid_placements(
    child: &Child,
    families: &[Family],
    constraints: &PlacementConstraints,
) -> Vec<(Family, ValidationResult)> {
    families
        .iter()
        .filter_map(|family| {
            let result = crate::core::symbolic_validator::validate(child, family, constraints);
            match &result {
                ValidationResult::Valid { .. } => Some((family.clone(), result)),
                ValidationResult::Violation { .. } => None,
            }
        })
        .collect()
}

/// Find all placements and include violations for forensic reporting.
pub fn evaluate_all_placements(
    child: &Child,
    families: &[Family],
    constraints: &PlacementConstraints,
) -> Vec<(Family, ValidationResult)> {
    families
        .iter()
        .map(|family| {
            let result = crate::core::symbolic_validator::validate(child, family, constraints);
            (family.clone(), result)
        })
        .collect()
}
