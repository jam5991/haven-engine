//! Benchmark: Throughput vs. Compliance
//!
//! Compares two placement strategies:
//! 1. **Symbolic-First (haven-engine):** Prune via constraints, then rank.
//!    Guarantees 100% compliance. Measures throughput.
//! 2. **Naive (simulated):** Rank all candidates, then filter violations.
//!    Measures compliance degradation at scale.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use chrono::NaiveDate;
use rand::Rng;
use uuid::Uuid;

use haven_engine::core::symbolic_validator::*;

// ---------------------------------------------------------------------------
// Data Generators
// ---------------------------------------------------------------------------

fn generate_child() -> Child {
    Child {
        id: Uuid::new_v4(),
        name_redacted: "BENCH_CHILD".into(),
        date_of_birth: NaiveDate::from_ymd_opt(2016, 6, 15).unwrap(),
        age: 10,
        trauma_flags: vec![TraumaFlag::Neglect, TraumaFlag::AbandonmentLoss],
        required_care_level: CareLevel::Moderate,
        sibling_group_id: None,
        county: "Los Angeles".into(),
        state: "CA".into(),
    }
}

fn generate_families(n: usize) -> Vec<Family> {
    let mut rng = rand::thread_rng();
    let states = ["CA", "CA", "CA", "TX", "NY", "FL", "WA", "OR"];
    let counties = ["Los Angeles", "San Francisco", "San Diego", "Sacramento", "Fresno"];

    (0..n)
        .map(|_| {
            let license_status = if rng.gen_bool(0.85) {
                LicenseStatus::Active
            } else if rng.gen_bool(0.5) {
                LicenseStatus::Provisional
            } else {
                LicenseStatus::Expired
            };

            let capacity_max: u8 = rng.gen_range(2..=6);
            let capacity_current: u8 = rng.gen_range(0..=capacity_max);

            let min_age: u8 = rng.gen_range(0..=10);
            let max_age: u8 = rng.gen_range(min_age..=18);

            let mut certs = vec![SafetyCertification::FirstAidCPR, SafetyCertification::FireSafety];
            if rng.gen_bool(0.6) {
                certs.push(SafetyCertification::TraumaInformedCare);
            }
            if rng.gen_bool(0.3) {
                certs.push(SafetyCertification::CrisisIntervention);
            }

            let care_levels = if rng.gen_bool(0.7) {
                vec![CareLevel::Basic, CareLevel::Moderate]
            } else {
                vec![CareLevel::Basic]
            };

            Family {
                id: Uuid::new_v4(),
                name_redacted: format!("BENCH_FAMILY_{}", rng.gen::<u16>()),
                license_status,
                license_expiry: NaiveDate::from_ymd_opt(2027, 12, 31).unwrap(),
                capacity_current,
                capacity_max,
                safety_certifications: certs,
                accepted_age_range: (min_age, max_age),
                accepted_care_levels: care_levels,
                county: counties[rng.gen_range(0..counties.len())].into(),
                state: states[rng.gen_range(0..states.len())].into(),
                sibling_group_capacity: rng.gen_range(1..=4),
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmark: Symbolic-First (haven-engine)
// ---------------------------------------------------------------------------

fn bench_symbolic_first(c: &mut Criterion) {
    let child = generate_child();
    let constraints = PlacementConstraints::default();

    let mut group = c.benchmark_group("symbolic_first_throughput");

    for size in [100, 1_000, 10_000] {
        let families = generate_families(size);

        group.bench_with_input(
            BenchmarkId::new("haven_engine", size),
            &families,
            |b, families| {
                b.iter(|| {
                    let valid: Vec<_> = families
                        .iter()
                        .filter(|f| {
                            matches!(
                                validate(black_box(&child), black_box(f), black_box(&constraints)),
                                ValidationResult::Valid { .. }
                            )
                        })
                        .collect();
                    black_box(valid)
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Naive Approach (rank all → filter)
// ---------------------------------------------------------------------------

/// Simulates a naive approach that scores ALL candidates before filtering.
fn naive_score_and_filter(child: &Child, families: &[Family], constraints: &PlacementConstraints) -> Vec<(&Family, f64)> {
    let mut scored: Vec<_> = families
        .iter()
        .map(|f| {
            // Simulate an expensive neural scoring step.
            let score = simulate_neural_score(child, f);
            (f, score)
        })
        .collect();

    // Sort by score (descending).
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // NOW filter violations (wasted compute on invalid families).
    scored
        .into_iter()
        .filter(|(f, _)| matches!(validate(child, f, constraints), ValidationResult::Valid { .. }))
        .collect()
}

/// Simulates a neural scoring step with some compute overhead.
fn simulate_neural_score(child: &Child, family: &Family) -> f64 {
    // Simulate latent space dot product (cheap approximation for benchmarking).
    let age_match = 1.0 - ((child.age as f64 - (family.accepted_age_range.0 as f64 + family.accepted_age_range.1 as f64) / 2.0).abs() / 18.0);
    let capacity_score = 1.0 - (family.capacity_current as f64 / family.capacity_max as f64);
    (age_match + capacity_score) / 2.0
}

fn bench_naive_approach(c: &mut Criterion) {
    let child = generate_child();
    let constraints = PlacementConstraints::default();

    let mut group = c.benchmark_group("naive_rank_then_filter");

    for size in [100, 1_000, 10_000] {
        let families = generate_families(size);

        group.bench_with_input(
            BenchmarkId::new("naive", size),
            &families,
            |b, families| {
                b.iter(|| {
                    black_box(naive_score_and_filter(
                        black_box(&child),
                        black_box(families),
                        black_box(&constraints),
                    ))
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Compliance Measurement
// ---------------------------------------------------------------------------

fn bench_compliance_rate(c: &mut Criterion) {
    let child = generate_child();
    let constraints = PlacementConstraints::default();

    let mut group = c.benchmark_group("compliance_measurement");

    for size in [100, 1_000, 10_000] {
        let families = generate_families(size);

        group.bench_with_input(
            BenchmarkId::new("compliance", size),
            &families,
            |b, families| {
                b.iter(|| {
                    let total = families.len();
                    let valid = families
                        .iter()
                        .filter(|f| {
                            matches!(
                                validate(black_box(&child), black_box(f), black_box(&constraints)),
                                ValidationResult::Valid { .. }
                            )
                        })
                        .count();

                    let compliance_rate = valid as f64 / total as f64;
                    black_box((valid, total, compliance_rate))
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_symbolic_first,
    bench_naive_approach,
    bench_compliance_rate
);
criterion_main!(benches);
