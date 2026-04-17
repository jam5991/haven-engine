//! # Relational Topology
//!
//! An in-memory vector-graph hybrid for relationship-aware foster care placement.
//! Combines an adjacency-list graph with vector embeddings for link-state
//! consistency: edges are automatically invalidated when a family's license
//! expires or capacity fills.
//!
//! ## Link-State Consistency Guarantee
//!
//! Every edge in the graph is guarded by a validity predicate. When querying
//! candidates, stale edges (expired licenses, full capacity) are pruned in
//! O(degree) time, ensuring the graph never returns an invalid match.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;

use crate::core::symbolic_validator::{
    Child, Family, LicenseStatus,
};

// ---------------------------------------------------------------------------
// Node Types
// ---------------------------------------------------------------------------

/// Unique identifier for a node in the relational graph.
pub type NodeId = Uuid;

/// A node in the relational graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    /// A child node with embedded feature vector.
    ChildNode {
        id: NodeId,
        child: Child,
        embedding: Option<Vec<f64>>,
    },
    /// A family node with embedded feature vector.
    FamilyNode {
        id: NodeId,
        family: Family,
        embedding: Option<Vec<f64>>,
    },
    /// A geographic region node (county or district).
    RegionNode {
        id: NodeId,
        name: String,
        state: String,
    },
}

impl Node {
    /// Get the node's unique identifier.
    pub fn id(&self) -> NodeId {
        match self {
            Node::ChildNode { id, .. } => *id,
            Node::FamilyNode { id, .. } => *id,
            Node::RegionNode { id, .. } => *id,
        }
    }
}

// ---------------------------------------------------------------------------
// Edge Types
// ---------------------------------------------------------------------------

/// An edge in the relational graph with weight and validity predicate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub edge_type: EdgeType,
    pub weight: f64,
}

/// Categories of relationships between nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Compatibility edge between a child and a family.
    Compatibility,
    /// Sibling link between two children.
    SiblingLink,
    /// Geographic proximity edge.
    GeographicProximity,
    /// A family belongs to a region.
    RegionMembership,
    /// Historical placement success between a child profile type and family.
    HistoricalSuccess,
}

// ---------------------------------------------------------------------------
// Candidate Match
// ---------------------------------------------------------------------------

/// A candidate placement match returned from a graph query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateMatch {
    pub family_id: NodeId,
    pub family: Family,
    /// Graph-based compatibility score (edge weight * centrality).
    pub graph_score: f64,
    /// Cosine similarity of embeddings, if available.
    pub embedding_similarity: Option<f64>,
    /// Combined score for ranking.
    pub combined_score: f64,
    /// Path length in the graph from child to family.
    pub graph_distance: u32,
}

// ---------------------------------------------------------------------------
// Relational Graph
// ---------------------------------------------------------------------------

/// The vector-graph hybrid data structure.
///
/// Maintains an adjacency list for O(1) edge lookups and optional vector
/// embeddings for neural similarity scoring.
#[derive(Debug, Clone)]
pub struct RelationalGraph {
    nodes: HashMap<NodeId, Node>,
    adjacency: HashMap<NodeId, Vec<Edge>>,
    node_count: usize,
    edge_count: usize,
}

impl RelationalGraph {
    /// Create an empty relational graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            adjacency: HashMap::new(),
            node_count: 0,
            edge_count: 0,
        }
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: Node) {
        let id = node.id();
        if !self.nodes.contains_key(&id) {
            self.node_count += 1;
        }
        self.nodes.insert(id, node);
        self.adjacency.entry(id).or_insert_with(Vec::new);
    }

    /// Add a directed edge between two nodes.
    pub fn add_edge(&mut self, edge: Edge) {
        self.adjacency
            .entry(edge.from)
            .or_insert_with(Vec::new)
            .push(edge.clone());
        self.edge_count += 1;

        // Add reverse edge for undirected relationships.
        if edge.edge_type != EdgeType::RegionMembership {
            let reverse = Edge {
                from: edge.to,
                to: edge.from,
                edge_type: edge.edge_type,
                weight: edge.weight,
            };
            self.adjacency
                .entry(reverse.from)
                .or_insert_with(Vec::new)
                .push(reverse);
            self.edge_count += 1;
        }
    }

    /// Get a node by its ID.
    pub fn get_node(&self, id: &NodeId) -> Option<&Node> {
        self.nodes.get(id)
    }

    /// Return the total number of nodes.
    pub fn node_count(&self) -> usize {
        self.node_count
    }

    /// Return the total number of edges.
    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    /// Query candidate family matches for a child, respecting link-state
    /// consistency.
    ///
    /// This function:
    /// 1. Finds all family nodes reachable from the child (1-hop or 2-hop).
    /// 2. Prunes edges to families with invalid licenses or full capacity.
    /// 3. Computes a combined score from edge weights and embedding similarity.
    /// 4. Returns results sorted by combined score (descending).
    pub fn query_candidates(&self, child_id: &NodeId) -> Vec<CandidateMatch> {
        let child_node = match self.nodes.get(child_id) {
            Some(Node::ChildNode { .. }) => self.nodes.get(child_id),
            _ => return Vec::new(),
        };

        let child_embedding = match child_node {
            Some(Node::ChildNode { embedding, .. }) => embedding.clone(),
            _ => None,
        };

        let mut candidates = Vec::new();

        // Direct edges (1-hop: child → family).
        if let Some(edges) = self.adjacency.get(child_id) {
            for edge in edges {
                if edge.edge_type != EdgeType::Compatibility {
                    continue;
                }

                if let Some(Node::FamilyNode { family, embedding, .. }) = self.nodes.get(&edge.to) {
                    // Link-state consistency: prune invalid families.
                    if !self.is_family_valid(family) {
                        continue;
                    }

                    let emb_sim = compute_cosine_similarity(
                        child_embedding.as_deref(),
                        embedding.as_deref(),
                    );

                    let combined = compute_combined_score(edge.weight, emb_sim);

                    candidates.push(CandidateMatch {
                        family_id: family.id,
                        family: family.clone(),
                        graph_score: edge.weight,
                        embedding_similarity: emb_sim,
                        combined_score: combined,
                        graph_distance: 1,
                    });
                }
            }
        }

        // 2-hop edges (child → region → family) for geographic proximity.
        if let Some(edges) = self.adjacency.get(child_id) {
            for edge in edges {
                if edge.edge_type != EdgeType::GeographicProximity
                    && edge.edge_type != EdgeType::RegionMembership
                {
                    continue;
                }

                // Find families connected to the intermediate node.
                if let Some(second_edges) = self.adjacency.get(&edge.to) {
                    for second_edge in second_edges {
                        if let Some(Node::FamilyNode { family, embedding, .. }) =
                            self.nodes.get(&second_edge.to)
                        {
                            if !self.is_family_valid(family) {
                                continue;
                            }

                            // Don't duplicate 1-hop matches.
                            if candidates.iter().any(|c| c.family_id == family.id) {
                                continue;
                            }

                            let emb_sim = compute_cosine_similarity(
                                child_embedding.as_deref(),
                                embedding.as_deref(),
                            );

                            let combined = compute_combined_score(
                                edge.weight * second_edge.weight,
                                emb_sim,
                            );

                            candidates.push(CandidateMatch {
                                family_id: family.id,
                                family: family.clone(),
                                graph_score: edge.weight * second_edge.weight,
                                embedding_similarity: emb_sim,
                                combined_score: combined,
                                graph_distance: 2,
                            });
                        }
                    }
                }
            }
        }

        // Sort by combined score descending.
        candidates.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());
        candidates
    }

    /// Invalidate all edges to a family (e.g., license expired, capacity full).
    ///
    /// This implements the link-state consistency guarantee by removing
    /// the family from the live graph without deleting the node itself
    /// (preserving historical data).
    pub fn invalidate_family(&mut self, family_id: &NodeId) {
        // Remove all edges pointing TO this family.
        for edges in self.adjacency.values_mut() {
            edges.retain(|e| e.to != *family_id);
        }

        // Remove all edges FROM this family.
        if let Some(edges) = self.adjacency.get_mut(family_id) {
            self.edge_count -= edges.len();
            edges.clear();
        }
    }

    /// Check if a family is currently valid (active license + available capacity).
    fn is_family_valid(&self, family: &Family) -> bool {
        matches!(
            family.license_status,
            LicenseStatus::Active | LicenseStatus::Provisional
        ) && family.capacity_current < family.capacity_max
    }

    /// Bulk-load families from a slice, creating nodes and connecting them
    /// to a region node.
    pub fn load_families(&mut self, families: &[Family]) {
        for family in families {
            let family_node = Node::FamilyNode {
                id: family.id,
                family: family.clone(),
                embedding: None,
            };
            self.add_node(family_node);

            // Create or connect to a region node.
            let region_id = Uuid::new_v5(
                &Uuid::NAMESPACE_DNS,
                format!("{},{}", family.county, family.state).as_bytes(),
            );

            if !self.nodes.contains_key(&region_id) {
                self.add_node(Node::RegionNode {
                    id: region_id,
                    name: family.county.clone(),
                    state: family.state.clone(),
                });
            }

            self.add_edge(Edge {
                from: family.id,
                to: region_id,
                edge_type: EdgeType::RegionMembership,
                weight: 1.0,
            });
        }
    }

    /// Add a child to the graph and create compatibility edges to all
    /// families in the same region.
    pub fn add_child_with_edges(&mut self, child: &Child) {
        let child_node = Node::ChildNode {
            id: child.id,
            child: child.clone(),
            embedding: None,
        };
        self.add_node(child_node);

        // Create compatibility edges to families in the same state.
        let family_ids: Vec<(NodeId, f64)> = self
            .nodes
            .values()
            .filter_map(|node| {
                if let Node::FamilyNode { id, family, .. } = node {
                    if family.state == child.state && self.is_family_valid(family) {
                        // Base compatibility weight: higher for same county.
                        let weight = if family.county == child.county {
                            0.8
                        } else {
                            0.5
                        };
                        Some((*id, weight))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        for (family_id, weight) in family_ids {
            self.add_edge(Edge {
                from: child.id,
                to: family_id,
                edge_type: EdgeType::Compatibility,
                weight,
            });
        }
    }
}

impl Default for RelationalGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Scoring Helpers
// ---------------------------------------------------------------------------

/// Compute cosine similarity between two embedding vectors.
fn compute_cosine_similarity(a: Option<&[f64]>, b: Option<&[f64]>) -> Option<f64> {
    match (a, b) {
        (Some(va), Some(vb)) if va.len() == vb.len() && !va.is_empty() => {
            let dot: f64 = va.iter().zip(vb.iter()).map(|(x, y)| x * y).sum();
            let norm_a: f64 = va.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_b: f64 = vb.iter().map(|x| x * x).sum::<f64>().sqrt();

            if norm_a == 0.0 || norm_b == 0.0 {
                None
            } else {
                Some(dot / (norm_a * norm_b))
            }
        }
        _ => None,
    }
}

/// Compute a combined score from graph weight and embedding similarity.
///
/// Formula: `0.4 * graph_score + 0.6 * embedding_similarity`
/// Falls back to graph_score if no embedding similarity is available.
fn compute_combined_score(graph_score: f64, embedding_similarity: Option<f64>) -> f64 {
    match embedding_similarity {
        Some(sim) => 0.4 * graph_score + 0.6 * sim,
        None => graph_score,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;
    use crate::core::symbolic_validator::{CareLevel, SafetyCertification};

    fn make_family_in(county: &str, state: &str) -> Family {
        Family {
            id: Uuid::new_v4(),
            name_redacted: format!("FAMILY_{}", county),
            license_status: LicenseStatus::Active,
            license_expiry: NaiveDate::from_ymd_opt(2027, 12, 31).unwrap(),
            capacity_current: 1,
            capacity_max: 4,
            safety_certifications: vec![
                SafetyCertification::FirstAidCPR,
                SafetyCertification::FireSafety,
            ],
            accepted_age_range: (0, 18),
            accepted_care_levels: vec![CareLevel::Basic, CareLevel::Moderate],
            county: county.into(),
            state: state.into(),
            sibling_group_capacity: 2,
        }
    }

    fn make_child_in(county: &str, state: &str) -> Child {
        Child {
            id: Uuid::new_v4(),
            name_redacted: "CHILD_TEST".into(),
            date_of_birth: NaiveDate::from_ymd_opt(2016, 1, 1).unwrap(),
            age: 10,
            trauma_flags: vec![],
            required_care_level: CareLevel::Basic,
            sibling_group_id: None,
            county: county.into(),
            state: state.into(),
        }
    }

    #[test]
    fn test_graph_construction() {
        let mut graph = RelationalGraph::new();
        let families = vec![
            make_family_in("LA", "CA"),
            make_family_in("SF", "CA"),
            make_family_in("NYC", "NY"),
        ];
        graph.load_families(&families);

        // 3 families + region nodes (LA,CA + SF,CA + NYC,NY = 3 regions)
        assert_eq!(graph.node_count(), 6);
    }

    #[test]
    fn test_query_candidates_same_state() {
        let mut graph = RelationalGraph::new();
        let families = vec![
            make_family_in("LA", "CA"),
            make_family_in("SF", "CA"),
            make_family_in("NYC", "NY"),
        ];
        graph.load_families(&families);

        let child = make_child_in("LA", "CA");
        graph.add_child_with_edges(&child);

        let candidates = graph.query_candidates(&child.id);
        // Should find 2 CA families, not the NY family.
        assert_eq!(candidates.len(), 2);
        assert!(candidates.iter().all(|c| c.family.state == "CA"));
    }

    #[test]
    fn test_same_county_higher_weight() {
        let mut graph = RelationalGraph::new();
        let families = vec![
            make_family_in("LA", "CA"),
            make_family_in("SF", "CA"),
        ];
        graph.load_families(&families);

        let child = make_child_in("LA", "CA");
        graph.add_child_with_edges(&child);

        let candidates = graph.query_candidates(&child.id);
        // Same-county family should have higher score (0.8 vs 0.5).
        assert!(candidates[0].graph_score > candidates[1].graph_score);
        assert_eq!(candidates[0].family.county, "LA");
    }

    #[test]
    fn test_link_state_consistency_prunes_invalid() {
        let mut graph = RelationalGraph::new();
        let mut full_family = make_family_in("LA", "CA");
        full_family.capacity_current = 4;
        full_family.capacity_max = 4;

        let valid_family = make_family_in("LA", "CA");
        graph.load_families(&[full_family, valid_family]);

        let child = make_child_in("LA", "CA");
        graph.add_child_with_edges(&child);

        let candidates = graph.query_candidates(&child.id);
        // Should only find the valid family, not the full one.
        assert_eq!(candidates.len(), 1);
    }

    #[test]
    fn test_invalidate_family() {
        let mut graph = RelationalGraph::new();
        let family = make_family_in("LA", "CA");
        let family_id = family.id;
        graph.load_families(&[family]);

        let child = make_child_in("LA", "CA");
        graph.add_child_with_edges(&child);

        assert_eq!(graph.query_candidates(&child.id).len(), 1);

        graph.invalidate_family(&family_id);
        assert_eq!(graph.query_candidates(&child.id).len(), 0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = compute_cosine_similarity(Some(&a), Some(&b));
        assert!((sim.unwrap() - 1.0).abs() < 1e-9);

        let c = vec![0.0, 1.0, 0.0];
        let sim2 = compute_cosine_similarity(Some(&a), Some(&c));
        assert!((sim2.unwrap()).abs() < 1e-9);
    }
}
