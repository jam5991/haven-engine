# haven-engine
A purely neural approach lacks logical rigor, while a purely symbolic approach cannot parse unstructured case-worker notes. Haven Engine uses a Neuro-Symbolic pipeline: an edge-deployed LLM extracts semantic features from unstructured data, passing them to a deterministic Rust-based Graph Engine that enforces strict logical constraints.
