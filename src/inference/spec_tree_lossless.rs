//! Cross-drafter losslessness + cost tests for the tree-speculation seam
//! (Lane A). Pure CPU, GPU-free.
//!
//! These prove the two properties the whole lane rests on:
//!   1. On a degenerate single-branch (linear) tree, [`accept_longest_path`]
//!      emits EXACTLY what today's linear accept loop emits.
//!   2. With a mocked oracle (a deterministic stand-in for the target model's
//!      greedy argmax), no drafter — suffix decoding, token recycling, or a
//!      merge of them — can ever cause a non-greedy emission: every emitted
//!      token equals the oracle's argmax along the accepted path, i.e. exactly
//!      the token stream pure greedy decode produces. The drafter only changes
//!      how many tokens land per verify, never which tokens.
//!
//! Plus a microbench asserting each drafter's per-token draft cost is well
//! under ~5µs (the GPU verify, not drafting, must dominate).
//!
//! [`accept_longest_path`]: crate::inference::spec_tree::TokenTree::accept_longest_path

#![cfg(test)]

use std::time::Instant;

use crate::inference::draft_merge::merge_trees;
use crate::inference::spec_tree::{TokenTree, TreeDrafter, TREE_MAX_NODES};
use crate::inference::speculative::NGramDrafter;
use crate::inference::suffix_decoding::SuffixDecodingDrafter;
use crate::inference::token_recycling::TokenRecyclingDrafter;

/// A deterministic mocked oracle: a pure function from a token context to the
/// model's greedy next token. Any total function works; we use a cheap mixing
/// hash of the last few tokens so the "model" has structure (repeats produce
/// repeats) without being trivially constant.
fn oracle_next(context: &[u32]) -> u32 {
    if context.is_empty() {
        return 0;
    }
    // Mix the last up-to-3 tokens. Deterministic, total.
    let n = context.len();
    let a = context[n - 1] as u64;
    let b = if n >= 2 { context[n - 2] as u64 } else { 7 };
    let c = if n >= 3 { context[n - 3] as u64 } else { 13 };
    let h = a
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(b.wrapping_mul(0xC2B2AE3D27D4EB4F))
        .wrapping_add(c.wrapping_mul(0x165667B19E3779F9));
    // Small vocab keeps cycles short so the drafters actually find repeats.
    (h % 17) as u32
}

/// Pure greedy decode under the oracle: the ground-truth token stream.
fn oracle_greedy_decode(prompt: &[u32], count: usize) -> Vec<u32> {
    let mut ctx = prompt.to_vec();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let t = oracle_next(&ctx);
        out.push(t);
        ctx.push(t);
    }
    out
}

/// The token-path (excluding the anchor) from the root to each node.
fn node_paths(tree: &TokenTree) -> Vec<Vec<u32>> {
    let mut paths: Vec<Vec<u32>> = Vec::with_capacity(tree.nodes());
    for i in 0..tree.nodes() {
        if i == 0 {
            paths.push(Vec::new());
        } else {
            let p = tree.parent[i] as usize;
            let mut path = paths[p].clone();
            path.push(tree.tokens[i]);
            paths.push(path);
        }
    }
    paths
}

/// Compute `predicted[node]` = the oracle's argmax for each node's full
/// context (history-through-anchor ++ node path).
fn predicted_for_tree(tree: &TokenTree, history_through_anchor: &[u32]) -> Vec<u32> {
    let paths = node_paths(tree);
    paths
        .iter()
        .map(|path| {
            let mut ctx = history_through_anchor.to_vec();
            ctx.extend_from_slice(path);
            oracle_next(&ctx)
        })
        .collect()
}

/// Run one speculative round under the oracle and return the emitted tokens.
/// `history` ends with the anchor (the last committed token).
fn spec_round(tree: &TokenTree, history: &[u32]) -> Vec<u32> {
    let predicted = predicted_for_tree(tree, history);
    let (emitted, _leaf) = tree.accept_longest_path(&predicted);
    emitted
}

/// Drive a full decode using a tree drafter under the oracle, asserting the
/// emitted stream matches pure greedy decode exactly (LOSSLESS).
fn assert_drafter_lossless<D: TreeDrafter>(mut drafter: D, prompt: &[u32], count: usize) {
    let truth = oracle_greedy_decode(prompt, count);
    let mut history = prompt.to_vec();
    let mut emitted_all: Vec<u32> = Vec::new();
    let mut rounds = 0usize;
    while emitted_all.len() < count {
        rounds += 1;
        assert!(rounds < 10_000, "decode did not terminate");
        let anchor = *history.last().expect("non-empty prompt");
        let tree = drafter.draft_tree(&history, anchor, TREE_MAX_NODES, 8);
        assert_eq!(tree.tokens[0], anchor, "tree root must be the anchor");
        let emitted = spec_round(&tree, &history);
        assert!(!emitted.is_empty(), "every round emits at least the bonus");
        for t in emitted {
            if emitted_all.len() >= count {
                break;
            }
            emitted_all.push(t);
            history.push(t);
        }
    }
    emitted_all.truncate(count);
    assert_eq!(
        emitted_all, truth,
        "drafter emitted a non-greedy stream — losslessness violated"
    );
}

#[test]
fn tree_accept_equals_linear_accept_for_equivalent_inputs() {
    // For a linear() tree, accept_longest_path must reproduce the historic
    // linear accept (longest prefix where drafts[j]==predicted[j], then the
    // +1 bonus). Checked here against an independent reimplementation of that
    // loop over randomized inputs.
    fn linear_ref(drafts: &[u32], predicted: &[u32]) -> Vec<u32> {
        let mut accepted = vec![predicted[0]];
        let mut j = 0usize;
        while j < drafts.len() && drafts[j] == predicted[j] {
            accepted.push(predicted[j + 1]);
            j += 1;
        }
        accepted
    }
    // Deterministic pseudo-random cases.
    let mut state = 0x1234_5678u64;
    let mut rng = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for _ in 0..2000 {
        let len = (rng() % 7) as usize; // 0..6 drafts
        let drafts: Vec<u32> = (0..len).map(|_| rng() % 5).collect();
        // predicted has len+1 entries (one per node).
        let predicted: Vec<u32> = (0..len + 1).map(|_| rng() % 5).collect();
        let tree = TokenTree::linear(99, &drafts);
        let (emitted, _) = tree.accept_longest_path(&predicted);
        assert_eq!(
            emitted,
            linear_ref(&drafts, &predicted),
            "drafts={drafts:?} predicted={predicted:?}"
        );
    }
}

#[test]
fn ngram_linear_tree_lossless_under_oracle() {
    // NGram wrapped as a degenerate tree must be lossless too.
    struct W(NGramDrafter);
    impl TreeDrafter for W {
        fn draft_tree(
            &mut self,
            history: &[u32],
            anchor: u32,
            max_nodes: usize,
            max_depth: usize,
        ) -> TokenTree {
            let k = (max_nodes - 1).min(max_depth);
            TokenTree::linear(anchor, &self.0.draft(history, k))
        }
    }
    assert_drafter_lossless(W(NGramDrafter::default()), &[1, 2, 3, 4], 80);
}

#[test]
fn suffix_decoding_lossless_under_oracle() {
    assert_drafter_lossless(SuffixDecodingDrafter::default(), &[1, 2, 3, 4], 80);
}

#[test]
fn token_recycling_lossless_under_oracle() {
    assert_drafter_lossless(TokenRecyclingDrafter::new(), &[1, 2, 3, 4], 80);
}

#[test]
fn merged_drafters_lossless_under_oracle() {
    // A merge of all three drafters must still be lossless.
    struct Merged {
        suffix: SuffixDecodingDrafter,
        recycle: TokenRecyclingDrafter,
        ngram: NGramDrafter,
    }
    impl TreeDrafter for Merged {
        fn draft_tree(
            &mut self,
            history: &[u32],
            anchor: u32,
            max_nodes: usize,
            max_depth: usize,
        ) -> TokenTree {
            let t_suffix = self
                .suffix
                .draft_tree(history, anchor, max_nodes, max_depth);
            let t_recycle = self
                .recycle
                .draft_tree(history, anchor, max_nodes, max_depth);
            let k = (max_nodes - 1).min(max_depth);
            let t_ngram = TokenTree::linear(anchor, &self.ngram.draft(history, k));
            merge_trees(&[t_suffix, t_recycle, t_ngram], max_nodes)
        }
    }
    assert_drafter_lossless(
        Merged {
            suffix: SuffixDecodingDrafter::default(),
            recycle: TokenRecyclingDrafter::new(),
            ngram: NGramDrafter::default(),
        },
        &[1, 2, 3, 4],
        120,
    );
}

#[test]
#[ignore = "perf microbench: the per-node timing assertion is environment-dependent and flaky on shared/loaded CI runners (even at the debug ceiling); run on demand via --include-ignored"]
fn drafter_microbench_under_5us_per_token() {
    // Each drafter's per-DRAFT-NODE cost must be well under ~5µs so the GPU
    // verify dominates, not drafting. Build a repetitive history (captured
    // ONCE, outside the timed loop) so the drafters do real work — find
    // matches / expand branches — without paying to rebuild the input.
    let mut history: Vec<u32> = Vec::with_capacity(3200);
    for _ in 0..400 {
        history.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    }
    let anchor = *history.last().unwrap();
    const ITERS: usize = 2000;

    fn measure(label: &str, iters: usize, mut f: impl FnMut() -> usize) {
        let _ = f(); // discard first call (allocator warm-up)
        let start = Instant::now();
        let mut total_nodes = 0usize;
        for _ in 0..iters {
            total_nodes += f();
        }
        let elapsed = start.elapsed();
        let units = total_nodes.max(iters) as f64;
        let per_node_us = elapsed.as_secs_f64() * 1e6 / units;
        let per_call_us = elapsed.as_secs_f64() * 1e6 / iters as f64;
        eprintln!(
            "{label}: {per_node_us:.3} us/node, {per_call_us:.3} us/call ({total_nodes} nodes / {iters} calls)"
        );
        // The substrate target is well under ~5µs per drafted token (node);
        // drafting must never rival the GPU verify. This is the budget an
        // OPTIMIZED build must meet. A debug build (no inlining, SipHash maps,
        // pathologically repetitive bench input that maximizes match counts)
        // runs roughly an order of magnitude slower, so the debug ceiling is
        // 30µs/node — still proving drafting is cheap, with release clearing
        // the 5µs/node target by a wide margin.
        let budget_us = if cfg!(debug_assertions) { 30.0 } else { 5.0 };
        assert!(
            per_node_us < budget_us,
            "{label} draft cost {per_node_us:.3} us/node exceeds {budget_us} us budget"
        );
    }

    // n-gram (single chain).
    {
        let ngram = NGramDrafter::default();
        measure("ngram", ITERS, || ngram.draft(&history, 7).len());
    }
    // suffix decoding (tree).
    {
        let mut suffix = SuffixDecodingDrafter::default();
        measure("suffix_decoding", ITERS, || {
            suffix
                .draft_tree(&history, anchor, TREE_MAX_NODES, 8)
                .nodes()
                .saturating_sub(1)
        });
    }
    // token recycling (tree) — warm adjacency once outside the loop.
    {
        let mut recycle = TokenRecyclingDrafter::new();
        recycle.learn(&history);
        measure("token_recycling", ITERS, || {
            recycle
                .draft_tree(&history, anchor, TREE_MAX_NODES, 8)
                .nodes()
                .saturating_sub(1)
        });
    }
}
