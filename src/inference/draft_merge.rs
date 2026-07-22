//! Draft-tree merging (Lane A).
//!
//! Combine several drafters' trees (n-gram, suffix decoding, token recycling)
//! into one bounded tree the batched verify consumes. Different drafters win on
//! different text (n-gram on exact repeats, suffix decoding on
//! frequency-weighted recurrence, token recycling on learned local
//! transitions), so a merged tree captures more candidate paths per verify pass
//! at no extra model cost.
//!
//! Dedup is keyed on the **parent path** (the sequence of tokens from the root
//! to the candidate), NOT the bare token id. Two different branches can both
//! propose token `t`, but if they sit under different prefixes they are
//! genuinely different continuations and must both survive; only a token that
//! repeats the *same path* is a duplicate. This keying is the crux and is
//! unit-tested.
//!
//! All-Rust, GPU-free, lossless (the verify is authoritative — the merged tree
//! only changes which candidates get a verify row).

use std::collections::HashMap;

use crate::inference::spec_tree::{TokenTree, TreeDrafter, TREE_MAX_NODES};

/// Merge `trees` into one bounded tree. All inputs must share the same root
/// anchor token (the last committed token); the merged tree keeps that anchor
/// as node 0. The merge is breadth-first so the result stays in BFS order
/// (`parent[i] < i`) and respects `max_nodes` (≤ [`TREE_MAX_NODES`]).
///
/// Dedup key: the full root-to-node token path. The first tree to contribute a
/// given path wins; later trees proposing the same path are folded in (their
/// subtrees still get a chance to extend that shared node).
pub fn merge_trees(trees: &[TokenTree], max_nodes: usize) -> TokenTree {
    let cap = max_nodes.clamp(1, TREE_MAX_NODES);
    // Determine the anchor. Empty input or empty trees ⇒ a 1-node placeholder
    // is impossible without a token, so require at least one non-empty tree;
    // callers always pass the anchor tree.
    let anchor = trees
        .iter()
        .find(|t| !t.tokens.is_empty())
        .map(|t| t.tokens[0])
        .unwrap_or(0);

    let mut out = TokenTree::linear(anchor, &[]);
    // Map from a node's token-path (Vec<u32>, excluding the anchor) to its
    // index in `out`. The root's path is the empty vec.
    let mut path_to_idx: HashMap<Vec<u32>, usize> = HashMap::new();
    path_to_idx.insert(Vec::new(), 0);

    // Insert source nodes in increasing depth so parents always exist first.
    // Each source node carries its own token-path; we look up (or create) the
    // corresponding node in `out`.
    let mut by_depth: Vec<(u16, Vec<u32>, usize)> = Vec::new(); // (depth, parent_path, token)
    for tree in trees {
        // Precompute each node's token-path within this tree.
        let mut node_path: Vec<Vec<u32>> = Vec::with_capacity(tree.nodes());
        for i in 0..tree.nodes() {
            if i == 0 {
                node_path.push(Vec::new());
            } else {
                let p = tree.parent[i] as usize;
                let mut path = node_path[p].clone();
                path.push(tree.tokens[i]);
                node_path.push(path);
            }
        }
        for i in 1..tree.nodes() {
            let parent_path = node_path[tree.parent[i] as usize].clone();
            by_depth.push((tree.depth[i], parent_path, tree.tokens[i] as usize));
        }
    }
    // Stable BFS insertion: by depth, then by insertion order.
    by_depth.sort_by_key(|x| x.0);

    for (depth, parent_path, token) in by_depth {
        if out.nodes() >= cap {
            break;
        }
        let token = token as u32;
        // Parent must already exist (we inserted shallower depths first). If
        // the parent path was dropped (cap), skip this node.
        let &parent_idx = match path_to_idx.get(&parent_path) {
            Some(idx) => idx,
            None => continue,
        };
        // Full path of this candidate = parent_path ++ [token].
        let mut child_path = parent_path.clone();
        child_path.push(token);
        // Dedup on the FULL path, not the bare token.
        if path_to_idx.contains_key(&child_path) {
            continue;
        }
        let new_idx = out.nodes();
        out.tokens.push(token);
        out.parent.push(parent_idx as i32);
        out.depth.push(depth);
        path_to_idx.insert(child_path, new_idx);
    }
    out
}

/// Convenience: run several [`TreeDrafter`]s and merge their trees.
pub fn draft_and_merge(
    drafters: &mut [&mut dyn TreeDrafter],
    history: &[u32],
    anchor: u32,
    max_nodes: usize,
    max_depth: usize,
) -> TokenTree {
    let trees: Vec<TokenTree> = drafters
        .iter_mut()
        .map(|d| d.draft_tree(history, anchor, max_nodes, max_depth))
        .collect();
    merge_trees(&trees, max_nodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_disjoint_branches() {
        // Two linear chains off the same anchor: 10 -> 11 -> 12 and 10 -> 20.
        let a = TokenTree::linear(10, &[11, 12]);
        let b = TokenTree::linear(10, &[20]);
        let merged = merge_trees(&[a, b], TREE_MAX_NODES);
        assert_eq!(merged.tokens[0], 10);
        // Root has two children: 11 and 20.
        let children: Vec<u32> = (1..merged.nodes())
            .filter(|&i| merged.parent[i] == 0)
            .map(|i| merged.tokens[i])
            .collect();
        assert!(children.contains(&11));
        assert!(children.contains(&20));
        // 12 hangs under 11.
        let idx12 = (1..merged.nodes())
            .find(|&i| merged.tokens[i] == 12)
            .unwrap();
        let parent12 = merged.parent[idx12] as usize;
        assert_eq!(merged.tokens[parent12], 11);
        // BFS invariant.
        for i in 1..merged.nodes() {
            assert!(merged.parent[i] < i as i32);
        }
    }

    #[test]
    fn dedup_keys_on_path_not_bare_token() {
        // KEY TEST. Same token 99 under two DIFFERENT paths must both survive;
        // the same token 99 under the SAME path is a duplicate and merged.
        //
        // Tree A: 10 -> 11 -> 99   (path [11,99])
        //         10 -> 22 -> 99   (path [22,99])  <- different path, same token
        let mut a = TokenTree::linear(10, &[11]);
        // attach 99 under 11 (node 1)
        a.tokens.push(99);
        a.parent.push(1);
        a.depth.push(2);
        // attach 22 under root (node 0)
        a.tokens.push(22);
        a.parent.push(0);
        a.depth.push(1);
        // attach 99 under 22 (node 3)
        a.tokens.push(99);
        a.parent.push(3);
        a.depth.push(2);

        // Tree B duplicates the path [11,99] exactly — should be deduped.
        let mut b = TokenTree::linear(10, &[11]);
        b.tokens.push(99);
        b.parent.push(1);
        b.depth.push(2);

        let merged = merge_trees(&[a, b], TREE_MAX_NODES);
        // Count occurrences of token 99 by their full path.
        let path_of = |t: &TokenTree, mut i: usize| -> Vec<u32> {
            let mut p = Vec::new();
            while i != 0 {
                p.push(t.tokens[i]);
                i = t.parent[i] as usize;
            }
            p.reverse();
            p
        };
        let paths_to_99: Vec<Vec<u32>> = (1..merged.nodes())
            .filter(|&i| merged.tokens[i] == 99)
            .map(|i| path_of(&merged, i))
            .collect();
        // Exactly two distinct 99-nodes: [11,99] and [22,99]. The duplicate
        // [11,99] from tree B was merged away.
        assert_eq!(paths_to_99.len(), 2, "two distinct paths, no over-merge");
        assert!(paths_to_99.contains(&vec![11, 99]));
        assert!(paths_to_99.contains(&vec![22, 99]));
    }

    #[test]
    fn respects_node_cap() {
        let a = TokenTree::linear(10, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        let merged = merge_trees(&[a], 5);
        assert!(merged.nodes() <= 5);
        // Still BFS-valid.
        for i in 1..merged.nodes() {
            assert!(merged.parent[i] < i as i32);
        }
    }

    #[test]
    fn shallower_nodes_inserted_first() {
        // A deep branch and a shallow branch; with a tight cap the shallow
        // (depth-1) nodes must win the budget over deep ones.
        let deep = TokenTree::linear(10, &[1, 2, 3]); // depths 1,2,3
        let mut shallow = TokenTree::linear(10, &[]);
        shallow.tokens.push(50);
        shallow.parent.push(0);
        shallow.depth.push(1);
        shallow.tokens.push(51);
        shallow.parent.push(0);
        shallow.depth.push(1);
        let merged = merge_trees(&[deep, shallow], 3); // root + 2 nodes
        assert_eq!(merged.nodes(), 3);
        // Both extra nodes should be depth 1 (the breadth-first budget).
        assert!(merged.depth.iter().all(|&d| d <= 1));
    }
}
