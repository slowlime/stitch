use std::collections::HashSet;

use slotmap::{SecondaryMap, SparseSecondaryMap};

use super::{BlockId, DomTree, FuncBody, Predecessors, Rpo};

pub struct Loops {
    /// Maps a loop header's block id to the block id of its immediately enclosing loop, if any.
    pub parent_loops: SparseSecondaryMap<BlockId, BlockId>,

    /// Maps a block id to the block id of the header of the innermost loop it's part of, if any.
    pub members: SecondaryMap<BlockId, BlockId>,

    /// A set of all loop headers.
    pub loop_headers: HashSet<BlockId>,
}

impl FuncBody {
    pub fn loop_headers(&self, rpo: &Rpo) -> HashSet<BlockId> {
        let mut result = HashSet::new();

        for (block_id, block) in &self.blocks {
            for &succ_block_id in block.successors() {
                if rpo.is_backward_edge(block_id, succ_block_id) {
                    result.insert(succ_block_id);
                }
            }
        }

        result
    }

    pub fn parent_loops(
        &self,
        loop_headers: &HashSet<BlockId>,
        dom_tree: &DomTree,
    ) -> SparseSecondaryMap<BlockId, BlockId> {
        let mut result = SparseSecondaryMap::new();
        let mut stack = vec![(self.entry, 0, None)];

        while let Some(&mut (block_id, ref mut next_idx, parent_loop)) = stack.last_mut() {
            if let Some(&child_block_id) = dom_tree.succ[block_id].get(*next_idx) {
                *next_idx += 1;
                stack.push((
                    child_block_id,
                    0,
                    if loop_headers.contains(&block_id) {
                        Some(block_id)
                    } else {
                        parent_loop
                    },
                ));
            } else {
                if loop_headers.contains(&block_id) {
                    if let Some(parent_block_id) = parent_loop {
                        result.insert(block_id, parent_block_id);
                    }
                }

                stack.pop();
            }
        }

        result
    }

    pub fn loops(&self, preds: &Predecessors, rpo: &Rpo, dom_tree: &DomTree) -> Loops {
        let loop_headers = self.loop_headers(rpo);
        let parent_loops = self.parent_loops(&loop_headers, dom_tree);
        let mut members = SecondaryMap::new();

        for &block_id in &loop_headers {
            members.insert(block_id, block_id);
        }

        let mut stack = vec![self.entry];
        let mut discovered = HashSet::new();
        discovered.insert(self.entry);

        while let Some(block_id) = stack.pop() {
            for &succ_block_id in self.blocks[block_id].successors() {
                if !rpo.is_backward_edge(block_id, succ_block_id) {
                    continue;
                }

                let loop_header_id = block_id;
                let mut loop_stack = vec![succ_block_id];
                let mut loop_discovered = HashSet::new();
                loop_discovered.extend([block_id, succ_block_id]);

                while let Some(block_id) = loop_stack.pop() {
                    for &pred_block_id in &preds[block_id] {
                        if loop_discovered.insert(pred_block_id) {
                            loop_stack.push(pred_block_id);
                        }
                    }

                    match members.get(block_id) {
                        Some(&block_id) if dom_tree.dominates(loop_header_id, block_id) => {
                            // block_id is nested within loop_header_id
                        }

                        _ => {
                            members.insert(block_id, loop_header_id);
                        }
                    }
                }
            }
        }

        Loops {
            loop_headers,
            parent_loops,
            members,
        }
    }
}
