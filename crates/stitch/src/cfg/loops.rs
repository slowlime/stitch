use std::collections::HashSet;

use super::{BlockId, FuncBody, Rpo};

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
}
