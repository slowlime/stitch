use std::collections::HashSet;

use log::trace;

use super::{FuncBody, Terminator};

impl FuncBody {
    pub fn merge_blocks(&mut self) {
        let mut preds = self.predecessors();
        let mut block_ids = self.blocks.keys().collect::<HashSet<_>>();

        while let Some(&block_id) = block_ids.iter().next() {
            block_ids.remove(&block_id);

            if let Terminator::Br(target_block_id) = self.blocks[block_id].term {
                if target_block_id != self.entry
                    && block_id != target_block_id
                    && preds[target_block_id].len() == 1
                {
                    trace!("merging {target_block_id:?} into {block_id:?}");
                    block_ids.remove(&target_block_id);
                    block_ids.insert(block_id);
                    let target_block = self.blocks.remove(target_block_id).unwrap();
                    let block = &mut self.blocks[block_id];

                    block.body.extend(target_block.body);
                    block.term = target_block.term;

                    preds.remove(target_block_id);

                    for &succ_block_id in block.successors() {
                        let succ_preds = &mut preds[succ_block_id];

                        if let Ok(idx) = succ_preds.binary_search(&target_block_id) {
                            // block.successors() may return the same succ_block_id multiple times,
                            // so its predecessor list may have already been updated
                            succ_preds.remove(idx);
                        }

                        if let Err(idx) = succ_preds.binary_search(&block_id) {
                            succ_preds.insert(idx, block_id);
                        }
                    }
                }
            }
        }
    }
}
