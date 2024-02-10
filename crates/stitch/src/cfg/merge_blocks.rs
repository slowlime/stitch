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
                if target_block_id == self.entry || block_id == target_block_id {
                    continue;
                }

                if self.blocks[block_id].body.is_empty() {
                    trace!("eliminating empty {block_id:?} in favor of {target_block_id:?}");
                    self.blocks.remove(block_id).unwrap();

                    let [&mut ref block_preds, target_preds] =
                        preds.get_disjoint_mut([block_id, target_block_id]).unwrap();

                    // make predecessors pointing to block_id refer to target_block_id
                    for &pred_block_id in block_preds {
                        for pred_succ_block_id in self.blocks[pred_block_id].successors_mut() {
                            if *pred_succ_block_id == block_id {
                                *pred_succ_block_id = target_block_id;
                            }
                        }
                    }

                    target_preds.remove(target_preds.binary_search(&block_id).unwrap());

                    // add block_id's predecessors to target_block_id's predecessor list
                    for &pred_block_id in block_preds {
                        if let Err(idx) = target_preds.binary_search(&pred_block_id) {
                            target_preds.insert(idx, pred_block_id);
                        }
                    }

                    preds.remove(block_id);

                    if self.entry == block_id {
                        self.entry = target_block_id;
                    }
                } else if preds[target_block_id].len() == 1 {
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

        assert!(self.blocks.contains_key(self.entry));

        for block in self.blocks.values() {
            for &succ_block_id in block.successors() {
                assert!(self.blocks.contains_key(succ_block_id));
            }
        }
    }
}
