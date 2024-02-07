use slotmap::SecondaryMap;

use super::{BlockId, FuncBody};

impl FuncBody {
    pub fn predecessors(&self) -> SecondaryMap<BlockId, Vec<BlockId>> {
        let mut preds = SecondaryMap::<BlockId, Vec<BlockId>>::new();

        for (block_id, block) in &self.blocks {
            for &succ_block_id in block.successors() {
                preds.entry(succ_block_id).unwrap().or_default().push(block_id);
            }
        }

        for preds in preds.values_mut() {
            preds.sort_unstable();
            preds.dedup();
        }

        preds
    }
}
