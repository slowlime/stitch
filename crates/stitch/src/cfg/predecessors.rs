use slotmap::SecondaryMap;

use super::{BlockId, FuncBody};

pub type Predecessors = SecondaryMap<BlockId, Vec<BlockId>>;

impl FuncBody {
    pub fn predecessors(&self) -> Predecessors {
        let mut preds = self
            .blocks
            .keys()
            .map(|block_id| (block_id, vec![]))
            .collect::<Predecessors>();

        for (block_id, block) in &self.blocks {
            for &succ_block_id in block.successors() {
                preds[succ_block_id].push(block_id);
            }
        }

        for preds in preds.values_mut() {
            preds.sort_unstable();
            preds.dedup();
        }

        preds
    }
}
