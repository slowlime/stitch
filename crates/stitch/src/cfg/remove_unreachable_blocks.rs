use log::trace;
use slotmap::SecondaryMap;

use super::{BlockId, FuncBody};

impl FuncBody {
    pub fn remove_unreachable_blocks(&mut self) {
        let mut stack = vec![self.entry];
        let mut reachable = SecondaryMap::<BlockId, ()>::new();
        reachable.insert(self.entry, ());

        while let Some(block_id) = stack.pop() {
            for &succ_block_id in self.blocks[block_id].successors() {
                if reachable.insert(succ_block_id, ()).is_none() {
                    stack.push(succ_block_id);
                }
            }
        }

        self.blocks.retain(|block_id, _| {
            let contains_key = reachable.contains_key(block_id);

            if !contains_key {
                trace!("removing {block_id:?}");
            }

            contains_key
        });
    }
}
