use slotmap::SecondaryMap;

use super::{BlockId, FuncBody};

#[derive(Debug, Default, Clone)]
pub struct Rpo {
    pub order: Vec<BlockId>,
    pub idx: SecondaryMap<BlockId, usize>,
}

impl FuncBody {
    pub fn rpo(&self) -> Rpo {
        struct Task {
            block_id: BlockId,
            next_child: usize,
        }

        let mut order = vec![];
        let mut discovered = SecondaryMap::<BlockId, ()>::new();
        discovered.insert(self.entry, ());

        let mut stack = vec![Task {
            block_id: self.entry,
            next_child: 0,
        }];

        while let Some(&mut Task {
            block_id,
            ref mut next_child,
        }) = stack.last_mut()
        {
            if let Some(&succ_block_id) = self.blocks[block_id].successors().get(*next_child) {
                *next_child += 1;

                if discovered.insert(succ_block_id, ()).is_none() {
                    stack.push(Task {
                        block_id: succ_block_id,
                        next_child: 0,
                    });
                }
            } else {
                order.push(block_id);
                stack.pop();
            }
        }

        order.reverse();
        let idx = order
            .iter()
            .enumerate()
            .map(|(idx, &block_id)| (block_id, idx))
            .collect();

        Rpo { order, idx }
    }
}
