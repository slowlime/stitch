use std::cmp::Ordering;

use slotmap::SecondaryMap;

use super::{BlockId, FuncBody, Predecessors, Rpo};

#[derive(Debug, Default, Clone)]
pub struct DomTree {
    pub idom: SecondaryMap<BlockId, BlockId>,
    pub succ: SecondaryMap<BlockId, Vec<BlockId>>,
}

impl DomTree {
    pub fn dominates(&self, lhs: BlockId, mut rhs: BlockId) -> bool {
        loop {
            if lhs == rhs {
                return true;
            } else if let Some(&idom) = self.idom.get(rhs) {
                rhs = idom;
            } else {
                return false;
            }
        }
    }
}

impl FuncBody {
    pub fn dom_tree(&self, preds: &Predecessors, rpo: &Rpo) -> DomTree {
        // See https://www.clear.rice.edu/comp512/Lectures/Papers/TR06-33870-Dom.pdf

        fn intersect(
            idom: &SecondaryMap<BlockId, BlockId>,
            rpo: &Rpo,
            mut lhs: BlockId,
            mut rhs: BlockId,
        ) -> BlockId {
            loop {
                match rpo.idx[lhs].cmp(&rpo.idx[rhs]) {
                    Ordering::Less => lhs = idom[lhs],
                    Ordering::Greater => rhs = idom[rhs],
                    Ordering::Equal => return lhs,
                }
            }
        }

        let mut idom = SecondaryMap::<BlockId, BlockId>::new();
        idom.insert(self.entry, self.entry);

        let mut changed = true;

        while changed {
            changed = false;

            for &block_id in &rpo.order {
                let Some((_, mut new_idom)) = preds[block_id]
                    .iter()
                    .filter_map(|&pred_block_id| {
                        rpo.idx
                            .get(pred_block_id)
                            .map(|&pred_idx| (pred_idx, pred_block_id))
                    })
                    // pred_idx < rpo.idx[block_id] implies pred_block_id has been processed
                    .find(|&(pred_idx, _)| pred_idx < rpo.idx[block_id])
                else {
                    continue;
                };

                for &pred_block_id in &preds[block_id] {
                    if pred_block_id == new_idom {
                        continue;
                    }

                    if idom.contains_key(pred_block_id) {
                        new_idom = intersect(&idom, rpo, pred_block_id, new_idom);
                    }
                }

                if idom.insert(block_id, new_idom) != Some(new_idom) {
                    changed = true;
                }
            }
        }

        idom.remove(self.entry);
        let mut succ = SecondaryMap::<BlockId, Vec<BlockId>>::new();

        for (block_id, &idom_block_id) in &idom {
            succ.entry(idom_block_id)
                .unwrap()
                .or_default()
                .push(block_id);
        }

        DomTree { idom, succ }
    }
}
