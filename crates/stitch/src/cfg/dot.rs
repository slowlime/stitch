use std::fmt::{self, Display};

use slotmap::Key;

use crate::cfg::BlockId;

use super::FuncBody;

impl FuncBody {
    pub fn dot(&self) -> impl Display + '_ {
        struct BlockIdPrinter(BlockId);

        impl Display for BlockIdPrinter {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let id = self.0.data().as_ffi();

                write!(f, "b{}_{}", id as u32, id >> 32)
            }
        }

        struct DotPrinter<'a>(&'a FuncBody);

        impl Display for DotPrinter<'_> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                writeln!(f, "digraph cfg {{")?;
                writeln!(f, "  node [shape=rect];")?;
                writeln!(f, "  edge [minlen=1; dir=forward];")?;
                writeln!(f)?;

                for block_id in self.0.blocks.keys() {
                    writeln!(
                        f,
                        "  {} [label = \"{block_id:?}\"];",
                        BlockIdPrinter(block_id),
                    )?;
                }

                writeln!(f)?;

                for (block_id, block) in &self.0.blocks {
                    for &succ_block_id in block.successors() {
                        writeln!(
                            f,
                            "  {} -> {};",
                            BlockIdPrinter(block_id),
                            BlockIdPrinter(succ_block_id),
                        )?;
                    }
                }

                writeln!(f, "}}")
            }
        }

        DotPrinter(self)
    }
}
