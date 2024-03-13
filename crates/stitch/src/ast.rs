//! A simple AST-based IR for a WebAssembly module.

pub mod expr;
pub mod func;
pub mod ty;

use std::fmt::{self, Display};
use std::ops::Range;

use slotmap::{new_key_type, SlotMap};
use thiserror::Error;

use crate::util::slot::BiSlotMap;

use self::func::FuncImport;
use self::ty::{ElemType, FuncType, GlobalType, MemoryType, TableType, Type, ValType};

pub use self::expr::{ConstExpr, Expr};
pub use self::func::{Func, FuncBody};

pub const PAGE_SIZE: usize = 65536;
pub const STITCH_MODULE_NAME: &str = "stitch";

new_key_type! {
    pub struct TypeId;
    pub struct FuncId;
    pub struct TableId;
    pub struct MemoryId;
    pub struct GlobalId;
    pub struct ImportId;
    pub struct ExportId;
    pub struct LocalId;
    pub struct BlockId;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IntrinsicDecl {
    ArgCount,
    ArgLen,
    ArgRead,
    Specialize,
    Unknown,
    ConstPtr,
    SymbolicPtr,
    ConcretePtr,
    PropagateLoad,
    PrintValue,
    PrintStr,
    IsSpecializing,
    Inline,
    NoInline,
    FileOpen,
    FileRead,
    FileClose,
    FuncSpecPolicy,
    SymbolicStackPtr,
}

impl IntrinsicDecl {
    pub fn check_ty(&self, func_ty: &FuncType) -> Result<(), String> {
        match self {
            Self::ArgCount if func_ty.params.is_empty() && func_ty.ret == Some(ValType::I32) => {
                Ok(())
            }
            Self::ArgCount => Err("[] -> [i32]".into()),

            Self::ArgLen
                if func_ty.params.len() == 1
                    && func_ty.params[0] == ValType::I32
                    && func_ty.ret == Some(ValType::I32) =>
            {
                Ok(())
            }
            Self::ArgLen => Err("[i32] -> [i32]".into()),

            Self::ArgRead
                if func_ty.params.len() == 4
                    && func_ty.params.iter().all(|ty| *ty == ValType::I32)
                    && func_ty.ret == Some(ValType::I32) =>
            {
                Ok(())
            }
            Self::ArgRead => Err("[i32 i32 i32 i32] -> [i32]".into()),

            Self::Specialize
                if func_ty.params.len() >= 3
                    && func_ty.params[0..3].iter().all(|ty| *ty == ValType::I32)
                    && func_ty.ret == Some(ValType::I32) =>
            {
                Ok(())
            }
            Self::Specialize => Err("[i32 i32 i32 t...] -> [i32]".into()),

            Self::Unknown if func_ty.params.is_empty() && func_ty.ret.is_some() => Ok(()),
            Self::Unknown => Err("[] -> [t]".into()),

            Self::ConstPtr | Self::SymbolicPtr | Self::ConcretePtr | Self::PropagateLoad
                if func_ty.params.len() == 1
                    && func_ty.params[0] == ValType::I32
                    && func_ty.ret == Some(ValType::I32) =>
            {
                Ok(())
            }
            Self::ConstPtr | Self::SymbolicPtr | Self::ConcretePtr | Self::PropagateLoad => {
                Err("[i32] -> [i32]".into())
            }

            Self::PrintValue if func_ty.params.len() == 1 && func_ty.ret.is_none() => Ok(()),
            Self::PrintValue => Err("[t] -> []".into()),

            Self::PrintStr
                if func_ty.params.len() == 2
                    && func_ty.params.iter().all(|ty| *ty == ValType::I32)
                    && func_ty.ret.is_none() =>
            {
                Ok(())
            }
            Self::PrintStr => Err("[i32 i32] -> []".into()),

            Self::IsSpecializing
                if func_ty.params.is_empty() && func_ty.ret == Some(ValType::I32) =>
            {
                Ok(())
            }
            Self::IsSpecializing => Err("[] -> [i32]".into()),

            Self::Inline | Self::NoInline
                if func_ty.params.len() == 1
                    && func_ty.params[0] == ValType::I32
                    && func_ty.ret == Some(ValType::I32) =>
            {
                Ok(())
            }
            Self::Inline | Self::NoInline => Err("[i32] -> [i32]".into()),

            Self::FileOpen
                if func_ty.params.len() == 3
                    && func_ty.params.iter().all(|ty| *ty == ValType::I32)
                    && func_ty.ret == Some(ValType::I32) =>
            {
                Ok(())
            }
            Self::FileOpen => Err("[i32 i32 i32] -> [i32]".into()),

            Self::FileRead
                if func_ty.params.len() == 4
                    && func_ty.params.iter().all(|ty| *ty == ValType::I32)
                    && func_ty.ret == Some(ValType::I32) =>
            {
                Ok(())
            }
            Self::FileRead => Err("[i32 i32 i32 i32] -> [i32]".into()),

            Self::FileClose
                if func_ty.params.len() == 1
                    && func_ty.params[0] == ValType::I32
                    && func_ty.ret == Some(ValType::I32) =>
            {
                Ok(())
            }
            Self::FileClose => Err("[i32] -> [i32]".into()),

            Self::FuncSpecPolicy
                if func_ty.params.len() == 3
                    && func_ty.params.iter().all(|ty| *ty == ValType::I32)
                    && func_ty.ret.is_none() => {
                Ok(())
            }
            Self::FuncSpecPolicy => Err("[i32 i32 i32] -> []".into()),

            Self::SymbolicStackPtr
                if func_ty.params.is_empty() && func_ty.ret.is_none() => Ok(()),
            Self::SymbolicStackPtr => Err("[] -> []".into()),
        }
    }
}

impl Display for IntrinsicDecl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{STITCH_MODULE_NAME}/{}",
            match self {
                Self::ArgCount => "arg-count",
                Self::ArgLen => "arg-len",
                Self::ArgRead => "arg-read",
                Self::Specialize => "specialize",
                Self::Unknown => "unknown",
                Self::ConstPtr => "const-ptr",
                Self::SymbolicPtr => "symbolic-ptr",
                Self::ConcretePtr => "concrete-ptr",
                Self::PropagateLoad => "propagate-load",
                Self::PrintValue => "print-value",
                Self::PrintStr => "print-str",
                Self::IsSpecializing => "is-specializing",
                Self::Inline => "inline",
                Self::NoInline => "no-inline",
                Self::FileOpen => "file-open",
                Self::FileRead => "file-read",
                Self::FileClose => "file-close",
                Self::FuncSpecPolicy => "func-spec-policy",
                Self::SymbolicStackPtr => "symbolic-stack-ptr",
            }
        )
    }
}

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum MemError {
    #[error("cannot access an imported memory")]
    Import,

    #[error("range 0x{:x}..0x{:x} is out of bounds for a memory of size {size}", .range.start, .range.end)]
    OutOfBounds { range: Range<usize>, size: usize },
}

#[derive(Debug, Default, Clone)]
pub struct Module {
    pub name: Option<String>,
    pub types: BiSlotMap<TypeId, Type>,
    pub funcs: SlotMap<FuncId, Func>,
    pub tables: SlotMap<TableId, Table>,
    pub default_table: TableId,
    pub mems: SlotMap<MemoryId, Memory>,
    pub default_mem: MemoryId,
    pub globals: SlotMap<GlobalId, Global>,
    pub start: Option<FuncId>,
    pub imports: SlotMap<ImportId, Import>,
    pub exports: SlotMap<ExportId, Export>,
}

impl Module {
    pub fn get_intrinsic(&self, import_id: ImportId) -> Option<IntrinsicDecl> {
        let name = match &self.imports[import_id] {
            Import { module, name, .. } if module == STITCH_MODULE_NAME => name.as_str(),
            _ => return None,
        };

        let (name, _suffix) = match name.find('#') {
            Some(idx) => name.split_at(idx),
            None => (name, ""),
        };

        Some(match name {
            "arg-count" => IntrinsicDecl::ArgCount,
            "arg-len" => IntrinsicDecl::ArgLen,
            "arg-read" => IntrinsicDecl::ArgRead,
            "specialize" => IntrinsicDecl::Specialize,
            "unknown" => IntrinsicDecl::Unknown,
            "const-ptr" => IntrinsicDecl::ConstPtr,
            "symbolic-ptr" => IntrinsicDecl::SymbolicPtr,
            "concrete-ptr" => IntrinsicDecl::ConcretePtr,
            "propagate-load" => IntrinsicDecl::PropagateLoad,
            "print-value" => IntrinsicDecl::PrintValue,
            "print-str" => IntrinsicDecl::PrintStr,
            "is-specializing" => IntrinsicDecl::IsSpecializing,
            "inline" => IntrinsicDecl::Inline,
            "no-inline" => IntrinsicDecl::NoInline,
            "file-open" => IntrinsicDecl::FileOpen,
            "file-read" => IntrinsicDecl::FileRead,
            "file-close" => IntrinsicDecl::FileClose,
             "func-spec-policy" => IntrinsicDecl::FuncSpecPolicy,
            "symbolic-stack-ptr" => IntrinsicDecl::SymbolicStackPtr,
            _ => return None,
        })
    }

    pub fn remove_func(&mut self, func_id: FuncId) -> Option<Func> {
        let result = self.funcs.remove(func_id)?;

        for table in self.tables.values_mut() {
            if !matches!(table.ty.elem_ty, ElemType::Funcref) {
                continue;
            }

            if let TableDef::Elems(elems) = &mut table.def {
                for elem in elems {
                    if elem.is_some_and(|elem_func_id| elem_func_id == func_id) {
                        elem.take();
                    }
                }
            }
        }

        if self
            .start
            .is_some_and(|start_func_id| start_func_id == func_id)
        {
            self.start.take();
        }

        if let Func::Import(FuncImport { import_id, .. }) = result {
            self.imports.remove(import_id);
        }

        self.exports.retain(|_, export| match export.def {
            ExportDef::Func(export_func_id) => func_id != export_func_id,
            _ => true,
        });

        // FIXME: remove direct calls to the function

        Some(result)
    }

    pub fn get_mem(&self, mem_id: MemoryId, range: Range<usize>) -> Result<&[u8], MemError> {
        match &self.mems[mem_id].def {
            MemoryDef::Import(_) => Err(MemError::Import),

            MemoryDef::Bytes(bytes) => bytes.get(range.clone()).ok_or(MemError::OutOfBounds {
                range,
                size: bytes.len(),
            }),
        }
    }

    pub fn get_mem_mut(
        &mut self,
        mem_id: MemoryId,
        range: Range<usize>,
    ) -> Result<&mut [u8], MemError> {
        match &mut self.mems[mem_id].def {
            MemoryDef::Import(_) => Err(MemError::Import),

            MemoryDef::Bytes(bytes) => {
                let size = bytes.len();

                bytes
                    .get_mut(range.clone())
                    .ok_or(MemError::OutOfBounds { range, size })
            }
        }
    }

    pub fn func_name(&self, func_id: FuncId) -> String {
        let func = &self.funcs[func_id];

        match self.funcs[func_id].name() {
            Some(name) => format!("${name}"),

            None => match func {
                Func::Import(import) => {
                    let Import { module, name, .. } = &self.imports[import.import_id];

                    format!("<import: module '{module}', name '{name}'>")
                }

                _ => format!("{func_id:?}"),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Table {
    pub name: Option<String>,
    pub ty: TableType,
    pub def: TableDef,
}

impl Table {
    pub fn new(ty: TableType) -> Self {
        let elems = vec![None; ty.limits.min as usize];

        Self {
            name: None,
            ty,
            def: TableDef::Elems(elems),
        }
    }
}

#[derive(Debug, Clone)]
pub enum TableDef {
    Import(ImportId),
    Elems(Vec<Option<FuncId>>),
}

impl TableDef {
    pub fn is_import(&self) -> bool {
        matches!(self, Self::Import(_))
    }
}

#[derive(Debug, Clone)]
pub struct Memory {
    pub name: Option<String>,
    pub ty: MemoryType,
    pub def: MemoryDef,
}

#[derive(Debug, Clone)]
pub enum MemoryDef {
    Import(ImportId),
    Bytes(Vec<u8>),
}

impl MemoryDef {
    pub fn is_import(&self) -> bool {
        matches!(self, Self::Import(_))
    }
}

#[derive(Debug, Clone)]
pub struct Global {
    pub name: Option<String>,
    pub ty: GlobalType,
    pub def: GlobalDef,
}

#[derive(Debug, Clone)]
pub enum GlobalDef {
    Import(ImportId),
    Value(ConstExpr),
}

impl GlobalDef {
    pub fn is_import(&self) -> bool {
        matches!(self, Self::Import(_))
    }
}

#[derive(Debug, Clone)]
pub struct Import {
    pub module: String,
    pub name: String,
    pub desc: ImportDesc,
}

#[derive(Debug, Clone)]
pub enum ImportDesc {
    Func(TypeId),
    Table(TableType),
    Memory(MemoryType),
    Global(GlobalType),
}

impl ImportDesc {
    pub fn kind(&self) -> ImportKind {
        match self {
            Self::Func(_) => ImportKind::Func,
            Self::Table(_) => ImportKind::Table,
            Self::Memory(_) => ImportKind::Memory,
            Self::Global(_) => ImportKind::Global,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImportKind {
    Func,
    Table,
    Memory,
    Global,
}

impl Display for ImportKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Func => write!(f, "function"),
            Self::Table => write!(f, "table"),
            Self::Memory => write!(f, "memory"),
            Self::Global => write!(f, "global"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Export {
    pub name: String,
    pub def: ExportDef,
}

#[derive(Debug, Clone)]
pub enum ExportDef {
    Func(FuncId),
    Table(TableId),
    Memory(MemoryId),
    Global(GlobalId),
}
