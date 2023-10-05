use crate::ast;

pub trait AstRecurse {
    fn recurse<V: Visitor>(&self, visitor: &mut V);
    fn recurse_mut<V: VisitorMut>(&mut self, visitor: &mut V);
}

macro_rules! define_visitor {
    ($( $type:ident { $( $name:ident ( $arg:ident : $ty:ty ) );+ $(;)? } )+) => {
        pub trait Visitor
        where
            Self: Sized,
        {
            $(
                $(
                    fn $name(&mut self, $arg: &$ty);
                )+
            )+
        }

        pub trait VisitorMut
        where
            Self: Sized,
        {
            $(
                $(
                    fn $name(&mut self, $arg: &mut $ty);
                )+
            )+
        }

        pub trait DefaultVisitor
        where
            Self: Sized,
        {
            $( define_visitor!(@ $type { $( $name ( $arg : &$ty ) => recurse; )+ } ); )+
        }

        impl<T> Visitor for T
        where
            T: DefaultVisitor,
        {
            $(
                $(
                    fn $name(&mut self, $arg: &$ty) {
                        <Self as DefaultVisitor>::$name(self, $arg);
                    }
                )+
            )+
        }

        pub trait DefaultVisitorMut
        where
            Self: Sized,
        {
            $( define_visitor!(@ $type { $( $name ( $arg : &mut $ty ) => recurse_mut; )+ } ); )+
        }

        impl<T> VisitorMut for T
        where
            T: DefaultVisitorMut,
        {
            $(
                $(
                    fn $name(&mut self, $arg: &mut $ty) {
                        <Self as DefaultVisitorMut>::$name(self, $arg);
                    }
                )+
            )+
        }
    };

    (@ NonTerminal { $( $name:ident ( $arg:ident : $ty:ty ) => $recurse:ident; )+ }) => {
        $(
            fn $name(&mut self, $arg: $ty) {
                $arg.$recurse(self);
            }
        )+
    };

    (@ Terminal { $( $name:ident ( $arg:ident : $ty:ty ) => $recurse:ident; )+ }) => {
        $(
            #[allow(unused_variables)]
            fn $name(&mut self, $arg: $ty) {}
        )+
    };
}

define_visitor! {
    NonTerminal {
        visit_class(class: ast::Class);
        visit_method(method: ast::Method);
        visit_block(block: ast::Block);

        visit_stmt(stmt: ast::Stmt);
        visit_expr(expr: ast::Expr);

        visit_assign(expr: ast::Assign);
        visit_array(expr: ast::ArrayLit);
        visit_dispatch(expr: ast::Dispatch);
    }

    Terminal {
        visit_unresolved_name(name: ast::UnresolvedName);
        visit_local(local: ast::Local);
        visit_upvalue(upvalue: ast::Upvalue);
        visit_field(field: ast::Field);
        visit_global(global: ast::Global);

        visit_symbol(expr: ast::SymbolLit);
        visit_string(expr: ast::StringLit);
        visit_int(expr: ast::IntLit);
        visit_float(expr: ast::FloatLit);
    }
}
