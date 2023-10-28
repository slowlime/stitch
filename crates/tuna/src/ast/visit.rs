use crate::ast;

pub trait AstRecurse {
    fn recurse<'a, V: Visitor<'a>>(&'a self, visitor: &mut V);
    fn recurse_mut<'a, V: VisitorMut<'a>>(&'a mut self, visitor: &mut V);
}

macro_rules! define_visitor {
    ($( $type:ident { $( $name:ident ( $arg:ident : $ty:ty ) );+ $(;)? } )+) => {
        pub trait Visitor<'a>
        where
            Self: Sized,
        {
            $(
                $(
                    fn $name(&mut self, $arg: &'a $ty);
                )+
            )+
        }

        pub trait VisitorMut<'a>
        where
            Self: Sized,
        {
            $(
                $(
                    fn $name(&mut self, $arg: &'a mut $ty);
                )+
            )+
        }

        pub trait DefaultVisitor<'a>
        where
            Self: Sized,
        {
            $( define_visitor!(@ $type { $( $name ( $arg : &'a $ty ) => recurse; )+ } ); )+
        }

        impl<'a, T> Visitor<'a> for T
        where
            T: DefaultVisitor<'a>,
        {
            $(
                $(
                    fn $name(&mut self, $arg: &'a $ty) {
                        <Self as DefaultVisitor>::$name(self, $arg);
                    }
                )+
            )+
        }

        pub trait DefaultVisitorMut<'a>
        where
            Self: Sized,
        {
            $( define_visitor!(@ $type { $( $name ( $arg : &'a mut $ty ) => recurse_mut; )+ } ); )+
        }

        impl<'a, T> VisitorMut<'a> for T
        where
            T: DefaultVisitorMut<'a>,
        {
            $(
                $(
                    fn $name(&mut self, $arg: &'a  mut $ty) {
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
