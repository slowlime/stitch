use std::collections::HashMap;
use std::sync::OnceLock;

use crate::ast;

macro_rules! define_primitives {
    {
        $( #[ $attr:meta ] )*
        $vis:vis enum Primitive {
            $( $variant:ident = ($class_name:literal, $method_name:literal) ),*
            $(,)?
        }
    } => {
        $( #[ $attr ] )*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        $vis enum Primitive {
            $( $variant, )*
        }

        impl Primitive {
            pub fn from_name(class_name: &str, method_name: &str) -> Option<Primitive> {
                static PRIMITIVES: OnceLock<HashMap<(&str, &str), Primitive>> = OnceLock::new();

                PRIMITIVES.get_or_init(|| HashMap::from([
                    $( (($class_name, $method_name), Self::$variant), )*
                ])).get(&(class_name, method_name)).copied()
            }

            pub fn from_selector(class_name: &str, selector: &ast::Selector) -> Option<Primitive> {
                Self::from_name(class_name, &selector.to_string())
            }

            pub fn as_selector(&self) -> &'static ast::Selector {
                static SELECTORS: OnceLock<Box<[ast::Selector]>> = OnceLock::new();

                &SELECTORS.get_or_init(|| vec![
                    $( ast::Selector::from_string($method_name.to_owned()), )*
                ].into())[*self as usize]
            }

            pub fn param_count(&self) -> usize {
                self.as_selector().param_count()
            }
        }
    };
}

define_primitives! {
    pub enum Primitive {
        ArrayAt = ("Array", "at:"),
        ArrayAtPut = ("Array", "at:put:"),
        ArrayLength = ("Array", "length"),
        ArrayNew = ("Array", "new"),

        BlockValue = ("Block", "value"),
        BlockRestart = ("Block", "restart"),

        Block1Value = ("Block1", "value"),

        Block2Value = ("Block2", "value:"),

        Block3ValueWith = ("Block3", "value:with:"),

        ClassName = ("Class", "name"),
        ClassNew = ("Class", "new"),
        ClassSuperclass = ("Class", "superclass"),
        ClassFields = ("Class", "fields"),
        ClassMethods = ("Class", "methods"),

        DoubleAdd = ("Double", "+"),
        DoubleSub = ("Double", "-"),
        DoubleMul = ("Double", "*"),
        DoubleDiv = ("Double", "//"),
        DoubleMod = ("Double", "%"),
        DoubleSqrt = ("Double", "sqrt"),
        DoubleRound = ("Double", "round"),
        DoubleAsInteger = ("Double", "asInteger"),
        DoubleCos = ("Double", "cos"),
        DoubleSin = ("Double", "sin"),
        DoubleEq = ("Double", "="),
        DoubleLt = ("Double", "<"),
        DoubleAsString = ("Double", "asString"),
        DoublePositiveInfinity = ("Double", "PositiveInfinity"),
        DoubleFromString = ("Double", "fromString"),

        MethodSignature = ("Method", "signature"),
        MethodHolder = ("Method", "holder"),
        MethodInvokeOnWith = ("Method", "invokeOn:with:"),

        PrimitiveSignature = ("Primitive", "signature"),
        PrimitiveHolder = ("Primitive", "holder"),
        PrimitiveInvokeOnWith = ("Primtive", "invokeOn:with:"),

        SymbolAsString = ("Symbol", "asString"),

        IntegerAdd = ("Integer", "+"),
        IntegerSub = ("Integer", "-"),
        IntegerMul = ("Integer", "*"),
        IntegerDiv = ("Integer", "/"),
        IntegerFDiv = ("Integer", "//"),
        IntegerMod = ("Integer", "%"),
        IntegerBand = ("Integer", "&"),
        IntegerShl = ("Integer", "<<"),
        IntegerShr = ("Integer", ">>>"),
        IntegerBxor = ("Integer", "bitXor:"),
        IntegerSqrt = ("Integer", "sqrt"),
        IntegerAtRandom = ("Integer", "atRandom"),
        IntegerEq = ("Integer", "="),
        IntegerLt = ("Integer", "<"),
        IntegerAsString = ("Integer", "asString"),
        IntegerAs32BitSignedValue = ("Integer", "as32BitSignedValue"),
        IntegerAs32BitUnsignedValue = ("Integer", "as32BitUnsignedValue"),
        IntegerAsDouble = ("Integer", "asDouble"),
        IntegerFromString = ("Integer", "fromString"),

        ObjectClass = ("Object", "class"),
        ObjectObjectSize = ("Object", "objectSize"),
        ObjectRefEq = ("Object", "=="),
        ObjectHashcode = ("Object", "hashcode"),
        ObjectInspect = ("Object", "inspect"),
        ObjectHalt = ("Object", "halt"),
        ObjectPerform = ("Object", "perform:"),
        ObjectPerformWithArguments = ("Object", "perform:withArguments:"),
        ObjectPerformInSuperclass = ("Object", "perform:inSuperclass:"),
        ObjectPerformWithArgumentsInSuperclass = ("Object", "perform:withArguments:inSuperclass:"),
        ObjectInstVarAt = ("Object", "instVarAt:"),
        ObjectInstVarAtPut = ("Object", "instVarAt:put:"),
        ObjectInstVarNamed = ("Object", "instVarNamed:"),

        StringConcatenate = ("String", "concatenate:"),
        StringAsSymbol = ("String", "asSymbol"),
        StringHashcode = ("String", "hashcode"),
        StringLength = ("String", "length"),
        StringIsWhitespace = ("String", "isWhitespace"),
        StringIsLetters = ("String", "isLetters"),
        StringIsDigits = ("String", "isDigits"),
        StringEq = ("String", "="),
        StringPrimSubstringFromTo = ("String", "primSubstringFrom:to:"),

        SystemGlobal = ("System", "global:"),
        SystemGlobalPut = ("System", "global:put:"),
        SystemHasGlobal = ("System", "hasGlobal:"),
        SystemLoadFile = ("System", "loadFile:"),
        SystemLoad = ("System", "load:"),
        SystemExit = ("System", "exit:"),
        SystemPrintString = ("System", "printString:"),
        SystemPrintNewline = ("System", "printNewline:"),
        SystemErrorPrintln = ("System", "errorPrintln:"),
        SystemErrorPrint = ("System", "errorPrint:"),
        SystemPrintStackTrace = ("System", "printStackTrace"),
        SystemTime = ("System", "time"),
        SystemTicks = ("System", "ticks"),
        SystemFullGC = ("System", "fullGC"),
    }
}

#[derive(Debug, Clone)]
pub enum MethodDef {
    Code(ast::Block),

    Primitive {
        primitive: Primitive,
        params: Vec<ast::Name>,
    },
}
