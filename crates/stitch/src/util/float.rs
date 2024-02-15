use std::fmt::{self, Debug, Display};
use std::ops::Neg;

macro_rules! declare_float {
    ($name:ident, $float:ident, $bits:ty, $from_cvt:ident, $to_cvt:ident) => {
        macro_rules! impl_arith {
            ($trait:ident::$method:ident as ($op:tt=)) => {
                impl ::std::ops::$trait for $name {
                    fn $method(&mut self, rhs: Self) {
                        *self = (self.$to_cvt() $op rhs.$to_cvt()).into();
                    }
                }
            };

            ($trait:ident::$method:ident as ($op:tt)) => {
                impl ::std::ops::$trait for $name {
                    type Output = Self;

                    fn $method(self, rhs: Self) -> Self::Output {
                        (self.$to_cvt() $op rhs.$to_cvt()).into()
                    }
                }
            };
        }

        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name($bits);

        impl $name {
            pub fn from_bits(bits: $bits) -> Self {
                Self(bits)
            }

            pub fn $from_cvt(value: $float) -> Self {
                Self::from_bits(value.to_bits())
            }

            pub fn to_bits(self) -> $bits {
                self.0
            }

            pub fn $to_cvt(self) -> $float {
                $float::from_bits(self.0)
            }

            pub fn abs(self) -> Self {
                self.$to_cvt().abs().into()
            }

            pub fn sqrt(self) -> Self {
                self.$to_cvt().sqrt().into()
            }

            pub fn ceil(self) -> Self {
                self.$to_cvt().ceil().into()
            }

            pub fn floor(self) -> Self {
                self.$to_cvt().floor().into()
            }

            pub fn trunc(self) -> Self {
                self.$to_cvt().trunc().into()
            }

            pub fn nearest(self) -> Self {
                let f = self.$to_cvt();

                if f.fract() == 0.5 && f.trunc() % 2.0 == 0.0 {
                    f.floor().into()
                } else {
                    f.round().into()
                }
            }

            pub fn min(self, other: Self) -> Self {
                self.$to_cvt().min(other.$to_cvt()).into()
            }

            pub fn max(self, other: Self) -> Self {
                self.$to_cvt().max(other.$to_cvt()).into()
            }

            pub fn copysign(self, other: Self) -> Self {
                self.$to_cvt().copysign(other.$to_cvt()).into()
            }

            pub fn trunc_i32(self) -> i32 {
                self.$to_cvt() as i32
            }

            pub fn trunc_u32(self) -> u32 {
                self.$to_cvt() as u32
            }

            pub fn trunc_i64(self) -> i64 {
                self.$to_cvt() as i64
            }

            pub fn trunc_u64(self) -> u64 {
                self.$to_cvt() as u64
            }
        }

        impl Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{:?}", self.$to_cvt())
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.$to_cvt())
            }
        }

        impl Neg for $name {
            type Output = Self;

            fn neg(self) -> Self {
                self.$to_cvt().neg().into()
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::$from_cvt(Default::default())
            }
        }

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.$to_cvt().partial_cmp(&other.$to_cvt())
            }
        }

        impl_arith!(Add::add as (+));
        impl_arith!(AddAssign::add_assign as (+ =));
        impl_arith!(Sub::sub as (-));
        impl_arith!(SubAssign::sub_assign as (- =));
        impl_arith!(Mul::mul as (*));
        impl_arith!(MulAssign::mul_assign as (* =));
        impl_arith!(Div::div as (/));
        impl_arith!(DivAssign::div_assign as (/ =));
        impl_arith!(Rem::rem as (%));
        impl_arith!(RemAssign::rem_assign as (% =));

        impl From<$float> for $name {
            fn from(value: $float) -> Self {
                Self::$from_cvt(value)
            }
        }
    };
}

declare_float!(F32, f32, u32, from_f32, to_f32);
declare_float!(F64, f64, u64, from_f64, to_f64);

impl F32 {
    pub fn promote(self) -> F64 {
        (self.to_f32() as f64).into()
    }
}

impl F64 {
    pub fn demote(self) -> F32 {
        (self.to_f64() as f32).into()
    }
}
