use std::fmt::Display;
use std::process::ExitCode;
use std::str;

use stitch_bindings::{arg_count, arg_len, arg_read, inline, print_str, specialize};

#[derive(Debug, Clone, Copy)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone)]
enum Expr {
    Int(i64),
    Neg(Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
}

impl Expr {
    extern "C" fn eval(&self) -> Result<i64, String> {
        let recurse = inline!(fn(&Expr) -> Result<i64, String>)(Self::eval);

        Ok(match self {
            &Self::Int(value) => value,

            Self::Neg(expr) => {
                let value = recurse(expr)?;

                value
                    .checked_neg()
                    .ok_or_else(|| format!("negating {value} resulted in an overflow"))?
            }

            Self::Binary(op, lhs, rhs) => {
                let lhs = recurse(lhs)?;
                let rhs = recurse(rhs)?;

                match *op {
                    BinOp::Add => lhs
                        .checked_add(rhs)
                        .ok_or_else(|| format!("adding {lhs} to {rhs} resulted in an overflow"))?,

                    BinOp::Sub => lhs.checked_sub(rhs).ok_or_else(|| {
                        format!("subtracting {lhs} from {rhs} resulted in an overflow")
                    })?,

                    BinOp::Mul => lhs.checked_mul(rhs).ok_or_else(|| {
                        format!("multiplying {lhs} by {rhs} resulted in an overflow")
                    })?,

                    BinOp::Div => lhs.checked_div(rhs).ok_or_else(|| {
                        format!("dividing {lhs} by {rhs} resulted in an overflow")
                    })?,
                }
            }
        })
    }
}

struct Parser<'a> {
    buf: &'a str,
    pos: usize,
    char_idx: usize,
}

impl<'a> Parser<'a> {
    fn new(buf: &'a str) -> Self {
        Self {
            buf,
            pos: 0,
            char_idx: 1,
        }
    }

    fn parse(mut self) -> Result<Expr, String> {
        let result = self.parse_expr()?;
        self.consume_space();

        if self.peek().is_some() {
            return Err(self.error("expected eof"));
        }

        Ok(result)
    }

    fn error(&self, msg: impl Display) -> String {
        format!("error at {} (byte {}): {msg}", self.char_idx, self.pos)
    }

    fn peek(&self) -> Option<char> {
        self.buf[self.pos..].chars().next()
    }

    fn next(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += c.len_utf8();
        self.char_idx += 1;

        Some(c)
    }

    fn consume_if(&mut self, f: impl FnOnce(char) -> bool) -> Option<char> {
        self.peek().filter(|&c| f(c))
    }

    fn consume(&mut self, c: char) -> bool {
        self.consume_if(|head| head == c).is_some()
    }

    fn consume_while(&mut self, mut f: impl FnMut(char) -> bool) -> &'a str {
        let start_pos = self.pos;

        while let Some(c) = self.peek() {
            if !f(c) {
                break;
            }

            self.next().unwrap();
        }

        &self.buf[start_pos..self.pos]
    }

    fn consume_space(&mut self) {
        self.consume_while(|c| c.is_ascii_whitespace());
    }

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_add_sub()
    }

    fn parse_add_sub(&mut self) -> Result<Expr, String> {
        let mut result = self.parse_mul_div()?;

        while let Some(c) = {
            self.consume_space();
            self.consume_if(|c| c == '+' || c == '-')
        } {
            let rhs = self.parse_mul_div()?;
            result = Expr::Binary(
                if c == '+' { BinOp::Add } else { BinOp::Sub },
                Box::new(result),
                Box::new(rhs),
            );
        }

        Ok(result)
    }

    fn parse_mul_div(&mut self) -> Result<Expr, String> {
        let mut result = self.parse_unary()?;

        while let Some(c) = {
            self.consume_space();
            self.consume_if(|c| c == '*' || c == '/')
        } {
            let rhs = self.parse_unary()?;
            result = Expr::Binary(
                if c == '*' { BinOp::Mul } else { BinOp::Div },
                Box::new(result),
                Box::new(rhs),
            );
        }

        Ok(result)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        self.consume_space();

        if self.consume('-') {
            Ok(Expr::Neg(Box::new(self.parse_atom()?)))
        } else {
            self.parse_atom()
        }
    }

    fn parse_atom(&mut self) -> Result<Expr, String> {
        self.consume_space();

        if self.consume('(') {
            let result = self.parse_expr()?;
            self.consume_space();

            return if self.consume(')') {
                Ok(result)
            } else {
                Err(self.error("expected ')'"))
            };
        }

        if self.peek().is_some_and(|c| c.is_ascii_digit()) {
            return self.parse_int();
        }

        if let Some(c) = self.peek() {
            Err(self.error(format_args!("unexpected character: '{c}'")))
        } else {
            Err(self.error("unexpected eof"))
        }
    }

    fn parse_int(&mut self) -> Result<Expr, String> {
        let literal = self.consume_while(|c| c.is_ascii_digit());
        debug_assert!(!literal.is_empty());

        i64::from_str_radix(literal, 10)
            .map(Expr::Int)
            .map_err(|e| self.error(e))
    }
}

static mut EVAL: Option<extern "C" fn() -> Result<i64, String>> = None;

#[export_name = "stitch-start"]
pub fn stitch_start() {
    if arg_count() != 1 {
        print_str(&format!("expected 1 argument, got {}", arg_count()));
        print_str("usage: <expr>");
        return;
    }

    let mut buf = vec![0u8; arg_len(0)];
    arg_read(0, &mut buf, 0);

    let buf = match str::from_utf8(&buf) {
        Ok(buf) => buf,
        Err(e) => {
            print_str!("could not decode the argument: {e}");
            return;
        }
    };

    let expr = match Parser::new(&buf).parse() {
        Ok(expr) => expr,
        Err(e) => {
            print_str!("could not parse the expression: {e}");
            return;
        }
    };

    unsafe {
        EVAL = specialize!("eval": fn(expr: &Expr) -> Result<i64, String>)(Expr::eval, None, &expr);
    }
}

pub fn main() -> ExitCode {
    match unsafe { EVAL }.expect("the specialized function is not available")() {
        Ok(value) => {
            println!("evaluated to {value}");

            ExitCode::SUCCESS
        }

        Err(e) => {
            eprintln!("{e}");

            ExitCode::FAILURE
        }
    }
}
