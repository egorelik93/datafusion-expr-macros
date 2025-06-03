extern crate proc_macro;

use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input, parse_quote, BinOp, Expr, ExprBinary, ExprClosure, ExprField, ExprPath, Ident, Lit, LitBool, Member, Pat, PathArguments, ReturnType
};

static QUERY_EXPR_NAME: &'static str = "query_expr";

fn panic_attributes() -> ! {
    panic!("{} does not support attributes", QUERY_EXPR_NAME)
}

#[proc_macro]
pub fn query_expr(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as ExprClosure);

    if input.attrs.len() != 0 {
        panic_attributes();
    }

    if input.lifetimes.is_some() {
        panic!("{} does not support lifetimes", QUERY_EXPR_NAME);
    }

    if input.constness.is_some() {
        panic!("{} does not support const", QUERY_EXPR_NAME);
    }

    if input.movability.is_some() {
        panic!("{} does not support static", QUERY_EXPR_NAME);
    }

    if input.asyncness.is_some() {
        panic!("{} does not support asyncness", QUERY_EXPR_NAME);
    }

    if input.capture.is_some() {
        panic!("{} does not support move", QUERY_EXPR_NAME);
    }

    match input.output {
        ReturnType::Default => {}
        _ => panic!("{} does not support explicit output types", QUERY_EXPR_NAME),
    }

    let args = input.inputs;
    if args.len() != 1 {
        panic!(
            "{} only supports closures with a single argument",
            QUERY_EXPR_NAME
        );
    }

    let row_arg = args.first().unwrap();
    let row_arg = match row_arg {
        Pat::Ident(ident) => ident,
        _ => panic!(
            "{} does not support patterns in the argument",
            QUERY_EXPR_NAME
        ),
    };

    if row_arg.attrs.len() != 0 {
        panic_attributes();
    }

    if row_arg.by_ref.is_some() {
        panic!("{} does not support ref specifiers", QUERY_EXPR_NAME);
    }

    if row_arg.mutability.is_some() {
        panic!("{} does not support mut specifiers", QUERY_EXPR_NAME);
    }

    if row_arg.subpat.is_some() {
        panic!("{} does not support subpatters", QUERY_EXPR_NAME);
    }

    let row_arg = &row_arg.ident;
    let translator = RowExprTranslator { row_arg: &row_arg };

    translator.translate_expr(&input.body).into_token_stream().into()
}

struct RowExprTranslator<'a> {
    row_arg: &'a Ident,
}

impl RowExprTranslator<'_> {
    fn translate_expr(&self, expr: &Expr) -> Expr {
        match expr {
            Expr::Lit(lit) => self.translate_literal(&lit),
            Expr::Field(field) => self.translate_field(&field),
            Expr::Binary(binary) => self.translate_binary_expr(&binary),
            _ => panic!("Unsupported expr"),
        }
    }

    fn translate_field(&self, field: &ExprField) -> Expr {
        if field.attrs.len() != 0 {
            panic!("{} does not support attributes", QUERY_EXPR_NAME);
        }

        match &*field.base {
            Expr::Path(p) => self.is_row_variable(&p),
            _ => panic!("Not a row field access"),
        }

        match &field.member {
            Member::Named(field_name) => parse_quote! {
                datafusion::prelude::col(stringify!(#field_name))
            },
            _ => panic!("Tuple fields not supported"),
        }
    }

    // TODO: Waiting for translate_block
    /*
    fn translate_if(&self, if_expr: &syn::ExprIf) -> Expr {
        if if_expr.attrs.len() != 0 {
            panic_attributes();
        }

        let cond = self.translate_expr(&if_expr.cond);
        let then_branch = self.translate_expr(&if_expr.then_branch);
        let else_branch = self.translate_expr(&if_expr.else_branch);
    }
    */

    fn translate_binary_expr(&self, binary: &syn::ExprBinary) -> Expr {
        if binary.attrs.len() != 0 {
            panic_attributes();
        }

        let op = &binary.op;

        match op {
            BinOp::Eq(_) | BinOp::Ne(_) | BinOp::Lt(_) | BinOp::Le(_) | BinOp::Gt(_) | BinOp::Ge(_) => {
                return self.translate_binary_cmp_expr(binary);
            },
            _ => {}
        }

        let left = self.translate_expr(&*binary.left);
        let right = self.translate_expr(&*binary.right);

        let expr = match *op {
            BinOp::And(_) => parse_quote! { (#left).and(#right) },
            BinOp::Or(_) => parse_quote! { (#left).or(#right) },
            BinOp::Add(_) => parse_quote! { (#left) + (#right) },
            BinOp::Sub(_) => parse_quote! { (#left) - (#right) },
            BinOp::Mul(_) => parse_quote! { (#left) * (#right) },
            BinOp::Div(_) => parse_quote! { (#left) / (#right) },
            BinOp::Rem(_) => parse_quote! { (#left) % (#right) },
            BinOp::BitAnd(_) => parse_quote! { (#left) & (#right) },
            BinOp::BitOr(_) => parse_quote! { #(left) | (#right) },
            BinOp::BitXor(_) => parse_quote! { #(left) ^ #(right) },
            BinOp::Shl(_) => parse_quote! { #(left) << (#right) },
            BinOp::Shr(_) => parse_quote! { #(left) >> (#right) },
            _ => panic!("Operator '{}' is not supported", op.to_token_stream()),
        };

        return expr;
    }

    fn translate_binary_cmp_expr(&self, binary: &syn::ExprBinary) -> Expr {
        match (&*binary.left, &binary.op, &*binary.right) {
            (Expr::Lit(syn::ExprLit { attrs: attrs_l, lit: Lit::Bool(syn::LitBool { value: left_b, .. }) }),
            BinOp::Eq(_) | BinOp::Ne(_),
            Expr::Lit(syn::ExprLit { attrs: attrs_r, lit: Lit::Bool(syn::LitBool { value: right_b, .. }) })) => {

                if attrs_l.len() != 0 || attrs_r.len() != 0 {
                    panic_attributes();
                }

                let result = if let BinOp::Eq(_) = binary.op { left_b == right_b } else { left_b != right_b };
                return self.translate_literal(&parse_quote! { Lit::Bool(#result) })
            },
            (Expr::Lit(syn::ExprLit { attrs, lit: Lit::Bool(syn::LitBool { value, .. }) }),
            BinOp::Eq(_) | BinOp::Ne(_),
            expr) |
            (expr,
             BinOp::Eq(_) | BinOp::Ne(_),
             Expr::Lit(syn::ExprLit { attrs, lit: Lit::Bool(syn::LitBool { value, .. }) })) => {

                if attrs.len() != 0 {
                    panic_attributes();
                }

                let expr = self.translate_expr(expr);

                match &binary.op {
                    BinOp::Eq(_) => return match value {
                        true => parse_quote! { (#expr).is_true() },
                        false => parse_quote! { (#expr).is_false() }
                    },
                    // If ne, should we require that the expr is stricly bool, or treat true/false
                    // as being implicitly convertible to Option<bool>?
                    BinOp::Ne(_) => return match value {
                        true => parse_quote! { (#expr).is_not_true() },
                        false => parse_quote! { (#expr).is_not_false() }
                    },
                    _ => unreachable!()
                }
            },
            (Expr::Path(syn::ExprPath { attrs, qself, path }),
            BinOp::Eq(_) | BinOp::Ne(_),
            expr) |
            (expr,
             BinOp::Eq(_) | BinOp::Ne(_),
             Expr::Path(syn::ExprPath { attrs, qself, path }))
             if qself.is_none() && path.is_ident("None") => {

                if attrs.len() != 0 {
                    panic_attributes()
                }

                let expr = self.translate_expr(expr);

                return match &binary.op {
                    BinOp::Eq(_) => parse_quote! { (#expr).is_null() },
                    BinOp::Ne(_) => parse_quote! { (#expr).is_not_null() },
                    _ => unreachable!()
                }
            },
            _ => {}
        }


        let left = self.translate_expr(&binary.left);
        let right = self.translate_expr(&binary.right);

        let expr: Expr = match &binary.op {
            BinOp::Eq(_) => parse_quote! { (#left).eq(#right) },
            BinOp::Ne(_) => parse_quote! { (#left).not_eq(#right) },
            BinOp::Lt(_) => parse_quote! { (#left).lt(#right) },
            BinOp::Le(_) => parse_quote! { (#left).lt_eq(#right) },
            BinOp::Gt(_) => parse_quote! { (#left).gt(#right) },
            BinOp::Ge(_) => parse_quote! { (#left).gt_eq(#right) },
            _ => unreachable!()
        };

        parse_quote! { (#expr).is_true() }
    }

    fn is_row_variable(&self, path: &syn::ExprPath) {
        if path.attrs.len() != 0 {
            panic_attributes();
        }

        if path.qself.is_some() {
            panic!("Expected the row variable {}", self.row_arg);
        }

        let ident = match path.path.require_ident() {
            Ok(i) => i,
            _ => panic!("Expected the row variable {}", self.row_arg),
        };

        if ident != self.row_arg {
            panic!("Expected the row variable {}", self.row_arg);
        }
    }

    fn translate_literal(&self, lit: &syn::ExprLit) -> Expr {
        if lit.attrs.len() != 0 {
            panic_attributes();
        }

        match &lit.lit {
            Lit::Int(_) | Lit::Float(_) | Lit::Bool(_) | Lit::Str(_) => {
                parse_quote! { datafusion::prelude::lit(#lit) }
            }
            _ => panic!("Unsupported literal type in {}", QUERY_EXPR_NAME),
        }
    }

    fn is_lit(&self, e: &Expr, lit: Lit) -> bool {
        match e {
            Expr::Lit(l) => l.attrs.len() == 0 && l.lit == lit,
            _ => false
        }
    }
}
