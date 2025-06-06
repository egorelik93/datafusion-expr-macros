extern crate proc_macro;

use std::{iter::Iterator, rc::Rc};

use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{
    parse_macro_input, parse_quote, BinOp, Expr, ExprBinary, ExprClosure, ExprField, ExprPath, Ident, Lit, LitBool, Member, Pat, Path, PathArguments, ReturnType, UnOp
};

static QUERY_EXPR_NAME: &'static str = "query_expr";

fn panic_attributes() -> ! {
    panic!("{} does not support attributes", QUERY_EXPR_NAME)
}

#[proc_macro_attribute]
pub fn query_fn(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as syn::ItemFn);

    if input.sig.variadic.is_some() {
        panic!("Variadics are not allowed");
    }

    let mut translator = Translator {
        row_arg: None,
        context: vec![]
    };

    let block = translator.translate_block(&input.block);

    let fnargs = input.sig.inputs.iter().map(|a| {
        let syn::FnArg::Typed(pat) = a
        else { panic!("self arguments are not allowed here") };

        let mut pat = pat.clone();
        pat.ty = Box::new( parse_quote! { ::datafusion_expr::Expr } );

        syn::FnArg::Typed(pat)
    }).collect();

    input.sig.inputs = fnargs;
    input.sig.output = syn::ReturnType::Type(Default::default(), Box::new(parse_quote! { ::datafusion_expr::Expr }));

    input.block = Box::new(parse_quote! {
        { #block }
    });

    input.into_token_stream().into()
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
    let row_arg = get_ident_from_pattern(&row_arg);

    let mut translator = Translator { row_arg: Some(&row_arg), context: vec!() };

    translator.translate_expr(&input.body).into_token_stream().into()
}

fn get_ident_from_pattern(pat: &Pat) -> &Ident {
    let pat = match pat {
        Pat::Ident(ident) => ident,
        _ => panic!("{} does not support patterns", QUERY_EXPR_NAME)
    };

    if pat.by_ref.is_some() {
        panic!("{} does not support ref specifiers", QUERY_EXPR_NAME);
    }

    if pat.mutability.is_some() {
        panic!("{} does not support mut specifiers", QUERY_EXPR_NAME);
    }

    if pat.subpat.is_some() {
        panic!("{} does not support subpatters", QUERY_EXPR_NAME);
    }

    &pat.ident
}

struct Translator<'a> {
    row_arg: Option<&'a Ident>,
    context: Vec<Vec<(Ident, LetValue)>>
}

impl Translator<'_> {
    fn translate_block(&mut self, block: &syn::Block) -> Expr {
        self.context.push(vec!());

        let mut expr = None;
        for stmt in &block.stmts {
            if let Some(_) = expr {
                panic!("More than one expression cannot be returned from a block");
            }

            expr = self.translate_stmt(&stmt)
        }

        self.context.pop();
        expr.expect("No expression was returned from the block")
    }

    fn translate_stmt(&mut self, stmt: &syn::Stmt) -> Option<Expr> {
        match stmt {
            syn::Stmt::Local(local) =>   {
                self.translate_let(local);
                None
            },
            syn::Stmt::Macro(_) => panic!("Macros are not allowed"),
            syn::Stmt::Expr(e, semicolon) => {
                if semicolon.is_some() {
                    panic!("Statement expressions are not allowed");
                }

                Some(self.translate_expr(e))
            },
            syn::Stmt::Item(item) => {
                match item {
                    syn::Item::Const(item) => {
                        if item.attrs.len() != 0 {
                            panic_attributes();
                        }

                        if let syn::Visibility::Inherited = item.vis {}
                        else { panic!("Visibility specifiers not allowed") }

                        if !item.generics.params.is_empty() || item.generics.where_clause.is_some() {
                            panic!("Generics not allowed");
                        }

                        let mut fresh_translator = Translator {
                            row_arg: None,
                            context: vec!(vec!())
                        };

                        fresh_translator.translate_let(&syn::Local {
                            attrs: vec!(),
                            let_token: Default::default(),
                            pat: Pat::Ident(syn::PatIdent {
                                attrs: vec!(),
                                by_ref: None,
                                mutability: None,
                                ident: item.ident.clone(),
                                subpat: None }),
                            init: Some(syn::LocalInit {
                                eq_token: Default::default(),
                                expr: item.expr.clone(),
                                diverge: None }),
                            semi_token: item.semi_token
                        });

                        let last_ctx = self.context.last_mut().unwrap();
                        last_ctx.append(fresh_translator.context.last_mut().unwrap());
                    },
                    _ => panic!("Statement item was not allowed")
                }

                None
            }
        }
    }

    fn translate_let(&mut self, local: &syn::Local) {
        if local.attrs.len() != 0 {
            panic_attributes();
        }

        let ident = get_ident_from_pattern(&local.pat);

        if Some(ident) == self.row_arg {
            panic!("Do not shadow row argument");
        }

        let body = local.init.as_ref().expect("Uninitialized let statements are not allowed");
        if body.diverge.is_some() {
            panic!("let-else statements are not allowed");
        }

        let body = self.translate_expr(&body.expr);

        let last_ctx = self.context.last_mut().unwrap();
        last_ctx.push((ident.clone(), LetValue::Value(body)));
    }

    fn translate_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::Lit(lit) => self.translate_literal(&lit),
            Expr::Field(field) => self.translate_field(&field),
            Expr::Unary(unary) => self.translate_unary_expr(&unary),
            Expr::Binary(binary) => self.translate_binary_expr(&binary),
            Expr::Path(path) => self.translate_path_expr(&path),
            Expr::Call(call) => self.translate_call(&call),
            Expr::Paren(paren) => {
                if paren.attrs.len() != 0 {
                    panic_attributes();
                }

                let expr = self.translate_expr(&paren.expr);
                parse_quote! { (#expr) }
            },
            Expr::Group(group) => {
                if group.attrs.len() != 0 {
                    panic_attributes();
                }

                let expr = self.translate_expr(&group.expr);
                parse_quote! { (#expr) }
            }
            Expr::Block(block) => {
                if block.attrs.len() != 0 {
                    panic_attributes();
                }

                if block.label.is_some() {
                    panic!("Labels are not supported");
                }

                self.translate_block(&block.block)
            },
            _ => panic!("Unsupported expr {}", expr.to_token_stream()),
        }
    }

    fn translate_call(&mut self, call: &syn::ExprCall) -> Expr {
        if call.attrs.len() != 0 {
            panic_attributes();
        }

        let func = match &*call.func {
            Expr::Path(path) =>  self.translate_path_fn(path),
            _ => panic!("A function was expected here")
        };

        let args: Vec<Expr> = call.args.iter().map(|e| self.translate_expr(e)).collect();
        func(&args)
    }

    fn translate_path_expr(&mut self, path: &syn::ExprPath) -> Expr {
        let value = self.translate_path(path);
        let Some(value) = value
        else  {
            if path.path.get_ident().is_none() {
                panic!("Only variable names are allowed here");
            }

            return parse_quote! { (#path) };
        };

        let LetValue::Value(expr) = value
        else { panic!("A function was not expected here") };

        expr
    }

    fn translate_path_fn(&mut self, path: &syn::ExprPath) -> Rc<dyn Fn(&[Expr]) -> Expr> {
        let value = self.translate_path(path);
        let path = match value {
            Some(LetValue::Value(_)) => panic!("A function was expected here"),
            Some(LetValue::Fn(f)) => return f,
            None => path.path.clone()
        };

        Rc::new(move |args: &[Expr]| {
            let n = args.len();
            let path = &path;

            parse_quote! { ::datafusion_expr_macros::ExprFn::<#n>::call_for_expr(
                &(::datafusion_expr_macros::ExprFnDispatcher::new(#path)),
                &[#(#args),*]) }
        })
    }

    fn translate_path(&mut self, path: &syn::ExprPath) -> Option<LetValue> {
        if path.attrs.len() != 0 {
            panic_attributes();
        }

        if path.qself.is_some() {
            panic!("self is not allowed here");
        }

        let Some(ident) = path.path.get_ident()
        else { return None };

        for ctx in self.context.iter().rev() {
            for pair in ctx.iter().rev() {
                if &pair.0 == ident {
                    return Some(pair.1.clone());
                }
            }
        }

        None
    }

    fn translate_field(&mut self, field: &ExprField) -> Expr {
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

    fn translate_unary_expr(&mut self, unary: &syn::ExprUnary) -> Expr {
        if unary.attrs.len() != 0 {
            panic_attributes();
        }

        let expr = self.translate_expr(&unary.expr);

        match unary.op {
            UnOp::Neg(_) => parse_quote! { -(#expr) },
            UnOp::Not(_) => parse_quote! { !(#expr) },
            _ => panic!("Operator '{}' is not supported", unary.op.to_token_stream())
        }
    }

    fn translate_binary_expr(&mut self, binary: &syn::ExprBinary) -> Expr {
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

    fn translate_binary_cmp_expr(&mut self, binary: &syn::ExprBinary) -> Expr {
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
            panic!("Expected the row variable {}", self.row_arg.unwrap());
        }

        let ident = match path.path.require_ident() {
            Ok(i) => i,
            _ => panic!("Expected the row variable {}", self.row_arg.unwrap()),
        };

        if Some(ident) != self.row_arg {
            panic!("Expected the row variable {}", self.row_arg.unwrap());
        }
    }

    fn translate_literal(&mut self, lit: &syn::ExprLit) -> Expr {
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


#[derive(Clone)]
enum LetValue {
    Value(Expr),
    Fn(Rc<dyn Fn(&[Expr]) -> Expr>)
}
