extern crate datafusion_expr_macros_proc;

use std::{marker::PhantomData};
use std::sync::Arc;

use datafusion_expr::{expr, AggregateUDF, Expr, ScalarUDF, WindowUDF};
pub use datafusion_expr_macros_proc::{query_expr, query_fn};

pub struct ExprFnDispatcher<F, Arity: ?Sized = F> {
    f: F,
    phantom: PhantomData<Arity>
}

impl<F, Arity: ?Sized> ExprFnDispatcher<F, Arity> {
    pub fn new(f : F) -> Self {
        ExprFnDispatcher { f, phantom: PhantomData }
    }
}


pub trait ExprFn<const N: usize> {
    fn call_for_expr(&self, args: &[Expr; N]) -> Expr;
}

impl<const N: usize> ExprFn<N> for ExprFnDispatcher<Arc<ScalarUDF>> {
    fn call_for_expr(&self, args: &[Expr; N]) -> Expr {
        Expr::ScalarFunction(expr::ScalarFunction {
            func: self.f.clone(),
            args: args.to_vec(),
        })
    }
}

impl<const N: usize, F> ExprFn<N> for ExprFnDispatcher<F, dyn Fn() -> Arc<ScalarUDF>> where F: Fn() -> Arc<ScalarUDF>  {
    fn call_for_expr(&self, args: &[Expr; N]) -> Expr {
        Expr::ScalarFunction(expr::ScalarFunction {
            func: (self.f)(),
            args: args.to_vec(),
        })
    }
}

impl<const N: usize> ExprFn<N> for ExprFnDispatcher<Arc<AggregateUDF>> {
    fn call_for_expr(&self, args: &[Expr; N]) -> Expr {
        Expr::AggregateFunction(expr::AggregateFunction {
            func: self.f.clone(),
            params: expr::AggregateFunctionParams {
                args: args.to_vec(),
                distinct: false,
                filter: None,
                order_by: None,
                null_treatment: None,
            },
        })
    }
}

impl<const N: usize, F> ExprFn<N> for ExprFnDispatcher<F, dyn Fn() -> Arc<AggregateUDF>> where F: Fn() -> Arc<AggregateUDF> {
    fn call_for_expr(&self, args: &[Expr; N]) -> Expr {
        Expr::AggregateFunction(expr::AggregateFunction {
            func: (self.f)(),
            params: expr::AggregateFunctionParams {
                args: args.to_vec(),
                distinct: false,
                filter: None,
                order_by: None,
                null_treatment: None,
            },
        })
    }
}

macro_rules! impl_expr_fn {
    ($n:literal $(,)? $($arg:ty),*) => {
        impl<F> ExprFn<$n> for ExprFnDispatcher<F, dyn Fn($($arg),*) -> Expr> where F: Fn($($arg),*) -> Expr
        {
            fn call_for_expr(&self, _args: &[Expr; $n]) -> Expr {
                let mut _i = 0;
                (self.f)(
                     $(
                         {
                             let arg: $arg = _args[_i].clone();
                             _i += 1;
                             arg
                         }
                     ),*
                )
            }
        }
    };
}

impl<const N: usize, F> ExprFn<N> for ExprFnDispatcher<F, dyn Fn(Vec<Expr>) -> Expr> where F: Fn(Vec<Expr>) -> Expr
{
    fn call_for_expr(&self, args: &[Expr; N]) -> Expr {
        (self.f)(args.to_vec())
    }
}


impl_expr_fn! { 0 }
impl_expr_fn! { 1, Expr }
impl_expr_fn! { 2, Expr, Expr }
impl_expr_fn! { 3, Expr, Expr, Expr }
impl_expr_fn! { 4, Expr, Expr, Expr, Expr }
impl_expr_fn! { 5, Expr, Expr, Expr, Expr, Expr }
impl_expr_fn! { 6, Expr, Expr, Expr, Expr, Expr, Expr }
impl_expr_fn! { 7, Expr, Expr, Expr, Expr, Expr, Expr, Expr }
impl_expr_fn! { 8, Expr, Expr, Expr, Expr, Expr, Expr, Expr, Expr }
impl_expr_fn! { 9, Expr, Expr, Expr, Expr, Expr, Expr, Expr, Expr, Expr }
