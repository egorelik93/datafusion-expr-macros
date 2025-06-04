use datafusion;
use datafusion_expr_macros::query_expr;

#[test]
fn fn_call() {
    let f = datafusion::functions::core::expr_fn::nullif;
    let g = datafusion::functions::core::least();

    use datafusion::functions::core::greatest;

    let e = query_expr!(|r|
      greatest(g(f(r.a, r.c))));
    println!("{}", e.human_display());
}
