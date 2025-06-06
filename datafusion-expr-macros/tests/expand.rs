use datafusion;
use datafusion_expr_macros::{query_expr, query_fn};

#[test]
fn fn_call() {
    let f = datafusion::functions::core::expr_fn::nullif;
    let g = datafusion::functions::core::least();

    use datafusion::functions::core::greatest;

    let e = query_expr!(|r|
      greatest(g(f(r.a, r.c))));
    assert_eq!(format!("{}", e.human_display()), "greatest(least(nullif(a, c)))");
}

use datafusion::functions::core::nullif;

#[query_fn]
fn example_query_fn(i: i32, j: i32) -> i32 {
    nullif(i + j, true)
}

#[test]
fn test_query_fn() {
    let e = query_expr!(|r|
        example_query_fn(r.a, r.b));
    assert_eq!(format!("{}", e.human_display()), "nullif(a + b, Boolean(true))");
}
