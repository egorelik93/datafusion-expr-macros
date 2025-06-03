use datafusion_expr_macros::query_expr;
use datafusion::prelude::*;

#[test]
fn test_simple() {
    let e = query_expr!(|r| r.a + 3 == 5);
    println!("{}", e.human_display());
}
