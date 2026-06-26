//! Demonstrates `Control` — cancellation, timeouts, and type-safe extensions.
//!
//! Run with:
//! ```bash
//! cargo run --example control_basics
//! ```

use std::sync::Arc;
use std::time::Duration;

use behest::runtime::Control;

#[derive(Debug)]
struct AppContext {
    user_id: String,
    request_id: String,
}

fn main() {
    let ctrl = Control::new();

    assert!(!ctrl.is_cancelled(), "new control should not be cancelled");
    println!("Control created, cancelled = {}", ctrl.is_cancelled());

    ctrl.cancel();
    assert!(
        ctrl.is_cancelled(),
        "control should be cancelled after cancel()"
    );
    println!("After cancel(), cancelled = {}", ctrl.is_cancelled());

    let cloned = ctrl.clone();
    assert!(
        cloned.is_cancelled(),
        "cloned control inherits cancellation state"
    );
    println!("Cloned control cancelled = {}", cloned.is_cancelled());

    let ctrl2 = Control::new();

    ctrl2.set_timeout(Duration::from_secs(30));
    if let Some(t) = ctrl2.timeout() {
        println!("Timeout set: {}s", t.as_secs());
    }

    ctrl2.set_concurrency_limit(4);
    if let Some(cl) = ctrl2.concurrency_limit() {
        println!("Concurrency limit: {cl}");
    }

    ctrl2.set_data(AppContext {
        user_id: "user_abc".into(),
        request_id: "req_123".into(),
    });

    let ctx: Option<Arc<AppContext>> = ctrl2.data();
    if let Some(ctx) = ctx {
        println!(
            "Extensions: user_id={}, request_id={}",
            ctx.user_id, ctx.request_id
        );
    }
}
