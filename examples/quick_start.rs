//! Quick-start example for `blr-core`.
//!
//! This example fits a BLR+ARD model to a small synthetic dataset and prints
//! the posterior weights, active features, and prediction uncertainty.
//!
//! Run from the repository root:
//!
//! ```bash
//! cargo run --example quick_start -p blr-core
//! ```

use blr_core::{features, fit, ArdConfig};

fn main() {
    // ── 1. Generate a simple synthetic dataset ─────────────────────────────
    //
    // True model: y = 2.0 * x + 0.5 * x² + noise
    // We engineer 4 polynomial features [1, x, x², x³] so ARD should
    // discover that the cubic feature is irrelevant.

    let n = 30; // 30 calibration points
    let x: Vec<f64> = (0..n)
        .map(|i| -3.0 + 6.0 * (i as f64) / (n as f64 - 1.0))
        .collect();

    // Build targets with a bit of Gaussian noise (std ≈ 0.2)
    let mut rng_state: u64 = 0xdeadbeef_cafebabe;
    let noise: Vec<f64> = (0..n)
        .map(|_| {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            0.2 * (rng_state as i64 as f64) / (i64::MAX as f64)
        })
        .collect();

    let y: Vec<f64> = x
        .iter()
        .zip(noise.iter())
        .map(|(xi, ni)| 2.0 * xi + 0.5 * xi * xi + ni)
        .collect();

    println!("Dataset: {} points, true model = 2·x + 0.5·x²", n);

    // ── 2. Build feature matrix Φ  (N × 4, row-major) ─────────────────────
    //
    // Columns: [1, x, x², x³]
    // features::polynomial returns a row-major Vec<f64>.
    let (phi, d) = features::polynomial(&x, 3);
    assert_eq!(d, 4, "degree 3 → 4 columns");

    println!("Feature matrix: {} rows × {} columns", n, d);

    // ── 3. Configure and run BLR+ARD ───────────────────────────────────────
    let config = ArdConfig {
        max_iter: 200,
        tol: 1e-6,
        ..ArdConfig::default()
    };

    let fitted =
        fit(&phi, &y, n, d, &config).expect("BLR+ARD fit failed — check feature matrix dimensions");

    // ── 4. Print posterior weights ─────────────────────────────────────────
    let feature_names = ["bias (1)", "x¹", "x²", "x³"];

    println!("\n── Posterior weights ──");
    println!("{:<12} {:>12}  {:>12}", "Feature", "Mean", "Std");
    for (name, (&mu, &var)) in feature_names.iter().zip(
        fitted
            .posterior
            .mean
            .iter()
            .zip(fitted.posterior.cov.iter().step_by(d + 1)),
    ) {
        println!("{:<12} {:>12.4}  {:>12.4}", name, mu, var.sqrt().abs());
    }

    // ── 5. ARD relevance — larger = more relevant ──────────────────────────
    println!("\n── ARD relevance (1/αd) ──");
    for (name, relevance) in feature_names.iter().zip(fitted.relevance().iter()) {
        let marker = if *relevance > 0.01 {
            "✓ active"
        } else {
            "  pruned"
        };
        println!("  {:<12} {:.3e}  {}", name, relevance, marker);
    }

    // ── 6. Noise estimate ──────────────────────────────────────────────────
    println!("\n── Noise ──");
    println!(
        "  Learned noise std: {:.4}  (true: 0.2)",
        fitted.noise_std()
    );
    println!("  EM iterations:     {}", fitted.log_evidences.len());

    // ── 7. Predictions with uncertainty ────────────────────────────────────
    //
    // Predict at a few test points; show mean ± total_std (epistemic + aleatoric).
    let x_test = vec![-2.0_f64, 0.0, 2.0];
    let (phi_test, _) = features::polynomial(&x_test, 3);
    let pred = fitted.predict(&phi_test, x_test.len(), d);

    println!("\n── Predictions (3 test points) ──");
    println!(
        "{:>8}  {:>10}  {:>10}  {:>10}",
        "x", "true y", "pred mean", "±total_std"
    );
    for (i, xi) in x_test.iter().enumerate() {
        let true_y = 2.0 * xi + 0.5 * xi * xi;
        println!(
            "{:>8.2}  {:>10.4}  {:>10.4}  {:>10.4}",
            xi, true_y, pred.mean[i], pred.total_std[i]
        );
    }

    println!("\nDone. Try `cargo run --example hall_sensor -p blr-core` for a real dataset.");
}
