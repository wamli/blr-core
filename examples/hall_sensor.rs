//! Hall Position Sensor calibration using BLR+ARD.
//!
//! Reproduces Example 1 from python/notebooks/blr_ard_demo.ipynb.
//!
//! **Physical scenario:**
//! A linear Hall-effect sensor exhibits nearly linear behavior in its nominal
//! operating range, but soft saturation occurs near the measurement extremes.
//! This is modelled by y(x) = 2.5 * tanh(x / 1.2) + noise.
//!
//! **Feature basis:**
//! [1, x, x², x³, tanh(x/0.8), tanh(x/1.5)]
//! Polynomial terms alone cannot reproduce saturation — only tanh features can.
//! ARD automatically prunes polynomial terms and retains saturation features.
//!
//! Reads data/hall_sensor_calibration.csv (60 synthetic calibration points
//! generated from the true saturation curve with Gaussian noise).
//!
//! Run from anywhere:
//!   cargo run --example hall_sensor -p blr-core

use blr_core::{features, fit, ArdConfig};
use std::path::Path;

fn main() {
    // ── Load calibration data ──────────────────────────────────────────────
    // Path is resolved relative to this crate's Cargo.toml, so it works from anywhere.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let csv = Path::new(manifest_dir).join("data/hall_sensor_calibration.csv");
    let content = std::fs::read_to_string(&csv)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", csv.display(), e));

    let mut x_vals: Vec<f64> = Vec::new();
    let mut y_vals: Vec<f64> = Vec::new();
    for line in content.lines().skip(1) {
        let mut parts = line.split(',');
        x_vals.push(parts.next().unwrap().trim().parse().unwrap());
        y_vals.push(parts.next().unwrap().trim().parse().unwrap());
    }
    let n = x_vals.len();

    // ── Build feature matrix Φ (N×6) ───────────────────────────────────────
    // Columns: [1, x, x², x³, tanh(x/0.8), tanh(x/1.5)]
    let (poly_mat, _) = features::polynomial(&x_vals, 3); // 4 cols
    let mut phi = vec![0.0f64; n * 6];
    for i in 0..n {
        phi[i * 6 + 0] = poly_mat[i * 4]; // 1
        phi[i * 6 + 1] = poly_mat[i * 4 + 1]; // x
        phi[i * 6 + 2] = poly_mat[i * 4 + 2]; // x²
        phi[i * 6 + 3] = poly_mat[i * 4 + 3]; // x³
        phi[i * 6 + 4] = (x_vals[i] / 0.8).tanh(); // tanh(x/0.8)
        phi[i * 6 + 5] = (x_vals[i] / 1.5).tanh(); // tanh(x/1.5)
    }

    // ── Fit BLR+ARD ────────────────────────────────────────────────────────
    let config = ArdConfig {
        max_iter: 500,
        tol: 1e-7,
        ..ArdConfig::default()
    };
    let fitted = fit(&phi, &y_vals, n, 6, &config).expect("BLR+ARD fit failed");

    // ── Print results ───────────────────────────────────────────────────────
    let feature_names = ["1 (bias)", "x", "x²", "x³", "tanh(x/0.8)", "tanh(x/1.5)"];

    println!("=== Hall Sensor BLR+ARD Results ===");
    println!("EM iterations:          {}", fitted.log_evidences.len());
    println!("Noise std (learned):    {:.6}", fitted.noise_std());
    println!(
        "Log marginal likelihood:{:.6}",
        fitted.log_marginal_likelihood()
    );

    println!("\nPosterior mean weights:");
    for (name, &mu_j) in feature_names.iter().zip(fitted.posterior.mean.iter()) {
        println!("  {:<18} {:+.6}", name, mu_j);
    }

    println!("\nARD relevance (1/α — larger = more relevant):");
    let rel = fitted.relevance();
    for (name, r) in feature_names.iter().zip(rel.iter()) {
        println!("  {:<18} {:.3e}", name, r);
    }

    println!("\nActive features (α < geometric-mean threshold):");
    let active = fitted.relevant_features(None);
    let mut n_active = 0;
    for (name, &is_active) in feature_names.iter().zip(active.iter()) {
        if is_active {
            println!("  ✓ {}", name);
            n_active += 1;
        }
    }
    if n_active == 0 {
        println!("  (none — try lowering the threshold)");
    }

    // ── In-sample predictions ──────────────────────────────────────────────
    let preds = fitted.predict(&phi, n, 6);
    let rmse = (preds
        .mean
        .iter()
        .zip(y_vals.iter())
        .map(|(p, y)| (p - y).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    println!("\nIn-sample RMSE:         {:.6}", rmse);
    println!(
        "Mean total std:         {:.6}",
        preds.total_std.iter().sum::<f64>() / n as f64
    );
}
