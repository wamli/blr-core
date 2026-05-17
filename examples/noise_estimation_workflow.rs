//! Noise estimation workflow example for `blr-core`.
//!
//! Demonstrates how to:
//! 1. Fit a BLR+ARD model to noisy sensor data
//! 2. Extract the noise standard deviation from the learned β hyperparameter
//! 3. Compute a confidence interval around the noise estimate
//! 4. Assess whether the estimate is reliable for calibration decisions
//!
//! Run from the repository root:
//!
//! ```bash
//! cargo run --example noise_estimation_workflow -p blr-core
//! ```

use blr_core::{
    features, fit,
    noise_estimation::{estimate_noise_with_confidence, estimate_sensor_noise},
    ArdConfig,
};

/// Returns the next standard-normal sample from a seeded LCG using the
/// Box-Muller transform.  Both `u1` and `u2` are drawn from the same LCG
/// state so no external RNG dependency is needed.
fn next_gaussian(state: &mut u64) -> f64 {
    // Advance LCG and map to (0, 1) — use upper 53 bits for IEEE-754 precision.
    let uniform = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Add 0.5 before dividing so u1 is never exactly 0 (avoids ln(0)).
        ((*s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let u1 = uniform(state);
    let u2 = uniform(state);
    // Box-Muller: maps two uniform samples to one standard-normal sample.
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn main() {
    // ── 1. Simulate noisy sensor measurements at varying noise levels ──────
    //
    // We create three datasets with known true noise levels:
    // σ_true ∈ {0.05, 0.20, 0.50}
    // then verify BLR+ARD recovers each one.
    //
    // True function: f(x) = 1 + x − 0.5x² + 0.1x³  (a cubic polynomial).
    // Because we use degree-3 polynomial features the model can fit f exactly,
    // so residuals are pure Gaussian noise — the ideal test for noise recovery.

    let true_noise_levels = [0.05_f64, 0.20, 0.50];
    let n = 16_usize; // larger N → smaller chi-squared variance on β̂
    let degree = 3; // polynomial features: [1, x, x², x³]

    // Input points evenly spaced on [-2, 2]
    let x: Vec<f64> = (0..n)
        .map(|i| -2.0 + 4.0 * (i as f64) / (n as f64 - 1.0))
        .collect();

    // Build feature matrix once (shared across all noise levels)
    let (phi, d) = features::polynomial(&x, degree);

    println!("Noise Estimation Workflow");
    println!("════════════════════════════════════════════════════════════");
    println!("  N = {n} observations, D = {d} polynomial features (degree {degree})");
    println!("  True signal: f(x) = 1 + x − 0.5x² + 0.1x³  (cubic)");
    println!("════════════════════════════════════════════════════════════\n");

    for &sigma_true in &true_noise_levels {
        println!("── True noise σ = {sigma_true:.2} ──────────────────────────────────────────");

        // Generate targets: y = f(x) + Gaussian noise
        let mut state: u64 = 0x1234567890abcdef_u64.wrapping_add(sigma_true.to_bits());
        let y: Vec<f64> = x
            .iter()
            .map(|xi| {
                let fx = 1.0 + xi - 0.5 * xi.powi(2) + 0.1 * xi.powi(3);
                fx + sigma_true * next_gaussian(&mut state)
            })
            .collect();

        // Fit BLR+ARD
        let config = ArdConfig {
            max_iter: 300,
            tol: 1e-6,
            ..ArdConfig::default()
        };
        let fitted = fit(&phi, &y, n, d, &config).expect("fit should succeed on valid input");

        // ── 2. Simple point estimate from fitted β ─────────────────────
        let beta = fitted.beta;
        let sigma_point =
            estimate_sensor_noise(beta).expect("beta should be positive after fitting");

        // ── 3. Confidence interval ─────────────────────────────────────
        let estimate = estimate_noise_with_confidence(beta, n, d);

        println!("  β (noise precision):    {:.4}", beta);
        println!(
            "  σ_noise (point est):    {:.4}  (true: {:.2})",
            sigma_point, sigma_true
        );
        println!(
            "  95% heuristic interval: [{:.4}, {:.4}]",
            estimate.lower_bound, estimate.upper_bound
        );
        println!("  Confidence label:       {}", estimate.confidence);

        // ── 4. Assess reliability ──────────────────────────────────────
        let error_pct = ((sigma_point - sigma_true) / sigma_true * 100.0).abs();
        let status = if error_pct < 10.0 {
            "✓ accurate"
        } else if error_pct < 25.0 {
            "~ acceptable"
        } else {
            "✗ inaccurate (more data needed)"
        };
        println!("  Recovery error:         {:.1}%  {}", error_pct, status);
        println!();
    }

    // ── 5. Guidance for calibration decisions ──────────────────────────────
    println!("════════════════════════════════════════════════════════════");
    println!("Guidance:");
    println!("  • 'preliminary' (n<30):    use as rough guide only");
    println!("  • 'transitional' (30–50):  reasonable for most applications");
    println!("  • 'stable' (n>50):         reliable for precision calibration");
    println!("  • Recovery error <10%:     BLR+ARD noise estimate is accurate");
    println!("  • If error >25%:           try more training points or");
    println!("                             adjust feature set to reduce underfitting");
    println!("════════════════════════════════════════════════════════════");

    println!("\nDone. See noise_estimation module docs for the underlying formula.");
}
