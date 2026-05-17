/// F3, F4, F5 — Fit correctness, ARD sparsity, log-evidence finite/convergence.
use blr_core::{features::polynomial, fit, ArdConfig};

// ─── F3 — Simple linear recovery ─────────────────────────────────────────────

#[test]
fn test_fit_simple_linear_recovery() {
    // y = 2x + 3,  features [1, x],  N=100,  low noise
    let n = 100usize;
    let d = 2usize;
    let noise_std = 0.05;

    // Deterministic pseudo-inputs: x in [-5, 5)
    let x: Vec<f64> = (0..n).map(|i| -5.0 + 10.0 * i as f64 / n as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, xi)| {
            // Simple LCG deterministic noise
            let noise = noise_std * ((i as f64 * 1234.5678).sin());
            2.0 * xi + 3.0 + noise
        })
        .collect();

    let (phi, _) = polynomial(&x, 1);
    let cfg = ArdConfig {
        max_iter: 200,
        tol: 1e-6,
        ..Default::default()
    };
    let model = fit(&phi, &y, n, d, &cfg).expect("fit failed");

    // posterior.mean ≈ [3.0, 2.0]  within atol=0.2
    let mu = &model.posterior.mean;
    assert_eq!(mu.len(), 2);
    assert!(
        (mu[0] - 3.0).abs() < 0.2,
        "intercept={:.4}, expected 3.0",
        mu[0]
    );
    assert!(
        (mu[1] - 2.0).abs() < 0.2,
        "slope={:.4}, expected 2.0",
        mu[1]
    );
}

// ─── F4 — ARD sparsity ratio ──────────────────────────────────────────────────

#[test]
fn test_ard_sparsity_ratio() {
    // 100 samples, 20 features, only first 5 relevant.
    // Feature matrix uses deterministic spread values (not sin(k*π) which ≈ 0).
    let n = 100usize;
    let d = 20usize;
    let n_relevant = 5usize;

    // Pseudorandom feature matrix: values spread in [-1, 1]
    let phi: Vec<f64> = (0..n * d)
        .map(|k| {
            let i = k / d;
            let j = k % d;
            let seed = (i * 997 + j * 991 + i * j * 7 + 1) as f64;
            (seed * 0.314159265).sin()
        })
        .collect();

    // True weights: only first n_relevant features are active
    let true_w: Vec<f64> = (0..d)
        .map(|j| {
            if j < n_relevant {
                ((j + 1) as f64 * 0.6).sin() * 2.0
            } else {
                0.0
            }
        })
        .collect();

    // y = phi * w + small noise
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let dot: f64 = (0..d).map(|j| phi[i * d + j] * true_w[j]).sum();
            dot + 0.05 * ((i as f64 * 7.0).sin())
        })
        .collect();

    let cfg = ArdConfig {
        max_iter: 500,
        tol: 1e-6,
        ..Default::default()
    };
    let model = fit(&phi, &y, n, d, &cfg).expect("fit failed");

    let mean_relevant: f64 = model.alpha[..n_relevant].iter().sum::<f64>() / n_relevant as f64;
    let mean_irrelevant: f64 =
        model.alpha[n_relevant..].iter().sum::<f64>() / (d - n_relevant) as f64;

    let ratio = mean_irrelevant / mean_relevant;
    assert!(
        ratio > 5.0,
        "ARD sparsity ratio={:.2} < 5.0 (irrelevant alpha should be >> relevant)",
        ratio
    );
}

// ─── F5 — Log evidence finite and convergence ─────────────────────────────────

#[test]
fn test_log_evidences_finite() {
    let n = 30usize;
    let d = 5usize;
    let phi: Vec<f64> = (0..n * d).map(|k| (k as f64 * 1.1).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();

    let cfg = ArdConfig::default();
    let model = fit(&phi, &y, n, d, &cfg).expect("fit failed");

    for (i, &lml) in model.log_evidences.iter().enumerate() {
        assert!(lml.is_finite(), "log_evidence[{i}] = {lml} is not finite");
    }
}

#[test]
fn test_converges_before_max_iter() {
    let n = 60usize;
    let d = 11usize;
    let phi: Vec<f64> = (0..n * d).map(|k| (k as f64 * 2.7).sin()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| {
            let dot: f64 = (0..d)
                .map(|j| phi[i * d + j] * ((j as f64 + 1.0) * 0.1))
                .sum();
            dot + 0.05 * ((i as f64).sin())
        })
        .collect();

    let cfg = ArdConfig {
        max_iter: 200,
        tol: 1e-3,
        ..Default::default()
    };
    let model = fit(&phi, &y, n, d, &cfg).expect("fit failed");
    assert!(
        model.log_evidences.len() < 200,
        "did not converge: {} iterations out of 200",
        model.log_evidences.len()
    );
}
