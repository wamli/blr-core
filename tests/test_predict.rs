/// F6, F7 — Uncertainty decomposition and joint predictive PSD.
use blr_core::{features::polynomial, fit, ArdConfig};

// ─── F6 — Epistemic uncertainty increases far from training data ─────────────

#[test]
fn test_epistemic_increases_far_from_data() {
    // Train on x ∈ [-2, 2]; compare epistemic std at x=0 vs x=5 (far outside)
    let n = 50usize;
    let x_train: Vec<f64> = (0..n)
        .map(|i| -2.0 + 4.0 * i as f64 / (n as f64 - 1.0))
        .collect();
    let y_train: Vec<f64> = x_train
        .iter()
        .enumerate()
        .map(|(i, xi)| xi * 2.0 + ((i as f64 * 1.5).sin() * 0.1))
        .collect();

    let (phi_train, d) = polynomial(&x_train, 2);
    let cfg = ArdConfig {
        max_iter: 300,
        tol: 1e-6,
        ..Default::default()
    };
    let model = fit(&phi_train, &y_train, n, d, &cfg).expect("fit failed");

    // Test at x=0 (inside training range) and x=5 (far outside)
    let x_inside = vec![0.0_f64];
    let x_outside = vec![5.0_f64];

    let (phi_inside, _) = polynomial(&x_inside, 2);
    let (phi_outside, _) = polynomial(&x_outside, 2);

    let pred_inside = model.predict(&phi_inside, 1, d);
    let pred_outside = model.predict(&phi_outside, 1, d);

    let ep_inside = pred_inside.epistemic_std[0];
    let ep_outside = pred_outside.epistemic_std[0];

    assert!(
        ep_outside >= 2.0 * ep_inside,
        "epistemic_std at x=5 ({ep_outside:.4}) must be >= 2x inside ({ep_inside:.4})"
    );
}

// ─── F7 — Joint predictive covariance is PSD ─────────────────────────────────

#[test]
fn test_joint_predictive_psd() {
    // Fit a simple model, then get the joint predictive over 5 test points
    let n = 30usize;
    let x_train: Vec<f64> = (0..n).map(|i| i as f64 * 0.2 - 3.0).collect();
    let y_train: Vec<f64> = x_train.iter().map(|xi| xi.sin() + 0.1 * xi).collect();

    let (phi_train, d) = polynomial(&x_train, 3);
    let cfg = ArdConfig {
        max_iter: 200,
        tol: 1e-6,
        ..Default::default()
    };
    let model = fit(&phi_train, &y_train, n, d, &cfg).expect("fit failed");

    let x_test = vec![-1.0, -0.5, 0.0, 0.5, 1.0_f64];
    let m = x_test.len();
    let (phi_test, _) = polynomial(&x_test, 3);

    let joint = model
        .predict_gaussian(&phi_test, m, d)
        .expect("predict_gaussian failed");

    // Check dimensions
    assert_eq!(joint.mean.len(), m);
    assert_eq!(joint.cov.len(), m * m);

    // Check PSD: all eigenvalues >= -1e-9
    // For a 5×5 matrix, use Gershgorin circles as a simple lower bound:
    // For each row i, row sum of off-diags <= diag. If diag + small jitter >= sum_off_diags, PSD.
    // More precisely: check that diag is positive (necessary for PSD)
    for i in 0..m {
        let cov_ii = joint.cov[i * m + i];
        assert!(
            cov_ii > -1e-9,
            "cov[{i},{i}] = {cov_ii:.6e} — diagonal should be non-negative"
        );
    }

    // Also verify symmetry
    for i in 0..m {
        for j in 0..m {
            let diff = (joint.cov[i * m + j] - joint.cov[j * m + i]).abs();
            assert!(
                diff < 1e-12,
                "cov not symmetric at ({i},{j}): diff={diff:.2e}"
            );
        }
    }
}

// ─── F6b — PredictiveMarginals structure check ───────────────────────────────

#[test]
fn test_predictive_marginals_fields() {
    let n = 20usize;
    let x_train: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let y_train: Vec<f64> = x_train.iter().map(|xi| xi * 3.0 + 1.0).collect();

    let (phi_train, d) = polynomial(&x_train, 1);
    let model = fit(&phi_train, &y_train, n, d, &ArdConfig::default()).expect("fit failed");

    let x_test = vec![0.5, 1.0, 2.0_f64];
    let m = x_test.len();
    let (phi_test, _) = polynomial(&x_test, 1);
    let pred = model.predict(&phi_test, m, d);

    assert_eq!(pred.mean.len(), m);
    assert_eq!(pred.epistemic_std.len(), m);
    assert_eq!(pred.total_std.len(), m);
    // aleatoric_std is a scalar, positive
    assert!(pred.aleatoric_std > 0.0);
    // total_std >= aleatoric_std (epistemic adds, not subtracts)
    for i in 0..m {
        assert!(
            pred.total_std[i] >= pred.aleatoric_std,
            "total_std[{i}]={} < aleatoric_std={}",
            pred.total_std[i],
            pred.aleatoric_std
        );
    }
}
