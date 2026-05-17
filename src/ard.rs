//! BLR+ARD EM fitting, prediction API, and configuration.
//!
//! Implements MacKay (1992) / Tipping (2001) empirical Bayes for Bayesian
//! Linear Regression with Automatic Relevance Determination.
//!
//! ## Overview
//!
//! The main entry point is [`fit`], which runs an EM loop to find the
//! posterior distribution over weights and the noise precision hyperparameter β.
//! ARD places an independent precision hyperparameter `α_d` on each weight;
//! features with low signal converge to `α_d → ∞`, effectively removing them
//! from the model.
//!
//! After fitting, use [`FittedArd`] to:
//! - Inspect posterior weight mean and covariance
//! - Identify active features via [`FittedArd::relevant_features`]
//! - Predict on new data via [`FittedArd::predict`]
//!
//! ## Example: Basic Fit
//!
//! ```rust
//! use blr_core::{fit, ArdConfig};
//!
//! // 20 observations, 3 features (row-major feature matrix)
//! let phi: Vec<f64> = vec![1.0; 60];
//! let y:   Vec<f64> = vec![0.5; 20];
//! let config = ArdConfig::default();
//!
//! let fitted = fit(&phi, &y, 20, 3, &config)
//!     .expect("fit should succeed with valid input");
//!
//! assert!(fitted.noise_std() > 0.0);
//! assert_eq!(fitted.relevant_features(None).len(), 3);
//! ```
//!
//! ## Example: Inspect ARD Relevance
//!
//! ```rust
//! use blr_core::{fit, ArdConfig};
//!
//! let phi: Vec<f64> = vec![1.0; 60];
//! let y:   Vec<f64> = vec![0.5; 20];
//! let fitted = fit(&phi, &y, 20, 3, &ArdConfig::default()).unwrap();
//!
//! // relevance() returns 1/αd — larger means more relevant
//! let rel = fitted.relevance();
//! println!("Feature relevances: {:?}", rel);
//!
//! // relevant_features() returns a boolean mask
//! let active = fitted.relevant_features(None);
//! let n_active = active.iter().filter(|&&x| x).count();
//! println!("{} of {} features are active", n_active, active.len());
//! ```
//!
//! ## EM Algorithm Summary
//!
//! Each iteration:
//!
//! 1. **E-step**: Compute posterior mean `μ` and covariance `Σ`
//!    using the current `{α_d, β}`.
//! 2. **M-step**: Update each `α_d` and optionally β using the posterior
//!    statistics (gamma updates from MacKay 1992 Eq. 32–33).
//! 3. **Convergence**: Stop when the change in log-evidence between
//!    consecutive iterations is below `ArdConfig::tol`, or `max_iter` is reached.
//!
//! ## References
//!
//! - MacKay, D. J. C. (1992). "Bayesian Interpolation."
//!   *Neural Computation*, 4(3), 415–447.
//! - Tipping, M. E. (2001). "Sparse Bayesian Learning and the Relevance Vector Machine."
//!   *Journal of Machine Learning Research*, 1, 211–244.

use std::f64::consts::PI;

use faer::linalg::{matmul, solvers::Solve};
use faer::{Accum, Mat, Par, Side};

use crate::gaussian::cholesky_logdet;
use crate::{BLRError, Gaussian};

// ─── BLRPrior ─────────────────────────────────────────────────────────────────

/// Custom prior for BLR+ARD fitting (batch transfer learning).
///
/// Encodes prior knowledge about the parameter distribution of a sensor
/// ensemble, aggregated from N reference sensors calibrated in Phase 1.
/// Pass this to [`fit_with_prior`] to accelerate calibration of new
/// production sensors (Phase 2) using transferred knowledge.
///
/// ## Fields
///
/// All three vectors must have the same length D (feature dimension).
///
/// ## Reference
///
/// Berger, Schott, Paul. "Bayesian Sensor Calibration."
/// IEEE Sensors Journal, Vol. 22, No. 20, October 2022.
/// Equations (20)–(21): prior mean and covariance from ensemble.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BLRPrior {
    /// Prior weight mean μ_0 of length D.
    pub mean: Vec<f64>,
    /// Prior weight covariance Σ_0 (D×D, symmetric positive-definite, row-major).
    pub cov: Vec<f64>,
    /// Prior ARD precision hyperparameters α_0 of length D.
    pub alphas: Vec<f64>,
}

impl BLRPrior {
    /// Validate that dimensions are consistent and covariance is positive-definite.
    ///
    /// Checks:
    /// 1. `mean.len() == alphas.len()` (both equal D).
    /// 2. `cov.len() == D * D` (square matrix).
    /// 3. Cholesky factorization of `cov` succeeds (confirms PSD).
    pub fn validate(&self) -> Result<(), BLRError> {
        let d = self.mean.len();
        if self.alphas.len() != d {
            return Err(BLRError::DimMismatch {
                expected: d,
                got: self.alphas.len(),
            });
        }
        if self.cov.len() != d * d {
            return Err(BLRError::DimMismatch {
                expected: d * d,
                got: self.cov.len(),
            });
        }
        if d == 0 {
            return Err(BLRError::DimMismatch {
                expected: 1,
                got: 0,
            });
        }
        // Verify positive-definiteness by attempting Cholesky factorization.
        let cov_mat = Mat::<f64>::from_fn(d, d, |i, j| self.cov[i * d + j]);
        cov_mat
            .llt(Side::Lower)
            .map_err(|_| BLRError::SingularMatrix)?;
        Ok(())
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the BLR+ARD EM fitting loop.
///
/// Defaults match the Python reference:
/// `alpha_init=1.0, beta_init=1.0, max_iter=100, tol=1e-5, update_beta=true`.
#[derive(Debug, Clone)]
pub struct ArdConfig {
    /// Initial value for all ARD precision hyperparameters α_j.
    pub alpha_init: f64,
    /// Initial noise precision β = 1/σ².
    pub beta_init: f64,
    /// Maximum number of EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on the period-2 log-evidence delta.
    pub tol: f64,
    /// Whether to update β during the M-step.
    pub update_beta: bool,
}

impl Default for ArdConfig {
    fn default() -> Self {
        Self {
            alpha_init: 1.0,
            beta_init: 1.0,
            max_iter: 100,
            tol: 1e-5,
            update_beta: true,
        }
    }
}

// ─── Predictive distribution (per test point) ────────────────────────────────

/// Marginal predictive distributions for a set of test points.
///
/// Uncertainty is decomposed into aleatoric (noise) and epistemic (model)
/// components, matching the Python `predict()` output.
pub struct PredictiveMarginals {
    /// Predictive mean E\[y_*\] for each test point.
    pub mean: Vec<f64>,
    /// Aleatoric std = 1/√β (noise; same for all points).
    pub aleatoric_std: f64,
    /// Epistemic std √(φ_* Σ φ_*ᵀ) for each test point.
    pub epistemic_std: Vec<f64>,
    /// Total std √(aleatoric² + epistemic²) for each test point.
    pub total_std: Vec<f64>,
}

// ─── Fitted model ─────────────────────────────────────────────────────────────

/// Result of a successful `fit()` call.
pub struct FittedArd {
    /// Weight-space posterior N(μ, Σ).
    pub posterior: Gaussian,
    /// ARD precision hyperparameters α (D,).
    pub alpha: Vec<f64>,
    /// Noise precision β.
    pub beta: f64,
    /// Log marginal likelihood per EM iteration.
    pub log_evidences: Vec<f64>,
    /// Number of training samples N used to fit this model.
    pub n_samples: usize,
}

impl FittedArd {
    // ── Prediction ──────────────────────────────────────────────────────────

    /// Marginal predictions with decomposed uncertainty.
    ///
    /// `phi_test` is the N_test × D feature matrix in row-major order.
    pub fn predict(
        &self,
        phi_test: &[f64],
        n_test: usize,
        n_features: usize,
    ) -> PredictiveMarginals {
        let d = n_features;
        let sigma_mat = Mat::<f64>::from_fn(d, d, |i, j| self.posterior.cov[i * d + j]);
        let mu_col = Mat::<f64>::from_fn(d, 1, |i, _| self.posterior.mean[i]);

        let aleatoric_var = 1.0 / self.beta;
        let aleatoric_std = aleatoric_var.sqrt();

        let mut mean = Vec::with_capacity(n_test);
        let mut epistemic_std = Vec::with_capacity(n_test);
        let mut total_std = Vec::with_capacity(n_test);

        for i in 0..n_test {
            let phi_row = Mat::<f64>::from_fn(1, d, |_, j| phi_test[i * d + j]);

            // mean[i] = phi_row * mu
            let mut m_mat = Mat::<f64>::zeros(1, 1);
            matmul::matmul(
                m_mat.as_mut(),
                Accum::Replace,
                phi_row.as_ref(),
                mu_col.as_ref(),
                1.0_f64,
                Par::Seq,
            );
            mean.push(m_mat[(0, 0)]);

            // epistemic_var[i] = phi_row * Sigma * phi_row^T
            let mut sigma_phi_t = Mat::<f64>::zeros(d, 1);
            matmul::matmul(
                sigma_phi_t.as_mut(),
                Accum::Replace,
                sigma_mat.as_ref(),
                phi_row.as_ref().transpose(),
                1.0_f64,
                Par::Seq,
            );
            let mut ep_var_mat = Mat::<f64>::zeros(1, 1);
            matmul::matmul(
                ep_var_mat.as_mut(),
                Accum::Replace,
                phi_row.as_ref(),
                sigma_phi_t.as_ref(),
                1.0_f64,
                Par::Seq,
            );
            let ep_var = ep_var_mat[(0, 0)].max(0.0);
            epistemic_std.push(ep_var.sqrt());
            total_std.push((aleatoric_var + ep_var).sqrt());
        }

        PredictiveMarginals {
            mean,
            aleatoric_std,
            epistemic_std,
            total_std,
        }
    }

    /// Full joint predictive Gaussian over all M test points.
    ///
    /// Returns N(Φ_test μ, Φ_test Σ Φ_test^T + (1/β) I_M).
    pub fn predict_gaussian(
        &self,
        phi_test: &[f64],
        n_test: usize,
        n_features: usize,
    ) -> Result<Gaussian, BLRError> {
        let d = n_features;
        let m = n_test;

        let phi_mat = Mat::<f64>::from_fn(m, d, |i, j| phi_test[i * d + j]);
        let sigma_mat = Mat::<f64>::from_fn(d, d, |i, j| self.posterior.cov[i * d + j]);
        let mu_col = Mat::<f64>::from_fn(d, 1, |i, _| self.posterior.mean[i]);

        // pred_mean = Φ_test * μ  (M×1)
        let mut pred_mean_mat = Mat::<f64>::zeros(m, 1);
        matmul::matmul(
            pred_mean_mat.as_mut(),
            Accum::Replace,
            phi_mat.as_ref(),
            mu_col.as_ref(),
            1.0_f64,
            Par::Seq,
        );

        // pred_cov = Φ_test * Σ * Φ_test^T + (1/β) I_M  (M×M)
        // Step 1: tmp = Φ_test * Σ  (M×D)
        let mut tmp = Mat::<f64>::zeros(m, d);
        matmul::matmul(
            tmp.as_mut(),
            Accum::Replace,
            phi_mat.as_ref(),
            sigma_mat.as_ref(),
            1.0_f64,
            Par::Seq,
        );
        // Step 2: pred_cov = tmp * Φ_test^T  (M×M)
        let mut pred_cov = Mat::<f64>::zeros(m, m);
        matmul::matmul(
            pred_cov.as_mut(),
            Accum::Replace,
            tmp.as_ref(),
            phi_mat.as_ref().transpose(),
            1.0_f64,
            Par::Seq,
        );
        // Step 3: add noise + jitter to diagonal
        let noise_var = 1.0 / self.beta;
        for i in 0..m {
            pred_cov[(i, i)] += noise_var + 1e-9; // jitter for PSD guarantee
        }

        let pred_cov_ref = pred_cov.as_ref();
        let pred_mean_vec: Vec<f64> = (0..m).map(|i| pred_mean_mat[(i, 0)]).collect();
        let pred_cov_vec: Vec<f64> = (0..m)
            .flat_map(|i| (0..m).map(move |j| pred_cov_ref[(i, j)]))
            .collect();

        Gaussian::new(pred_mean_vec, pred_cov_vec)
    }

    // ── Interpretability ────────────────────────────────────────────────────

    /// Feature relevance scores: 1/α_j (higher = more relevant).
    pub fn relevance(&self) -> Vec<f64> {
        self.alpha.iter().map(|a| 1.0 / a).collect()
    }

    /// Boolean mask: `true` where feature j is relevant (α_j < threshold).
    ///
    /// Default threshold = geometric mean of α (exp(mean(ln(α_j)))).
    ///
    /// TODO: replace geometric mean with median for a more robust heuristic
    /// in a future iteration.
    pub fn relevant_features(&self, threshold: Option<f64>) -> Vec<bool> {
        let t = threshold.unwrap_or_else(|| {
            let ln_mean = self.alpha.iter().map(|a| a.ln()).sum::<f64>() / self.alpha.len() as f64;
            ln_mean.exp()
        });
        self.alpha.iter().map(|a| *a < t).collect()
    }

    // ── Summary scalars ─────────────────────────────────────────────────────

    /// Noise standard deviation 1/√β.
    pub fn noise_std(&self) -> f64 {
        1.0 / self.beta.sqrt()
    }

    /// Log marginal likelihood at the last EM iteration.
    pub fn log_marginal_likelihood(&self) -> f64 {
        *self.log_evidences.last().unwrap_or(&f64::NEG_INFINITY)
    }

    // ── Active Learning API ─────────────────────────────────────────────────

    /// Noise precision β accessor (= 1/σ²_noise).
    pub fn noise_precision(&self) -> f64 {
        self.beta
    }

    /// Posterior covariance Σ as a flat row-major D×D slice.
    pub fn posterior_covariance(&self) -> &[f64] {
        &self.posterior.cov
    }

    /// Number of training samples N used during fitting.
    pub fn sample_count(&self) -> usize {
        self.n_samples
    }

    /// Posterior standard deviations for arbitrary test points.
    ///
    /// # Arguments
    /// - `phi_test`: N_test × D feature matrix, row-major flat slice
    /// - `n_test`: number of test points
    /// - `n_features`: feature dimension D (must match the training feature dim)
    ///
    /// # Returns
    /// Vec of length `n_test` with posterior std for each point.
    pub fn posterior_std(&self, phi_test: &[f64], n_test: usize, n_features: usize) -> Vec<f64> {
        let d = n_features;
        let sigma_cov = &self.posterior.cov;
        let noise_var = 1.0 / self.beta.max(1e-10);
        (0..n_test)
            .map(|i| {
                let phi_i = &phi_test[i * d..(i + 1) * d];
                let mut sigma_phi = vec![0.0_f64; d];
                for row in 0..d {
                    for col in 0..d {
                        sigma_phi[row] += sigma_cov[row * d + col] * phi_i[col];
                    }
                }
                let epistemic: f64 = phi_i.iter().zip(sigma_phi.iter()).map(|(a, b)| a * b).sum();
                (noise_var + epistemic.max(0.0)).sqrt()
            })
            .collect()
    }

    /// Posterior std evaluated on a uniform 1-D input grid.
    ///
    /// # Arguments
    /// - `input_range`: (min, max) of the input domain
    /// - `resolution`: number of grid points (≥ 2)
    /// - `feature_fn`: maps a scalar input to a feature vector of length D
    ///
    /// # Returns
    /// `(grid_points, std_devs)` — both of length `resolution`.
    pub fn posterior_std_grid(
        &self,
        input_range: (f64, f64),
        resolution: usize,
        feature_fn: &dyn Fn(f64) -> Vec<f64>,
    ) -> (Vec<f64>, Vec<f64>) {
        let d_sq = self.posterior.cov.len();
        let d = (d_sq as f64).sqrt() as usize;
        let resolution = resolution.max(2);
        let step = (input_range.1 - input_range.0) / (resolution - 1) as f64;
        let grid: Vec<f64> = (0..resolution)
            .map(|k| input_range.0 + k as f64 * step)
            .collect();
        let mut phi_grid = Vec::with_capacity(resolution * d);
        for &x in &grid {
            let feats = feature_fn(x);
            let actual = feats.len().min(d);
            phi_grid.extend_from_slice(&feats[..actual]);
            if actual < d {
                phi_grid.extend(std::iter::repeat(0.0).take(d - actual));
            }
        }
        let stds = self.posterior_std(&phi_grid, resolution, d);
        (grid, stds)
    }
}

// ─── Log evidence helper ──────────────────────────────────────────────────────

/// Compute log marginal likelihood (evidence) matching the Python
/// `_log_evidence` implementation, including the `+D·log(2π)/2` term.
///
/// L = 0.5 * (Σ log(α_j) + N log(β) - logdet(Σ_inv) - β ||r||² - μᵀΛμ
///           + D log(2π)) - 0.5 N log(2π)
fn log_evidence(
    n: usize,
    d: usize,
    alpha: &[f64],
    beta: f64,
    mu: &[f64],
    logdet_sigma_inv: f64,
    residual_sq: f64,
) -> f64 {
    let log_alpha_sum: f64 = alpha.iter().map(|a| a.ln()).sum();
    let mu_lambda_mu: f64 = alpha.iter().zip(mu.iter()).map(|(a, m)| a * m * m).sum();

    0.5 * (log_alpha_sum + (n as f64) * beta.ln()
        - logdet_sigma_inv
        - beta * residual_sq
        - mu_lambda_mu
        + (d as f64) * (2.0 * PI).ln())
        - 0.5 * (n as f64) * (2.0 * PI).ln()
}

// ─── Fit entry point ──────────────────────────────────────────────────────────

/// Fit BLR+ARD via EM (Type-II maximum likelihood / empirical Bayes).
///
/// # Arguments
/// - `phi`: N×D feature matrix, row-major.
/// - `y`: N target values.
/// - `n`: number of training points (rows of phi).
/// - `d`: number of features (columns of phi).
/// - `config`: fitting hyperparameters.
///
/// # Returns
/// `Ok(FittedArd)` on success, `Err(BLRError)` if a matrix inversion fails.
pub fn fit(
    phi: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    config: &ArdConfig,
) -> Result<FittedArd, BLRError> {
    if phi.len() != n * d {
        return Err(BLRError::DimMismatch {
            expected: n * d,
            got: phi.len(),
        });
    }
    if y.len() != n {
        return Err(BLRError::DimMismatch {
            expected: n,
            got: y.len(),
        });
    }

    let phi_mat = Mat::<f64>::from_fn(n, d, |i, j| phi[i * d + j]);
    let y_mat = Mat::<f64>::from_fn(n, 1, |i, _| y[i]);

    // Pre-compute Φᵀ Φ (D×D) and Φᵀ y (D×1) — reused every iteration.
    let mut phi_t_phi = Mat::<f64>::zeros(d, d);
    matmul::matmul(
        phi_t_phi.as_mut(),
        Accum::Replace,
        phi_mat.as_ref().transpose(),
        phi_mat.as_ref(),
        1.0_f64,
        Par::Seq,
    );

    let mut phi_t_y = Mat::<f64>::zeros(d, 1);
    matmul::matmul(
        phi_t_y.as_mut(),
        Accum::Replace,
        phi_mat.as_ref().transpose(),
        y_mat.as_ref(),
        1.0_f64,
        Par::Seq,
    );

    // Initialise hyperparameters.
    let mut alpha = vec![config.alpha_init; d];
    let mut beta = config.beta_init;
    let mut log_evidences: Vec<f64> = Vec::new();

    // Working storage reused across iterations.
    let mut sigma_mat = Mat::<f64>::zeros(d, d);
    let mut mu_vec = vec![0.0_f64; d];

    for _iter in 0..config.max_iter {
        // ── E-step ────────────────────────────────────────────────────────
        // σ_inv = diag(α) + β Φᵀ Φ
        let mut sigma_inv = Mat::<f64>::from_fn(d, d, |i, j| beta * phi_t_phi[(i, j)]);
        for j in 0..d {
            sigma_inv[(j, j)] += alpha[j];
        }

        // Cholesky: L Lᵀ = σ_inv
        let llt = sigma_inv
            .llt(Side::Lower)
            .map_err(|_| BLRError::SingularMatrix)?;

        // Σ = σ_inv⁻¹  (solve with identity)
        let eye = Mat::<f64>::identity(d, d);
        sigma_mat = llt.solve(eye.as_ref());

        // μ = β Σ Φᵀ y  (solve σ_inv · μ = β Φᵀ y)
        let mut rhs = phi_t_y.clone();
        for i in 0..d {
            rhs[(i, 0)] *= beta;
        }
        let mu_mat = llt.solve(rhs.as_ref());
        for i in 0..d {
            mu_vec[i] = mu_mat[(i, 0)];
        }

        // Log-determinant of σ_inv via manual Cholesky diagonal
        let logdet_sigma_inv = cholesky_logdet(&sigma_inv, d)?;

        // ── Residuals (needed for log-evidence and β update) ──────────────
        let mut phi_mu = Mat::<f64>::zeros(n, 1);
        let mu_mat_ref = Mat::<f64>::from_fn(d, 1, |i, _| mu_vec[i]);
        matmul::matmul(
            phi_mu.as_mut(),
            Accum::Replace,
            phi_mat.as_ref(),
            mu_mat_ref.as_ref(),
            1.0_f64,
            Par::Seq,
        );
        let residual_sq: f64 = (0..n)
            .map(|i| {
                let r = y[i] - phi_mu[(i, 0)];
                r * r
            })
            .sum();

        // ── M-step ────────────────────────────────────────────────────────
        // γ_j = 1 − α_j Σ_jj     (effective parameters per feature)
        let gamma: Vec<f64> = (0..d).map(|j| 1.0 - alpha[j] * sigma_mat[(j, j)]).collect();

        // α_j = γ_j / (μ_j² + ε),  clamp to ≥ 1e-8
        for j in 0..d {
            alpha[j] = (gamma[j] / (mu_vec[j] * mu_vec[j] + 1e-10)).max(1e-8);
        }

        // β = (N − Σγ_j) / (||r||² + ε),  clamp to ≥ 1e-8
        if config.update_beta {
            let gamma_sum: f64 = gamma.iter().sum();
            beta = ((n as f64 - gamma_sum) / (residual_sq + 1e-10)).max(1e-8);
        }

        // ── Log evidence — computed after M-step (matches Python ordering) ─
        // Uses updated alpha/beta but E-step logdet_sigma_inv and mu from this iter.
        let lml = log_evidence(n, d, &alpha, beta, &mu_vec, logdet_sigma_inv, residual_sq);
        log_evidences.push(lml);

        // ── Convergence: period-2 paired log-evidence delta ───────────────
        let n_ev = log_evidences.len();
        let delta = if n_ev >= 4 {
            let mean_curr = 0.5 * (log_evidences[n_ev - 1] + log_evidences[n_ev - 2]);
            let mean_prev = 0.5 * (log_evidences[n_ev - 3] + log_evidences[n_ev - 4]);
            (mean_curr - mean_prev).abs()
        } else if n_ev >= 2 {
            (log_evidences[n_ev - 1] - log_evidences[n_ev - 2]).abs()
        } else {
            f64::INFINITY
        };

        if delta < config.tol {
            break;
        }
    }

    // Build final posterior Gaussian.
    let mu_final: Vec<f64> = mu_vec.clone();
    let cov_final: Vec<f64> = {
        let sigma_ref = sigma_mat.as_ref();
        (0..d)
            .flat_map(|i| (0..d).map(move |j| sigma_ref[(i, j)]))
            .collect()
    };
    let posterior = Gaussian::new(mu_final, cov_final)?;

    Ok(FittedArd {
        posterior,
        alpha,
        beta,
        log_evidences,
        n_samples: n,
    })
}

// ─── fit_with_prior entry point ───────────────────────────────────────────────

/// Fit BLR+ARD with an optional informed prior (batch transfer learning).
///
/// When `prior` is `Some`, the EM loop initialises weight mean and ARD alphas
/// from the prior values rather than the `ArdConfig` defaults. This allows
/// knowledge from a reference batch of sensors to accelerate convergence on a
/// new production sensor.
///
/// When `prior` is `None`, this function is numerically equivalent to `fit()`.
///
/// # Arguments
/// - `phi`: N×D feature matrix, row-major.
/// - `y`: N target values.
/// - `n`: number of training points (rows of phi).
/// - `d`: number of features (columns of phi).
/// - `config`: fitting hyperparameters.
/// - `prior`: optional `BLRPrior` from a reference batch.
///
/// # Returns
/// `Ok(FittedArd)` on success, `Err(BLRError)` if input validation or
/// matrix operations fail.
pub fn fit_with_prior(
    phi: &[f64],
    y: &[f64],
    n: usize,
    d: usize,
    config: &ArdConfig,
    prior: Option<&BLRPrior>,
) -> Result<FittedArd, BLRError> {
    if phi.len() != n * d {
        return Err(BLRError::DimMismatch {
            expected: n * d,
            got: phi.len(),
        });
    }
    if y.len() != n {
        return Err(BLRError::DimMismatch {
            expected: n,
            got: y.len(),
        });
    }

    // Validate prior dimensions if provided.
    if let Some(p) = prior {
        p.validate()?;
        if p.mean.len() != d {
            return Err(BLRError::DimMismatch {
                expected: d,
                got: p.mean.len(),
            });
        }
    }

    let phi_mat = Mat::<f64>::from_fn(n, d, |i, j| phi[i * d + j]);
    let y_mat = Mat::<f64>::from_fn(n, 1, |i, _| y[i]);

    // Pre-compute Φᵀ Φ (D×D) and Φᵀ y (D×1) — reused every iteration.
    let mut phi_t_phi = Mat::<f64>::zeros(d, d);
    matmul::matmul(
        phi_t_phi.as_mut(),
        Accum::Replace,
        phi_mat.as_ref().transpose(),
        phi_mat.as_ref(),
        1.0_f64,
        Par::Seq,
    );

    let mut phi_t_y = Mat::<f64>::zeros(d, 1);
    matmul::matmul(
        phi_t_y.as_mut(),
        Accum::Replace,
        phi_mat.as_ref().transpose(),
        y_mat.as_ref(),
        1.0_f64,
        Par::Seq,
    );

    // Initialise hyperparameters from prior or config defaults.
    let mut alpha: Vec<f64> = prior
        .map(|p| p.alphas.clone())
        .unwrap_or_else(|| vec![config.alpha_init; d]);
    let mut beta = config.beta_init;
    let mut log_evidences: Vec<f64> = Vec::new();

    // Working storage reused across iterations.
    let mut sigma_mat = Mat::<f64>::zeros(d, d);
    // Initialise mu from prior mean if available, else zeros.
    let mut mu_vec: Vec<f64> = prior
        .map(|p| p.mean.clone())
        .unwrap_or_else(|| vec![0.0f64; d]);

    for _iter in 0..config.max_iter {
        // ── E-step ────────────────────────────────────────────────────────
        // σ_inv = diag(α) + β Φᵀ Φ
        let mut sigma_inv = Mat::<f64>::from_fn(d, d, |i, j| beta * phi_t_phi[(i, j)]);
        for j in 0..d {
            sigma_inv[(j, j)] += alpha[j];
        }

        // Cholesky: L Lᵀ = σ_inv
        let llt = sigma_inv
            .llt(Side::Lower)
            .map_err(|_| BLRError::SingularMatrix)?;

        // Σ = σ_inv⁻¹  (solve with identity)
        let eye = Mat::<f64>::identity(d, d);
        sigma_mat = llt.solve(eye.as_ref());

        // μ = β Σ Φᵀ y  (solve σ_inv · μ = β Φᵀ y)
        let mut rhs = phi_t_y.clone();
        for i in 0..d {
            rhs[(i, 0)] *= beta;
        }
        let mu_mat = llt.solve(rhs.as_ref());
        for i in 0..d {
            mu_vec[i] = mu_mat[(i, 0)];
        }

        // Log-determinant of σ_inv via manual Cholesky diagonal
        let logdet_sigma_inv = cholesky_logdet(&sigma_inv, d)?;

        // ── Residuals (needed for log-evidence and β update) ──────────────
        let mut phi_mu = Mat::<f64>::zeros(n, 1);
        let mu_mat_ref = Mat::<f64>::from_fn(d, 1, |i, _| mu_vec[i]);
        matmul::matmul(
            phi_mu.as_mut(),
            Accum::Replace,
            phi_mat.as_ref(),
            mu_mat_ref.as_ref(),
            1.0_f64,
            Par::Seq,
        );
        let residual_sq: f64 = (0..n)
            .map(|i| {
                let r = y[i] - phi_mu[(i, 0)];
                r * r
            })
            .sum();

        // ── M-step ────────────────────────────────────────────────────────
        let gamma: Vec<f64> = (0..d).map(|j| 1.0 - alpha[j] * sigma_mat[(j, j)]).collect();

        for j in 0..d {
            alpha[j] = (gamma[j] / (mu_vec[j] * mu_vec[j] + 1e-10)).max(1e-8);
        }

        if config.update_beta {
            let gamma_sum: f64 = gamma.iter().sum();
            beta = ((n as f64 - gamma_sum) / (residual_sq + 1e-10)).max(1e-8);
        }

        let lml = log_evidence(n, d, &alpha, beta, &mu_vec, logdet_sigma_inv, residual_sq);
        log_evidences.push(lml);

        let n_ev = log_evidences.len();
        let delta = if n_ev >= 4 {
            let mean_curr = 0.5 * (log_evidences[n_ev - 1] + log_evidences[n_ev - 2]);
            let mean_prev = 0.5 * (log_evidences[n_ev - 3] + log_evidences[n_ev - 4]);
            (mean_curr - mean_prev).abs()
        } else if n_ev >= 2 {
            (log_evidences[n_ev - 1] - log_evidences[n_ev - 2]).abs()
        } else {
            f64::INFINITY
        };

        if delta < config.tol {
            break;
        }
    }

    let mu_final: Vec<f64> = mu_vec.clone();
    let cov_final: Vec<f64> = {
        let sigma_ref = sigma_mat.as_ref();
        (0..d)
            .flat_map(|i| (0..d).map(move |j| sigma_ref[(i, j)]))
            .collect()
    };
    let posterior = Gaussian::new(mu_final, cov_final)?;

    Ok(FittedArd {
        posterior,
        alpha,
        beta,
        log_evidences,
        n_samples: n,
    })
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ard_config_defaults() {
        let cfg = ArdConfig::default();
        assert_eq!(cfg.alpha_init, 1.0);
        assert_eq!(cfg.beta_init, 1.0);
        assert_eq!(cfg.max_iter, 100);
        assert_eq!(cfg.tol, 1e-5);
        assert!(cfg.update_beta);
    }

    #[test]
    fn test_log_evidence_helper() {
        // Smoke test: result must be finite.
        let lml = log_evidence(10, 3, &[1.0; 3], 1.0, &[0.0; 3], 5.0, 2.0);
        assert!(lml.is_finite(), "log_evidence = {lml}");
    }

    #[test]
    fn test_blr_prior_valid() {
        let d = 3;
        let prior = BLRPrior {
            mean: vec![0.0; d],
            cov: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], // identity
            alphas: vec![1.0; d],
        };
        assert!(prior.validate().is_ok());
    }

    #[test]
    fn test_blr_prior_invalid_dimensions() {
        let prior = BLRPrior {
            mean: vec![0.0; 3],
            cov: vec![1.0, 0.0, 0.0, 1.0], // 2×2, should be 3×3
            alphas: vec![1.0; 3],
        };
        assert!(prior.validate().is_err());
    }

    #[test]
    fn test_blr_prior_not_psd() {
        let d = 2;
        let prior = BLRPrior {
            mean: vec![0.0; d],
            cov: vec![-1.0, 0.0, 0.0, -1.0], // negative diagonal → not PSD
            alphas: vec![1.0; d],
        };
        assert!(matches!(prior.validate(), Err(BLRError::SingularMatrix)));
    }

    #[test]
    fn test_fit_with_prior_none_equals_fit() {
        // fit_with_prior(None) must be numerically equivalent to fit().
        let phi: Vec<f64> = vec![1.0, 0.5, 0.25, 2.0, 1.0, 0.5, 0.5, 0.25, 0.125];
        let y: Vec<f64> = vec![1.0, 2.0, 0.5];
        let config = ArdConfig::default();

        let r1 = fit(&phi, &y, 3, 3, &config).unwrap();
        let r2 = fit_with_prior(&phi, &y, 3, 3, &config, None).unwrap();

        // Same number of features.
        assert_eq!(r1.alpha.len(), r2.alpha.len());
        // Alpha values should be essentially identical.
        for (a1, a2) in r1.alpha.iter().zip(r2.alpha.iter()) {
            assert!((a1 - a2).abs() < 1e-10, "alpha mismatch: {a1} vs {a2}");
        }
        assert!((r1.beta - r2.beta).abs() < 1e-10);
    }

    #[test]
    fn test_fit_with_prior_some_compiles_and_runs() {
        let d = 3;
        let phi: Vec<f64> = vec![
            1.0, 0.5, 0.25, 2.0, 1.0, 0.5, 0.5, 0.25, 0.125, 1.5, 0.75, 0.3, 0.8, 0.4, 0.2,
        ];
        let y: Vec<f64> = vec![1.0, 2.0, 0.5, 1.5, 0.8];
        let config = ArdConfig::default();

        let prior = BLRPrior {
            mean: vec![0.5; d],
            cov: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            alphas: vec![0.5; d],
        };
        let result = fit_with_prior(&phi, &y, 5, d, &config, Some(&prior));
        assert!(
            result.is_ok(),
            "fit_with_prior should succeed: {:?}",
            result.err()
        );
        let fitted = result.unwrap();
        assert!(fitted.noise_std() > 0.0);
        assert_eq!(fitted.alpha.len(), d);
    }

    #[test]
    fn test_fit_with_prior_convergence_faster() {
        // With an informed prior the EM loop should need fewer iterations.
        let d = 3;
        let n = 5;
        let phi: Vec<f64> = vec![
            1.0, 0.5, 0.25, 2.0, 1.0, 0.5, 0.5, 0.25, 0.125, 1.5, 0.75, 0.3, 0.8, 0.4, 0.2,
        ];
        let y: Vec<f64> = vec![1.0, 2.0, 0.5, 1.5, 0.8];
        // Use tight tolerance so convergence differences are visible.
        let config = ArdConfig {
            max_iter: 200,
            tol: 1e-9,
            ..ArdConfig::default()
        };

        let baseline = fit_with_prior(&phi, &y, n, d, &config, None).unwrap();

        // Prior centred near the baseline posterior → should converge faster.
        let prior = BLRPrior {
            mean: baseline.posterior.mean.clone(),
            cov: baseline.posterior.cov.clone(),
            alphas: baseline.alpha.clone(),
        };
        let informed = fit_with_prior(&phi, &y, n, d, &config, Some(&prior)).unwrap();

        // Both must produce valid results.
        assert!(informed.noise_std() > 0.0);
        // The informed fit should need ≤ baseline iterations.
        assert!(
            informed.log_evidences.len() <= baseline.log_evidences.len(),
            "informed iterations {} should be <= baseline iterations {}",
            informed.log_evidences.len(),
            baseline.log_evidences.len()
        );
    }
}
