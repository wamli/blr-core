//! Multivariate Gaussian distribution N(mean, cov).
//!
//! Provides a simple wrapper around a mean vector and a covariance matrix
//! stored as a row-major flat `Vec<f64>`. This module is used internally by
//! [`crate::ard`] to represent the posterior weight distribution; most users
//! interact with it indirectly through [`crate::FittedArd`].
//!
//! ## Overview
//!
//! A D-dimensional Gaussian `N(μ, Σ)` is constructed with:
//!
//! - `mean: Vec<f64>` — length D posterior mean vector μ
//! - `covariance: Vec<f64>` — length D² covariance matrix Σ (row-major)
//!
//! The covariance is stored as a **row-major** D×D flattened vector.
//! All public methods use `Vec<f64>` / `&[f64]` so callers are not forced
//! to depend on `faer`.
//!
//! ## Example
//!
//! ```rust
//! use blr_core::Gaussian;
//!
//! // 2D Gaussian: mean=[1.0, 2.0], covariance=identity
//! let mean = vec![1.0_f64, 2.0];
//! let cov  = vec![1.0, 0.0,   // row 0
//!                 0.0, 1.0];  // row 1
//! let g = Gaussian::new(mean.clone(), cov).expect("valid 2×2 covariance");
//! assert_eq!(g.mean, mean);
//! ```

use faer::linalg::solvers::Solve;
use faer::{Accum, Mat, Par, Side};

use crate::BLRError;

// ── Helper: Cholesky log-determinant ──────────────────────────────────────────

/// Compute log|A| for a symmetric positive-definite matrix via Cholesky.
///
/// Returns `Err(BLRError::SingularMatrix)` if any pivot is non-positive.
pub(crate) fn cholesky_logdet(mat: &Mat<f64>, d: usize) -> Result<f64, BLRError> {
    let mut a = mat.clone();
    for j in 0..d {
        let mut diag = a[(j, j)];
        for k in 0..j {
            let l_jk = a[(j, k)];
            diag -= l_jk * l_jk;
        }
        if diag <= 0.0 {
            return Err(BLRError::SingularMatrix);
        }
        let l_jj = diag.sqrt();
        a[(j, j)] = l_jj;
        for i in (j + 1)..d {
            let mut s = a[(i, j)];
            for k in 0..j {
                s -= a[(i, k)] * a[(j, k)];
            }
            a[(i, j)] = s / l_jj;
        }
    }
    Ok(2.0 * (0..d).map(|j| a[(j, j)].ln()).sum::<f64>())
}

pub struct Gaussian {
    /// Posterior mean — length D.
    pub mean: Vec<f64>,
    /// Posterior covariance, row-major D×D.
    pub cov: Vec<f64>,
    dim: usize,
}

impl Gaussian {
    /// Create a new Gaussian, validating dimensions.
    pub fn new(mean: Vec<f64>, cov: Vec<f64>) -> Result<Self, BLRError> {
        let d = mean.len();
        if cov.len() != d * d {
            return Err(BLRError::DimMismatch {
                expected: d * d,
                got: cov.len(),
            });
        }
        Ok(Self { mean, cov, dim: d })
    }

    /// Dimension D of the distribution.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Per-dimension standard deviations: `sqrt(diag(cov))`.
    pub fn std(&self) -> Vec<f64> {
        let d = self.dim;
        (0..d).map(|i| self.cov[i * d + i].sqrt()).collect()
    }

    /// Marginal distribution at index `i`: returns `(mean[i], std[i])`.
    pub fn marginal(&self, i: usize) -> (f64, f64) {
        let d = self.dim;
        (self.mean[i], self.cov[i * d + i].sqrt())
    }

    /// Log probability density `log N(x; mean, cov)`.
    ///
    /// Uses Cholesky of `cov` for numerical stability.
    pub fn log_pdf(&self, x: &[f64]) -> f64 {
        let d = self.dim;
        debug_assert_eq!(x.len(), d);

        let sigma = Mat::<f64>::from_fn(d, d, |i, j| self.cov[i * d + j]);
        let diff = Mat::<f64>::from_fn(d, 1, |i, _| x[i] - self.mean[i]);

        let llt = sigma
            .llt(Side::Lower)
            .expect("Covariance must be positive-definite for log_pdf");

        // Solve L · L^T · z = diff  →  ||z||^2 = diff^T Σ^{-1} diff
        let z = llt.solve(diff.as_ref());
        let quadratic: f64 = (0..d)
            .map(|i| {
                let v = z[(i, 0)];
                v * v
            })
            .sum();

        // logdet(Σ) via manual Cholesky diagonal (reuse sigma clone)
        let logdet = cholesky_logdet(&sigma, d).expect("Covariance must be PD");

        -0.5 * quadratic - 0.5 * logdet - (d as f64 / 2.0) * (2.0 * std::f64::consts::PI).ln()
    }

    /// Bayesian update: computes `p(self | y)` where `y = A·self + ε`,
    /// `ε ~ N(0, σ²·I_N)` (homoscedastic noise).
    ///
    /// `a` is the measurement matrix `A` (n_obs × d_feat), row-major flat slice.
    /// `noise_variance` is the scalar observation noise variance σ² > 0.
    ///
    /// ## Adaptive dispatch
    ///
    /// Two algebraically equivalent forms are available; this method selects
    /// whichever minimises the size of the required Cholesky factorisation:
    ///
    /// | Condition | Form chosen | Cholesky size |
    /// |-----------|-------------|---------------|
    /// | `n_obs < d_feat`  | Gram / Kalman-gain (observation-space) | N×N |
    /// | `n_obs >= d_feat` | Precision / Woodbury (parameter-space)  | D×D |
    ///
    /// The two forms are related by the Woodbury matrix identity; see
    /// `dev/blog/blr-and-ard.md` Appendix A for the derivation.
    /// The precision form derives Σ_prior⁻¹ directly from `self.cov` — no
    /// isotropic approximation is made, and the forms agree within floating-point
    /// rounding error.
    ///
    /// Returns the updated Gaussian representing the posterior distribution.
    pub fn condition(
        self,
        a: &[f64],
        n_obs: usize,
        d_feat: usize,
        y: &[f64],
        noise_variance: f64,
    ) -> Result<Self, BLRError> {
        debug_assert_eq!(a.len(), n_obs * d_feat);
        debug_assert_eq!(y.len(), n_obs);
        debug_assert!(noise_variance > 0.0, "noise_variance must be positive");
        // Private helpers derive D from self.dim; assert caller agrees (DD-B).
        debug_assert_eq!(self.dim, d_feat, "d_feat must equal Gaussian dimension");

        if n_obs < d_feat {
            self.condition_gram_form(a, n_obs, y, noise_variance)
        } else {
            self.condition_precision_form(a, n_obs, y, noise_variance)
        }
    }

    /// (internal) Observation-space (N×N Gram / Kalman-gain) form of condition().
    /// Cheaper when n_obs < d_feat (N×N Cholesky vs D×D).
    fn condition_gram_form(
        self,
        a: &[f64],
        n_obs: usize,
        y: &[f64],
        noise_variance: f64,
    ) -> Result<Self, BLRError> {
        let d = self.dim;

        let a_mat = Mat::<f64>::from_fn(n_obs, d, |i, j| a[i * d + j]);
        let mu_mat = Mat::<f64>::from_fn(d, 1, |i, _| self.mean[i]);
        let sigma_mat = Mat::<f64>::from_fn(d, d, |i, j| self.cov[i * d + j]);

        // Gram = A Σ A^T + σ²·I_N  (N×N)
        let a_sigma_t = {
            // A Σ = (N×D) * (D×D) → N×D
            let mut tmp = Mat::<f64>::zeros(n_obs, d);
            faer::linalg::matmul::matmul(
                tmp.as_mut(),
                Accum::Replace,
                a_mat.as_ref(),
                sigma_mat.as_ref(),
                1.0_f64,
                Par::Seq,
            );
            tmp
        };
        let mut gram = {
            // A_sigma * A^T  (N×D) * (D×N) → N×N
            // i.e. A Σ A^T
            let mut tmp = Mat::<f64>::zeros(n_obs, n_obs);
            faer::linalg::matmul::matmul(
                tmp.as_mut(),
                Accum::Replace,
                a_sigma_t.as_ref(),
                a_mat.as_ref().transpose(),
                1.0_f64,
                Par::Seq,
            );
            tmp
        };
        // Add σ²·I_N to gram
        for i in 0..n_obs {
            gram[(i, i)] += noise_variance;
        }

        let llt_gram = gram
            .llt(Side::Lower)
            .map_err(|_| BLRError::SingularMatrix)?;

        // sigma_at = Σ A^T  (D×N)
        let sigma_at = {
            let mut tmp = Mat::<f64>::zeros(d, n_obs);
            faer::linalg::matmul::matmul(
                tmp.as_mut(),
                Accum::Replace,
                sigma_mat.as_ref(),
                a_mat.as_ref().transpose(),
                1.0_f64,
                Par::Seq,
            );
            tmp
        };

        // residual = y - A μ  (N)
        let a_mu = {
            let mut tmp = Mat::<f64>::zeros(n_obs, 1);
            faer::linalg::matmul::matmul(
                tmp.as_mut(),
                Accum::Replace,
                a_mat.as_ref(),
                mu_mat.as_ref(),
                1.0_f64,
                Par::Seq,
            );
            tmp
        };
        let residual_mat = Mat::<f64>::from_fn(n_obs, 1, |i, _| y[i] - a_mu[(i, 0)]);

        // Solve Gram * Z = sigma_at^T  →  Z is N×D
        let z = llt_gram.solve(sigma_at.as_ref().transpose());

        // mu' = mu + sigma_at * Gram^{-1} * residual = mu + Z^T * residual
        let delta_mu = {
            let mut tmp = Mat::<f64>::zeros(d, 1);
            faer::linalg::matmul::matmul(
                tmp.as_mut(),
                Accum::Replace,
                z.as_ref().transpose(),
                residual_mat.as_ref(),
                1.0_f64,
                Par::Seq,
            );
            tmp
        };

        // Sigma' = Sigma - sigma_at * Z  (D×D)
        let mut sigma_new_mat = sigma_mat.clone();
        faer::linalg::matmul::matmul(
            sigma_new_mat.as_mut(),
            Accum::Add,
            sigma_at.as_ref(),
            z.as_ref(),
            -1.0_f64,
            Par::Seq,
        );

        let sigma_new_ref = sigma_new_mat.as_ref();
        let new_mean: Vec<f64> = (0..d).map(|i| self.mean[i] + delta_mu[(i, 0)]).collect();
        let new_cov: Vec<f64> = (0..d)
            .flat_map(|i| (0..d).map(move |j| sigma_new_ref[(i, j)]))
            .collect();

        Gaussian::new(new_mean, new_cov)
    }

    /// (internal) Parameter-space (D×D precision / Woodbury) form of condition().
    /// Cheaper when n_obs >= d_feat (D×D Cholesky vs N×N).
    /// Derives Σ_prior⁻¹ exactly from self.cov — algebraically equivalent to
    /// condition_gram_form() via the Woodbury matrix identity.
    fn condition_precision_form(
        self,
        a: &[f64],
        n_obs: usize,
        y: &[f64],
        noise_variance: f64,
    ) -> Result<Self, BLRError> {
        let d = self.dim; // canonical D; public condition() asserts self.dim == d_feat
        let beta = 1.0 / noise_variance;

        let a_mat = Mat::<f64>::from_fn(n_obs, d, |i, j| a[i * d + j]);
        let y_mat = Mat::<f64>::from_fn(n_obs, 1, |i, _| y[i]);
        let sigma_prior = Mat::<f64>::from_fn(d, d, |i, j| self.cov[i * d + j]);
        let mu_prior = Mat::<f64>::from_fn(d, 1, |i, _| self.mean[i]);

        // Step 1: Cholesky of Σ_prior → Σ_prior⁻¹ and Σ_prior⁻¹·μ_prior
        let llt_prior = sigma_prior
            .llt(Side::Lower)
            .map_err(|_| BLRError::SingularMatrix)?;
        let eye_d = Mat::<f64>::identity(d, d);
        let sigma_prior_inv = llt_prior.solve(eye_d.as_ref()); // D×D
        let prec_mu_prior = llt_prior.solve(mu_prior.as_ref()); // D×1

        // Step 2: A^T A  (D×D)
        let mut at_a = Mat::<f64>::zeros(d, d);
        faer::linalg::matmul::matmul(
            at_a.as_mut(),
            Accum::Replace,
            a_mat.as_ref().transpose(),
            a_mat.as_ref(),
            1.0_f64,
            Par::Seq,
        );

        // Step 3: Precision matrix P = Σ_prior⁻¹ + β·(A^T A)  (D×D)
        //         Exact Woodbury form — no isotropic approximation.
        let mut precision = sigma_prior_inv;
        for i in 0..d {
            for j in 0..d {
                precision[(i, j)] += beta * at_a[(i, j)];
            }
        }

        // Step 4: Cholesky of P
        let llt_post = precision
            .llt(Side::Lower)
            .map_err(|_| BLRError::SingularMatrix)?;

        // Step 5: Σ_post = P⁻¹  (solve with identity)
        let sigma_post = llt_post.solve(eye_d.as_ref());

        // Step 6: RHS = Σ_prior⁻¹·μ_prior + β·A^T·y
        let mut at_y = Mat::<f64>::zeros(d, 1);
        faer::linalg::matmul::matmul(
            at_y.as_mut(),
            Accum::Replace,
            a_mat.as_ref().transpose(),
            y_mat.as_ref(),
            1.0_f64,
            Par::Seq,
        );
        let rhs = Mat::<f64>::from_fn(d, 1, |i, _| prec_mu_prior[(i, 0)] + beta * at_y[(i, 0)]);

        // Step 7: μ_post = Σ_post·rhs
        let mu_post = llt_post.solve(rhs.as_ref());

        let new_mean: Vec<f64> = (0..d).map(|i| mu_post[(i, 0)]).collect();
        let sigma_ref = sigma_post.as_ref();
        let new_cov: Vec<f64> = (0..d)
            .flat_map(|i| (0..d).map(move |j| sigma_ref[(i, j)]))
            .collect();

        Gaussian::new(new_mean, new_cov)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_std_known() {
        // 2D Gaussian with cov = [[4, 0], [0, 9]]
        let g = Gaussian::new(vec![0.0, 0.0], vec![4.0, 0.0, 0.0, 9.0]).unwrap();
        let std = g.std();
        let tol = 1e-10;
        assert!((std[0] - 2.0).abs() < tol, "std[0]={}", std[0]);
        assert!((std[1] - 3.0).abs() < tol, "std[1]={}", std[1]);
    }

    #[test]
    fn test_log_pdf_standard_normal() {
        // log N(0; 0, I) = -D/2 * log(2*pi)
        let d = 3usize;
        let cov: Vec<f64> = (0..d * d)
            .map(|k| if k % (d + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let g = Gaussian::new(vec![0.0; d], cov).unwrap();
        let lp = g.log_pdf(&vec![0.0; d]);
        let expected = -(d as f64) / 2.0 * (2.0 * std::f64::consts::PI).ln();
        assert!(
            (lp - expected).abs() < 1e-10,
            "log_pdf={lp:.6}, expected={expected:.6}"
        );
    }

    #[test]
    fn test_marginal() {
        // 3D Gaussian with cov = diag([1,4,9])
        let cov = vec![1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 9.0];
        let g = Gaussian::new(vec![1.0, 2.0, 3.0], cov).unwrap();
        let (m, s) = g.marginal(1);
        assert!((m - 2.0).abs() < 1e-10);
        assert!((s - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_dim_mismatch() {
        let result = Gaussian::new(vec![0.0; 3], vec![1.0; 4]);
        assert!(result.is_err());
    }

    // ── condition() tests ─────────────────────────────────────────────────────

    #[test]
    fn test_condition_gram_form_analytic_2d() {
        // μ_prior=0, Σ_prior=I₂, A=I₂, σ²=1, y=[1,2]
        // → Σ_post = 0.5·I₂, μ_post = [0.5, 1.0]
        let g = Gaussian::new(vec![0.0, 0.0], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let a = vec![1.0_f64, 0.0, 0.0, 1.0]; // 2×2 identity row-major
        let y = vec![1.0_f64, 2.0];
        let post = g.condition_gram_form(&a, 2, &y, 1.0).unwrap();
        let tol = 1e-12;
        assert!((post.mean[0] - 0.5).abs() < tol, "mean[0]={}", post.mean[0]);
        assert!((post.mean[1] - 1.0).abs() < tol, "mean[1]={}", post.mean[1]);
        assert!((post.cov[0] - 0.5).abs() < tol, "cov[0,0]={}", post.cov[0]);
        assert!((post.cov[1]).abs() < tol, "cov[0,1]={}", post.cov[1]);
        assert!((post.cov[2]).abs() < tol, "cov[1,0]={}", post.cov[2]);
        assert!((post.cov[3] - 0.5).abs() < tol, "cov[1,1]={}", post.cov[3]);
    }

    #[test]
    fn test_condition_precision_form_analytic_2d() {
        // Same analytic case as gram test — both forms must agree.
        let g = Gaussian::new(vec![0.0, 0.0], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let a = vec![1.0_f64, 0.0, 0.0, 1.0];
        let y = vec![1.0_f64, 2.0];
        let post = g.condition_precision_form(&a, 2, &y, 1.0).unwrap();
        let tol = 1e-12;
        assert!((post.mean[0] - 0.5).abs() < tol, "mean[0]={}", post.mean[0]);
        assert!((post.mean[1] - 1.0).abs() < tol, "mean[1]={}", post.mean[1]);
        assert!((post.cov[0] - 0.5).abs() < tol, "cov[0,0]={}", post.cov[0]);
        assert!((post.cov[1]).abs() < tol, "cov[0,1]={}", post.cov[1]);
        assert!((post.cov[2]).abs() < tol, "cov[1,0]={}", post.cov[2]);
        assert!((post.cov[3] - 0.5).abs() < tol, "cov[1,1]={}", post.cov[3]);
    }

    #[test]
    fn test_condition_parity_n8_d6() {
        let n = 8usize;
        let d = 6usize;
        // Deterministic synthetic A (n×d) — values spread in (-1, 1)
        let a: Vec<f64> = (0..n * d)
            .map(|k| {
                let seed = (k as f64 * 0.3141592653589793).sin();
                seed * 0.5 // scale to keep conditioning reasonable
            })
            .collect();
        // Synthetic y
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).cos()).collect();
        // Prior: identity covariance, zero mean
        let cov_prior: Vec<f64> = (0..d * d)
            .map(|k| if k % (d + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let noise_variance = 0.5_f64;

        let g_gram = Gaussian::new(vec![0.0; d], cov_prior.clone()).unwrap();
        let g_prec = Gaussian::new(vec![0.0; d], cov_prior).unwrap();

        let post_gram = g_gram
            .condition_gram_form(&a, n, &y, noise_variance)
            .unwrap();
        let post_prec = g_prec
            .condition_precision_form(&a, n, &y, noise_variance)
            .unwrap();

        let tol = 1e-10;
        for i in 0..d {
            assert!(
                (post_gram.mean[i] - post_prec.mean[i]).abs() < tol,
                "mean[{}]: gram={}, prec={}",
                i,
                post_gram.mean[i],
                post_prec.mean[i]
            );
        }
        for k in 0..d * d {
            assert!(
                (post_gram.cov[k] - post_prec.cov[k]).abs() < tol,
                "cov[{}]: gram={}, prec={}",
                k,
                post_gram.cov[k],
                post_prec.cov[k]
            );
        }
    }

    #[test]
    fn test_condition_dispatch_n_lt_d() {
        // N=3 < D=10 → should dispatch to gram form
        let n = 3usize;
        let d = 10usize;
        let a: Vec<f64> = (0..n * d).map(|k| (k as f64 * 0.17).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
        let cov: Vec<f64> = (0..d * d)
            .map(|k| if k % (d + 1) == 0 { 2.0 } else { 0.0 })
            .collect();
        let mean = vec![0.5_f64; d];
        let noise_variance = 0.3_f64;

        let g1 = Gaussian::new(mean.clone(), cov.clone()).unwrap();
        let g2 = Gaussian::new(mean, cov).unwrap();

        let post_dispatch = g1.condition(&a, n, d, &y, noise_variance).unwrap();
        let post_gram = g2.condition_gram_form(&a, n, &y, noise_variance).unwrap();

        for i in 0..d {
            assert_eq!(
                post_dispatch.mean[i], post_gram.mean[i],
                "mean[{}] mismatch — dispatch did not route to gram form",
                i
            );
        }
    }

    #[test]
    fn test_condition_dispatch_n_gt_d() {
        // N=100 >= D=6 → should dispatch to precision form
        let n = 100usize;
        let d = 6usize;
        let a: Vec<f64> = (0..n * d).map(|k| (k as f64 * 0.13).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.23).cos()).collect();
        let cov: Vec<f64> = (0..d * d)
            .map(|k| if k % (d + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let mean = vec![0.0_f64; d];
        let noise_variance = 1.0_f64;

        let g1 = Gaussian::new(mean.clone(), cov.clone()).unwrap();
        let g2 = Gaussian::new(mean, cov).unwrap();

        let post_dispatch = g1.condition(&a, n, d, &y, noise_variance).unwrap();
        let post_prec = g2
            .condition_precision_form(&a, n, &y, noise_variance)
            .unwrap();

        for i in 0..d {
            assert_eq!(
                post_dispatch.mean[i], post_prec.mean[i],
                "mean[{}] mismatch — dispatch did not route to precision form",
                i
            );
        }
    }
}
