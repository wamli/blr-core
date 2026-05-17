//! Sensor noise estimation from a fitted BLR+ARD model.
//!
//! The noise standard deviation is extracted directly from the noise precision
//! hyperparameter β learned during BLR+ARD fitting:
//!
//! $$\sigma_{\text{noise}} = \frac{1}{\sqrt{\beta}}$$
//!
//! This is the standard formulation from Mackay (1992) "Bayesian Interpolation"
//! where β is the precision (inverse variance) of the residual noise.
//!
//! # Example
//!
//! ```rust
//! use blr_core::noise_estimation::{estimate_sensor_noise, estimate_noise_with_confidence};
//!
//! // β = 1.5625 → σ_noise = 0.8
//! let sigma = estimate_sensor_noise(1.5625).unwrap();
//! assert!((sigma - 0.8).abs() < 1e-10);
//!
//! // With confidence interval (n=20 samples, d=3 features)
//! let est = estimate_noise_with_confidence(1.5625, 20, 3);
//! assert_eq!(est.confidence, "preliminary");
//! assert!(est.lower_bound < est.point_estimate);
//! assert!(est.point_estimate < est.upper_bound);
//! ```
//!
//! # References
//! - Mackay, D. J. C. (1992). Bayesian Interpolation. *Neural Computation*, 4(3), 415–447.
//! - Tipping, M. E., & Bishop, C. M. (2001). Sparse Bayesian Learning and the Relevance
//!   Vector Machine. *Journal of Machine Learning Research*, 1, 211–244.

use serde::{Deserialize, Serialize};

// ─── SensorType ───────────────────────────────────────────────────────────────

/// Physical sensing principle that produced the calibration data.
///
/// This is a local enum for use within `blr-core`. It mirrors the variants in
/// `sensor-features::SensorType` and should be kept in sync with that definition.
///
/// **TODO (Phase 2):** Import from `sensor_features` crate once `blr-core` takes
/// it as a dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensorType {
    /// Hall-effect position / magnetic-field sensor.
    Hall,
    /// Capacitive displacement sensor.
    Capacitive,
    /// Metal-oxide (MOX) gas sensor.
    Gas,
    /// Resistance Temperature Detector.
    Rtd,
    /// Thermistor (NTC/PTC).
    Thermistor,
    /// Generic sensor with no physics-specific feature set.
    Generic,
}

// ─── NoiseEstimate ────────────────────────────────────────────────────────────

/// Point estimate and heuristic confidence bounds for sensor noise std.
///
/// The `confidence` field reflects the reliability of the estimate based on the
/// number of samples used for fitting:
/// - `"preliminary"` — fewer than 30 samples; treat with caution.
/// - `"stable"`      — more than 50 samples; estimate is generally reliable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseEstimate {
    /// Best estimate of σ_noise = 1/√β (in the same units as measured output).
    pub point_estimate: f64,
    /// Lower bound (point_estimate × (1 – relative_uncertainty)).
    pub lower_bound: f64,
    /// Upper bound (point_estimate × (1 + relative_uncertainty)).
    pub upper_bound: f64,
    /// Qualitative reliability: `"preliminary"` (n < 30) or `"stable"` (n > 50).
    pub confidence: String,
}

// ─── Public functions ─────────────────────────────────────────────────────────

/// Extract sensor noise standard deviation from the fitted noise precision β.
///
/// # Formula
///
/// $$\sigma_{\text{noise}} = \frac{1}{\sqrt{\beta}}$$
///
/// # Errors
///
/// Returns `Err` if `beta <= 0`, which indicates the model was not fitted or
/// diverged.
///
/// # Example
///
/// ```rust
/// use blr_core::noise_estimation::estimate_sensor_noise;
///
/// let sigma = estimate_sensor_noise(1.5625).unwrap();
/// assert!((sigma - 0.8).abs() < 1e-10);
/// ```
pub fn estimate_sensor_noise(beta: f64) -> Result<f64, String> {
    if beta <= 0.0 {
        return Err(format!(
            "Model not fitted or diverged: beta = {beta:.6} (must be > 0)"
        ));
    }
    Ok(1.0 / beta.sqrt())
}

/// Compute a noise estimate with heuristic confidence bounds.
///
/// Uses a degrees-of-freedom heuristic to approximate uncertainty on σ_noise:
/// the relative uncertainty on β scales roughly as `0.5 / sqrt(n - d)`.
///
/// **Note:** This is a rough approximation. A full Bayesian credible interval
/// on σ would require placing a prior on β and computing the marginal posterior
/// (deferred to Phase 2).
///
/// The `confidence` label is:
/// - `"preliminary"` for n < 30 — treat as a rough guide only.
/// - `"stable"`      for n > 50 — estimate has converged sufficiently.
/// - `"transitional"` for 30 ≤ n ≤ 50.
///
/// # Example
///
/// ```rust
/// use blr_core::noise_estimation::estimate_noise_with_confidence;
///
/// let est = estimate_noise_with_confidence(1.5625, 100, 3);
/// assert_eq!(est.confidence, "stable");
/// assert!(est.lower_bound < est.point_estimate);
/// assert!(est.point_estimate < est.upper_bound);
/// ```
pub fn estimate_noise_with_confidence(
    beta: f64,
    n_samples: usize,
    d_features: usize,
) -> NoiseEstimate {
    // Clamp to avoid sqrt of negative / zero
    let sigma = if beta > 0.0 {
        1.0 / beta.sqrt()
    } else {
        f64::INFINITY
    };

    // Effective degrees of freedom — at least 1 to avoid division by zero
    let dof = (n_samples.saturating_sub(d_features)).max(1) as f64;

    // Rough relative uncertainty shrinks with sqrt(dof)
    let relative_uncertainty = 0.5 / dof.sqrt();

    let lower = sigma * (1.0 - relative_uncertainty).max(0.0);
    let upper = sigma * (1.0 + relative_uncertainty);

    let confidence = if n_samples < 30 {
        "preliminary".to_string()
    } else if n_samples > 50 {
        "stable".to_string()
    } else {
        "transitional".to_string()
    };

    NoiseEstimate {
        point_estimate: sigma,
        lower_bound: lower,
        upper_bound: upper,
        confidence,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_extraction_known_beta() {
        // β = 1.5625 → σ = 1/√1.5625 = 1/1.25 = 0.8
        let sigma = estimate_sensor_noise(1.5625).unwrap();
        assert!((sigma - 0.8).abs() < 1e-10, "expected 0.8, got {sigma}");
    }

    #[test]
    fn test_noise_extraction_beta_one() {
        let sigma = estimate_sensor_noise(1.0).unwrap();
        assert!((sigma - 1.0).abs() < 1e-10, "β=1 → σ=1");
    }

    #[test]
    fn test_noise_extraction_large_beta() {
        // Large β means very small noise
        let sigma = estimate_sensor_noise(10_000.0).unwrap();
        assert!((sigma - 0.01).abs() < 1e-10, "β=10000 → σ=0.01");
    }

    #[test]
    fn test_noise_extraction_small_beta() {
        // Small β means large noise
        let sigma = estimate_sensor_noise(0.04).unwrap();
        assert!((sigma - 5.0).abs() < 1e-10, "β=0.04 → σ=5");
    }

    #[test]
    fn test_noise_extraction_negative_beta() {
        let result = estimate_sensor_noise(-1.0);
        assert!(result.is_err(), "negative beta must return Err");
    }

    #[test]
    fn test_noise_extraction_zero_beta() {
        let result = estimate_sensor_noise(0.0);
        assert!(result.is_err(), "zero beta must return Err");
    }

    #[test]
    fn test_noise_confidence_preliminary() {
        let est = estimate_noise_with_confidence(1.5625, 10, 3);
        assert_eq!(est.confidence, "preliminary", "n=10 → preliminary");
    }

    #[test]
    fn test_noise_confidence_stable() {
        let est = estimate_noise_with_confidence(1.5625, 100, 3);
        assert_eq!(est.confidence, "stable", "n=100 → stable");
    }

    #[test]
    fn test_noise_confidence_transitional() {
        let est = estimate_noise_with_confidence(1.5625, 40, 3);
        assert_eq!(est.confidence, "transitional", "n=40 → transitional");
    }

    #[test]
    fn test_noise_confidence_bounds_ordering() {
        let est = estimate_noise_with_confidence(1.5625, 100, 3);
        assert!(
            est.lower_bound < est.point_estimate,
            "lower_bound must be < point_estimate"
        );
        assert!(
            est.point_estimate < est.upper_bound,
            "point_estimate must be < upper_bound"
        );
    }

    #[test]
    fn test_noise_confidence_point_matches_estimate() {
        let est = estimate_noise_with_confidence(1.5625, 100, 3);
        assert!(
            (est.point_estimate - 0.8).abs() < 1e-10,
            "point_estimate should equal estimate_sensor_noise output"
        );
    }
}
