//! Synthetic Hall-effect sensor data for the end-to-end calibration demo.
//!
//! Generates deterministic, reproducible sensor measurements using a seeded
//! xorshift64 PRNG (no platform-specific dependencies; identical output on any target
//! including `wasm32-wasip2`).
//!
//! ## Hall Sensor Model
//!
//! ```text
//! V(B) = K0 + K1 × B_norm + ε        ε ~ N(0, σ_noise²)
//! ```
//!
//! where `B_norm = B / B_MAX_MT ∈ [0, 1]`, `K0 = 0.5 V` (offset),
//! `K1 = 0.5 V` (sensitivity), and `B_MAX_MT = 100 mT`.
//!
//! The true model is **linear** in B. Features [B², B³] are irrelevant; ARD
//! correctly assigns them large precision (α → large), demonstrating sparsification.
//!
//! ## Features
//!
//! The BLR feature map is degree-3 polynomial:
//!
//! ```text
//! φ(B) = [1, B_norm, B_norm², B_norm³]
//! ```
//!
//! ARD learns:
//! - α\[0\] (bias): moderate — captures voltage offset K0.
//! - α\[1\] (B-field): very small — highly relevant linear term.
//! - α\[2\] (B²): large — irrelevant quadratic (no quadratic signal).
//! - α\[3\] (B³): very large — irrelevant cubic (no cubic signal).
//!
//! Ratio `α[3] / α[1] > 100×` is expected and validates physics correctness.
//!
//! ## Usage
//!
//! ```rust
//! use blr_core::synthetic_data::{generate_hall_samples, hall_feature_fn, GROUND_TRUTH_NOISE_STD};
//!
//! let (b_vals, v_vals) = generate_hall_samples(10, 0.0, 100.0, GROUND_TRUTH_NOISE_STD, 42);
//! assert_eq!(b_vals.len(), 10);
//! assert_eq!(v_vals.len(), 10);
//!
//! let feats = hall_feature_fn(50.0);  // B = 50 mT
//! assert!((feats[0] - 1.0).abs() < 1e-12);  // bias
//! assert!((feats[1] - 0.5).abs() < 1e-12);  // B_norm = 50/100
//! ```

use core::f64::consts::PI;

// ─── Physics constants ─────────────────────────────────────────────────────────

/// Nominal bias voltage (Hall sensor output at zero B-field). Units: V.
pub const K0: f64 = 0.5;

/// Hall sensitivity gain (voltage change per normalized B-field unit). Units: V.
pub const K1: f64 = 0.5;

/// Maximum B-field used to normalize features. Units: mT.
pub const B_MAX_MT: f64 = 100.0;

/// Ground-truth sensor noise standard deviation used in synthetic data generation. Units: V.
///
/// Represents realistic Hall-sensor measurement noise at 10 kHz bandwidth.
pub const GROUND_TRUTH_NOISE_STD: f64 = 0.008;

/// Number of BLR input features: `[1, B_norm, B_norm², B_norm³]`.
pub const N_FEATURES: usize = 4;

/// Human-readable names for each BLR feature (index-aligned with [`hall_feature_fn`]).
pub const FEATURE_NAMES: [&str; N_FEATURES] = ["bias", "B-field", "B-field²", "B-field³"];

// ─── Seeded PRNG (xorshift64) ──────────────────────────────────────────────────

/// Minimal seeded pseudo-random number generator.
///
/// Uses xorshift64: no heap allocations, no platform-specific dependencies,
/// and produces identical output on `wasm32-wasip2` and `x86_64`.
///
/// NOT cryptographically secure — only for reproducible synthetic data.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create a new RNG seeded with `seed`.
    ///
    /// The seed is mixed with a constant to avoid degenerate all-zero state.
    /// Eight warm-up steps eliminate any bias from the initial seed mix.
    pub fn new(seed: u64) -> Self {
        let state = seed ^ 0x6C62_272E_07BB_0142;
        let state = if state == 0 {
            0x6C62_272E_07BB_0142
        } else {
            state
        };
        let mut rng = Self { state };
        for _ in 0..8 {
            rng.next_u64();
        }
        rng
    }

    /// Advance state and return the next u64.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform sample in `[min, max)`.
    pub fn uniform(&mut self, min: f64, max: f64) -> f64 {
        // Map 53-bit mantissa integer to [0.0, 1.0)
        let bits = self.next_u64() >> 11;
        let u = bits as f64 * (1.0 / 9_007_199_254_740_992.0_f64);
        min + u * (max - min)
    }

    /// Standard normal sample via Box-Muller transform.
    ///
    /// Avoids `ln(0)` by clamping the uniform draw from below.
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform(1e-15, 1.0);
        let u2 = self.uniform(0.0, 2.0 * PI);
        (-2.0 * u1.ln()).sqrt() * u2.cos()
    }
}

// ─── Feature function ──────────────────────────────────────────────────────────

/// Map a B-field value to the BLR feature vector.
///
/// Returns `[1.0, B_norm, B_norm², B_norm³]` where `B_norm = b_mt / B_MAX_MT`.
///
/// # Arguments
/// - `b_mt` — B-field value in milli-Tesla.
///
/// # Example
/// ```rust
/// use blr_core::synthetic_data::hall_feature_fn;
/// let phi = hall_feature_fn(50.0);
/// assert!((phi[0] - 1.0).abs() < 1e-14);  // bias = 1
/// assert!((phi[1] - 0.5).abs() < 1e-14);  // B_norm = 0.5
/// assert!((phi[2] - 0.25).abs() < 1e-14); // B_norm² = 0.25
/// assert!((phi[3] - 0.125).abs() < 1e-14);// B_norm³ = 0.125
/// ```
pub fn hall_feature_fn(b_mt: f64) -> Vec<f64> {
    let b = b_mt / B_MAX_MT;
    vec![1.0, b, b * b, b * b * b]
}

// ─── Hall model ────────────────────────────────────────────────────────────────

/// True (noiseless) Hall sensor voltage at B-field `b_mt` mT.
///
/// `V_true(B) = K0 + K1 × (B / B_MAX_MT)`
pub fn hall_voltage_true(b_mt: f64) -> f64 {
    K0 + K1 * (b_mt / B_MAX_MT)
}

// ─── Data generation ───────────────────────────────────────────────────────────

/// Generate `n` synthetic Hall-sensor measurements with Gaussian noise.
///
/// # Arguments
/// - `n`         — number of (B, V) pairs to generate.
/// - `b_min_mt`  — lower bound of B-field range in mT.
/// - `b_max_mt`  — upper bound of B-field range in mT.
/// - `noise_std` — Gaussian noise standard deviation (V).
/// - `seed`      — RNG seed for deterministic reproducibility.
///
/// # Returns
/// `(b_values, voltages)` — each a `Vec<f64>` of length `n`.
///
/// B-field values are sampled uniformly in `[b_min_mt, b_max_mt)`;
/// voltages follow `V = K0 + K1×B_norm + N(0, noise_std²)`.
///
/// # Example
/// ```rust
/// use blr_core::synthetic_data::{generate_hall_samples, GROUND_TRUTH_NOISE_STD};
///
/// let (bs, vs) = generate_hall_samples(20, 0.0, 100.0, GROUND_TRUTH_NOISE_STD, 0);
/// assert_eq!(bs.len(), 20);
/// // Voltages should be near K0 + K1*B_norm ± 3σ
/// for (&b, &v) in bs.iter().zip(vs.iter()) {
///     let v_true = 0.5 + 0.5 * (b / 100.0);
///     assert!((v - v_true).abs() < 0.05, "voltage too far from truth");
/// }
/// ```
pub fn generate_hall_samples(
    n: usize,
    b_min_mt: f64,
    b_max_mt: f64,
    noise_std: f64,
    seed: u64,
) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let mut b_values = Vec::with_capacity(n);
    let mut voltages = Vec::with_capacity(n);

    for _ in 0..n {
        let b = rng.uniform(b_min_mt, b_max_mt);
        let v = hall_voltage_true(b) + noise_std * rng.normal();
        b_values.push(b);
        voltages.push(v);
    }

    (b_values, voltages)
}

/// Build a row-major feature matrix φ (N×D) from B-field values.
///
/// Each row `i` is `hall_feature_fn(b_vals[i])`.
/// Layout: `phi[i * D + j]` = feature `j` for sample `i`.
pub fn build_phi(b_vals: &[f64]) -> Vec<f64> {
    let n = b_vals.len();
    let mut phi = Vec::with_capacity(n * N_FEATURES);
    for &b in b_vals {
        phi.extend_from_slice(&hall_feature_fn(b));
    }
    phi
}

// ─── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_deterministic() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_u64(), r2.next_u64());
        }
    }

    #[test]
    fn test_rng_different_seeds() {
        let mut r1 = Rng::new(1);
        let mut r2 = Rng::new(2);
        // Very unlikely to be equal for first 10 samples
        let same = (0..10).all(|_| r1.next_u64() == r2.next_u64());
        assert!(!same, "Different seeds produced identical output");
    }

    #[test]
    fn test_hall_feature_fn() {
        let phi = hall_feature_fn(0.0);
        assert_eq!(phi.len(), N_FEATURES);
        assert!((phi[0] - 1.0).abs() < 1e-14);
        assert!((phi[1] - 0.0).abs() < 1e-14);
        assert!((phi[2] - 0.0).abs() < 1e-14);
        assert!((phi[3] - 0.0).abs() < 1e-14);

        let phi100 = hall_feature_fn(100.0);
        assert!((phi100[1] - 1.0).abs() < 1e-14);
        assert!((phi100[2] - 1.0).abs() < 1e-14);
        assert!((phi100[3] - 1.0).abs() < 1e-14);

        let phi50 = hall_feature_fn(50.0);
        assert!((phi50[1] - 0.5).abs() < 1e-14);
        assert!((phi50[2] - 0.25).abs() < 1e-14);
        assert!((phi50[3] - 0.125).abs() < 1e-14);
    }

    #[test]
    fn test_hall_voltage_true_range() {
        // V(0mT) = K0 = 0.5V
        assert!((hall_voltage_true(0.0) - 0.5).abs() < 1e-14);
        // V(100mT) = K0 + K1 = 1.0V
        assert!((hall_voltage_true(100.0) - 1.0).abs() < 1e-14);
        // V(50mT) = 0.75V
        assert!((hall_voltage_true(50.0) - 0.75).abs() < 1e-14);
    }

    #[test]
    fn test_generate_hall_samples_length() {
        let (bs, vs) = generate_hall_samples(20, 0.0, 100.0, GROUND_TRUTH_NOISE_STD, 42);
        assert_eq!(bs.len(), 20);
        assert_eq!(vs.len(), 20);
    }

    #[test]
    fn test_generate_hall_samples_deterministic() {
        let (bs1, vs1) = generate_hall_samples(10, 0.0, 100.0, GROUND_TRUTH_NOISE_STD, 99);
        let (bs2, vs2) = generate_hall_samples(10, 0.0, 100.0, GROUND_TRUTH_NOISE_STD, 99);
        assert_eq!(bs1, bs2);
        assert_eq!(vs1, vs2);
    }

    #[test]
    fn test_generate_hall_samples_range() {
        let (bs, _vs) = generate_hall_samples(100, 0.0, 100.0, GROUND_TRUTH_NOISE_STD, 7);
        for b in &bs {
            assert!(*b >= 0.0 && *b < 100.0, "B out of range: {b}");
        }
    }

    #[test]
    fn test_generate_hall_samples_close_to_truth() {
        let (bs, vs) = generate_hall_samples(1000, 0.0, 100.0, GROUND_TRUTH_NOISE_STD, 42);
        let mut max_err = 0.0_f64;
        for (&b, &v) in bs.iter().zip(vs.iter()) {
            let err = (v - hall_voltage_true(b)).abs();
            if err > max_err {
                max_err = err;
            }
        }
        // 6-sigma bound: max_err < 6 * 0.008 = 0.048V (essentially certain for N=1000)
        assert!(
            max_err < 0.1,
            "Max error too large: {max_err:.4} (expected < 0.1 for noise_std=0.008)"
        );
    }

    #[test]
    fn test_build_phi_shape() {
        let bs = vec![0.0, 50.0, 100.0];
        let phi = build_phi(&bs);
        assert_eq!(phi.len(), 3 * N_FEATURES);
        // Row 1 (B=50mT): [1, 0.5, 0.25, 0.125]
        assert!((phi[N_FEATURES + 1] - 0.5).abs() < 1e-12);
    }
}
