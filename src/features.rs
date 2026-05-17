//! Polynomial and RBF basis-function constructors for BLR feature engineering.
//!
//! Each function accepts a 1-D input slice `x` (length N) and returns
//! `(matrix, ncols)` where `matrix` is the N×ncols feature matrix in
//! **row-major** order (same layout as the `phi` argument to [`crate::fit`]).
//!
//! ## Row-major layout
//!
//! The feature matrix is stored as a flat `Vec<f64>` where element
//! `matrix[i * ncols + j]` is the value of feature `j` for observation `i`.
//! This matches the convention expected by [`crate::fit`].
//!
//! ## Example
//!
//! ```rust
//! use blr_core::features::{polynomial, rbf};
//!
//! // 3 observations, polynomial degree 2 → 3 columns [1, x, x²]
//! let x = [0.0_f64, 1.0, 2.0];
//! let (mat, ncols) = polynomial(&x, 2);
//! assert_eq!(ncols, 3);
//! assert_eq!(mat.len(), 3 * 3);
//! // Row 2 (x=2): [1.0, 2.0, 4.0]
//! assert!((mat[2 * 3 + 0] - 1.0).abs() < 1e-12);
//! assert!((mat[2 * 3 + 1] - 2.0).abs() < 1e-12);
//! assert!((mat[2 * 3 + 2] - 4.0).abs() < 1e-12);
//! ```
//!
//! ## Performance
//!
//! All constructors allocate a single `Vec<f64>` of size N×ncols.
//! Construction is O(N×D) time and memory. For typical sensor calibration
//! (N ≤ 200, D ≤ 20) this is negligible.
//!
//! ## References
//!
//! - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 3.
//! - For physics-informed sensor features, see the `sensor-features` crate.

use std::f64::consts::PI;

/// Polynomial feature map: `[1, x, x², ..., x^degree]`.
///
/// Returns `(mat, ncols)` where `ncols = degree + 1`.
///
/// # Example
/// ```
/// use blr_core::features::polynomial;
/// let (mat, ncols) = polynomial(&[0.0, 1.0, 2.0], 2);
/// assert_eq!(ncols, 3);
/// // row 2: [1.0, 2.0, 4.0]
/// assert!((mat[2 * 3 + 2] - 4.0).abs() < 1e-12);
/// ```
pub fn polynomial(x: &[f64], degree: usize) -> (Vec<f64>, usize) {
    let n = x.len();
    let ncols = degree + 1;
    let mut mat = vec![0.0_f64; n * ncols];
    for i in 0..n {
        for p in 0..ncols {
            mat[i * ncols + p] = x[i].powi(p as i32);
        }
    }
    (mat, ncols)
}

/// Radial Basis Function (RBF / Gaussian) feature map.
///
/// Entry `(i, j) = exp(-0.5 * ((x[i] - centers[j]) / width)²)`.
///
/// Returns `(mat, ncols)` where `ncols = centers.len()`.
pub fn rbf(x: &[f64], centers: &[f64], width: f64) -> (Vec<f64>, usize) {
    let n = x.len();
    let ncols = centers.len();
    let mut mat = vec![0.0_f64; n * ncols];
    for i in 0..n {
        for j in 0..ncols {
            let diff = (x[i] - centers[j]) / width;
            mat[i * ncols + j] = (-0.5 * diff * diff).exp();
        }
    }
    (mat, ncols)
}

/// Trigonometric (sine/cosine) feature map.
///
/// Columns: `[1, sin(πx), cos(πx), sin(2πx), cos(2πx), ..., sin(n·πx), cos(n·πx)]`.
///
/// Returns `(mat, ncols)` where `ncols = 2 * n_freq + 1`.
///
/// **Note:** uses `i * π * x` (not `i * 2π * x`) matching the Python
/// `trig_features` reference implementation.
pub fn trig(x: &[f64], n_freq: usize) -> (Vec<f64>, usize) {
    let n = x.len();
    let ncols = 2 * n_freq + 1;
    let mut mat = vec![0.0_f64; n * ncols];
    for i in 0..n {
        mat[i * ncols] = 1.0; // constant column
        for freq in 1..=n_freq {
            let arg = (freq as f64) * PI * x[i];
            mat[i * ncols + 2 * freq - 1] = arg.sin();
            mat[i * ncols + 2 * freq] = arg.cos();
        }
    }
    (mat, ncols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_parity() {
        // degree=3 on [0.5, 1.0, 1.5]
        // Row 0: [1.0, 0.5, 0.25, 0.125]
        let (mat, ncols) = polynomial(&[0.5, 1.0, 1.5], 3);
        assert_eq!(ncols, 4);
        let row0 = &mat[..4];
        let expected = [1.0_f64, 0.5, 0.25, 0.125];
        for (a, b) in row0.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-10, "poly row0 mismatch: {a} vs {b}");
        }
        // Row 1: [1.0, 1.0, 1.0, 1.0]
        let row1 = &mat[4..8];
        for v in row1 {
            assert!((v - 1.0).abs() < 1e-10);
        }
        // Row 2: [1.0, 1.5, 2.25, 3.375]
        let row2 = &mat[8..12];
        let expected2 = [1.0_f64, 1.5, 2.25, 3.375];
        for (a, b) in row2.iter().zip(expected2.iter()) {
            assert!((a - b).abs() < 1e-10, "poly row2 mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_rbf_parity() {
        // x=[0.0], centers=[0.0,1.0], width=1.0
        let (mat, ncols) = rbf(&[0.0], &[0.0, 1.0], 1.0);
        assert_eq!(ncols, 2);
        assert!((mat[0] - 1.0).abs() < 1e-10);
        assert!((mat[1] - (-0.5_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_trig_parity() {
        // x=[1.0], n_freq=2 → [1, sin(π), cos(π), sin(2π), cos(2π)]
        let (mat, ncols) = trig(&[1.0], 2);
        assert_eq!(ncols, 5);
        let expected = [
            1.0_f64,
            (PI).sin(),       // ≈ 0
            (PI).cos(),       // ≈ -1
            (2.0 * PI).sin(), // ≈ 0
            (2.0 * PI).cos(), // ≈ 1
        ];
        for (i, (a, b)) in mat.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-10, "trig col {i}: {a} vs {b}");
        }
    }

    #[test]
    fn test_trig_ncols() {
        let (_, ncols) = trig(&[0.0; 10], 5);
        assert_eq!(ncols, 11);
    }
}
