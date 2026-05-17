//! # blr-core: Bayesian Linear Regression with Automatic Relevance Determination
//!
//! A performant, pure-Rust implementation of **Bayesian Linear Regression (BLR)** with
//! **Automatic Relevance Determination (ARD)**, designed for interpretable, sparse modeling
//! in embedded, edge, and WASM environments.
//!
//! ## Part of da-on-demand
//!
//! `blr-core` is the mathematical foundation of the
//! [da-on-demand](https://github.com/finfalter/da-on-demand) project, which ships business
//! logic as portable **WebAssembly Components** that run "where the data is" — on edge
//! devices, embedded controllers, and cloud hosts alike.
//!
//! The dependency chain is:
//!
//! ```text
//! blr-core  (this crate: pure BLR+ARD math)
//!     └── blr-active  (active learning orchestration)
//!             └── sensor-calibration-component  (WASM Component, deployable binary)
//!                         └── drift-detection-component  (companion monitoring component)
//! ```
//!
//! ## Key Features
//!
//! - **Interpretable Sparse Models**: ARD automatically discovers and deactivates irrelevant
//!   features, yielding interpretable coefficients and uncertainty estimates.
//! - **Uncertainty Quantification**: Posterior distribution over weights with epistemic
//!   (model) and aleatoric (noise) uncertainty decomposition.
//! - **Physics-Aware Design**: Sensor calibration and noise estimation modules tailored
//!   for real-world scientific instruments.
//! - **Production-Ready**: No unsafe code in core logic, thoroughly tested, suitable for
//!   WASM compilation and embedded deployment.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use blr_core::{fit, ArdConfig, PredictiveMarginals};
//!
//! // Example: 3 features, 20 training points
//! let phi = vec![1.0; 60]; // N=20, D=3: feature matrix (row-major)
//! let y = vec![0.5; 20];    // target observations
//!
//! let config = ArdConfig {
//!     max_iter: 100,
//!     tol: 1e-5,
//!     ..ArdConfig::default()
//! };
//!
//! let result = fit(&phi, &y, 20, 3, &config).expect("Fit failed");
//!
//! // Predictions with uncertainty
//! let test_phi = vec![1.0; 30]; // 10 test points × 3 features
//! let predictions = result.predict(&test_phi, 10, 3);
//! println!("Mean: {:?}", predictions.mean);
//! println!("Total std: {:?}", predictions.total_std);
//! ```
//!
//! ## Core Algorithm
//!
//! BLR assumes a Gaussian likelihood:
//!
//! $$y_n = \\mathbf{\\phi}_n^T \\mathbf{w} + \\epsilon_n, \\quad \\epsilon_n \\sim \\mathcal{N}(0, \\beta^{-1})$$
//!
//! ARD places a hierarchical prior on weights:
//!
//! $$w_d \\sim \\mathcal{N}(0, \\alpha_d^{-1}), \\quad \\alpha_d \\sim \\text{Gamma}(a_0, b_0)$$
//!
//! The algorithm iteratively refines hyperparameters \\(\\{\\alpha_d, \\beta\\}\\) via
//! empirical Bayes (EM), automatically driving irrelevant \\(\\alpha_d \\to \\infty\\).
//!
//! ## Use Cases
//!
//! - **Sensor Calibration**: Fit nonlinear response models with automatic feature selection
//! - **System Identification**: Discover sparse linear relationships in high-dimensional data
//! - **Edge Inference**: Deploy pretrained models on IoT devices or browser via WASM
//! - **Scientific Computing**: Physics-informed basis functions with interpretable weights
//!
//! ## When NOT to Use blr-core
//!
//! - **Large datasets (N > 10 000)**: Matrix inversion scales O(D³); consider gradient-based
//!   methods for high-feature-count problems.
//! - **Deep non-linear functions**: BLR is a linear model; complex non-linearities require
//!   kernel methods or neural networks.
//! - **Real-time hard-deadline systems**: EM convergence is iterative; worst-case runtime
//!   is bounded by `max_iter` but not deterministic.
//! - **Multi-output regression**: The current implementation is single-output only.
//!
//! ## Modules
//!
//! - [`ard`] — Core BLR+ARD algorithm, model fitting and prediction
//! - [`noise_estimation`] — Automated noise characterization from residuals
//! - [`features`] — Polynomial, RBF, and physics-informed basis functions
//! - [`gaussian`] — Multivariate Gaussian utilities (used internally by `ard`)
//! - [`synthetic_data`] — Benchmark datasets and ground-truth simulators
//! - [`error`] — [`BLRError`] error type
//!
//! Active learning orchestration, precision tiers, and calibration sessions have
//! been moved to the `blr-active` crate for better separation of concerns.
//!
//! ## Feature Flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std`   | ✓       | Enable standard library support (required for faer's thread pool) |
//!
//! ## Performance
//!
//! See the [BENCHMARK_GUIDE.md](https://github.com/finfalter/da-on-demand/blob/main/crates/blr-core/BENCHMARK_GUIDE.md)
//! for detailed performance metrics and how to reproduce them on your hardware.
//!
//! Quick reference on a single core (Intel i7-12700K, `--release`):
//!
//! | Problem | N | D | Fit time |
//! |---------|---|---|----------|
//! | Small   | 30 | 6 | ~3 ms |
//! | Medium  | 60 | 11 | ~12 ms |
//! | Large   | 500 | 30 | ~240 ms |
//!
//! ## Compatibility
//!
//! - **Platforms**: Linux, macOS, Windows, WebAssembly (WASI Preview 2)
//! - **Rust**: 1.70+ (MSRV subject to dependency updates)
//! - **No unsafe**: Core library is 100% safe Rust
//! - **Dependencies**: [`faer`](https://crates.io/crates/faer) for linear algebra
//!
//! ## References
//!
//! - Tipping, M. E. (2001). "Sparse Bayesian learning and the relevance vector machine."
//!   *Journal of Machine Learning Research*, 1, 211–244.
//! - Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
//! - Hennig, P. (2024). *Probabilistic Machine Learning* \[Course\].
//!   University of Tübingen.

pub mod ard;
pub mod error;
pub mod features;
pub mod gaussian;
pub mod noise_estimation;
pub mod synthetic_data;

pub use ard::{fit, fit_with_prior, ArdConfig, BLRPrior, FittedArd, PredictiveMarginals};
pub use error::BLRError;
pub use gaussian::Gaussian;
