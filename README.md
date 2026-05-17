# blr-core: Bayesian Linear Regression (BLR) with Automatic Relevance Detection (ARD)

**Pure-Rust BLR+ARD curve fitting for interpretable, sparse modeling at the edge.**

[![Crates.io](https://img.shields.io/crates/v/blr-core.svg)](https://crates.io/crates/blr-core)
[![Docs](https://img.shields.io/docsrs/blr-core/latest)](https://docs.rs/blr-core/)
[![License](https://img.shields.io/crates/l/blr-core.svg)](https://github.com/wamli/blr-core/blob/main/LICENSE)

## What Is blr-core?

`blr-core` is a production-ready **Bayesian Linear Regression (BLR) engine with Automatic Relevance Determination (ARD)** — a statistically principled approach to sparse, interpretable curve fitting with automatic hyperparameter tuning.

### Why BLR?

Standard least-squares regression gives point estimates, not calibrated uncertainty. BLR inverts this:

- **Uncertainty propagation** — every prediction includes epistemic (model) and aleatoric (measurement noise) uncertainty bounds
- **Principled regularization** — Bayesian priors naturally prevent overfitting without manual hyperparameter tuning
- **Interpretability** — posterior weights and their covariance tell you exactly what the model learned
- **Empirical Bayes** — noise level is estimated from data, not guessed
- **Single-model inference** — marginalizes over weight uncertainty for well-calibrated predictions, avoiding ensemble complexity

### Why ARD?

Traditional regression requires manual feature selection and hyperparameter tuning. ARD eliminates both by learning which input features are truly relevant to your problem, automatically driving irrelevant ones to zero. You get:

- **Automatic sparse models** — only the features that matter remain active
- **Calibrated uncertainty** — each prediction includes epistemic and aleatoric uncertainty bounds
- **Interpretability** — understand exactly why the model makes decisions
- **Zero manual tuning** — the EM algorithm learns hyperparameters from data

### Industrial Curve Fitting & Sensor Calibration

`blr-core` is purpose-built for **Industry 4.0 regression tasks**:

- **Sensor calibration** — fit physics-based models to sensor data; quantify measurement uncertainty
- **Anomaly detection** — ARD automatically flags broken or misconfigured sensors
- **Real-time inference** — predict on new data with propagated uncertainty in milliseconds
- **Edge deployment** — runs on embedded controllers, IoT devices, and WebAssembly runtimes

### Designed for Integration

`blr-core` is intentionally lightweight and composable:

- **Minimal dependencies** — pure Rust + [`faer`](https://crates.io/crates/faer) for linear algebra
- **WASM-first** — compiles to `wasm32-wasip2` with no unsafe code in core logic
- **Embeddable** — integrate directly or wrap as a service/component in larger systems

## Features

- ✓ **Sparse Feature Selection**: ARD automatically discovers and deactivates irrelevant features
- ✓ **Uncertainty Quantification**: Epistemic and aleatoric uncertainty decomposition
- ✓ **Noise Estimation**: Automated residual analysis and noise floor detection
- ✓ **Physics-Aware Basis Functions**: Polynomial, RBF, and sensor-specific feature maps
- ✓ **WASM-Ready**: Compiles to `wasm32-wasip2` with no unsafe code in core logic
- ✓ **Production-Ready**: Extensive test coverage, benchmark harness, zero panics on valid inputs
- ✓ **No external runtime**: Pure CPU math via [`faer`](https://crates.io/crates/faer)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
blr-core = "0.1"
```

## Quick Start

```rust
use blr_core::{features, fit, ArdConfig};

// Fit a sparse model to a synthetic dataset
// True model: y = 2·x + 0.5·x² + noise
let n = 30;
let x: Vec<f64> = (0..n)
    .map(|i| -3.0 + 6.0 * (i as f64) / (n as f64 - 1.0))
    .collect();
let y: Vec<f64> = x.iter()
    .map(|xi| 2.0 * xi + 0.5 * xi * xi + 0.2 * xi.sin())
    .collect();

// Polynomial feature basis: [1, x, x², x³]
let (phi, d) = features::polynomial(&x, 3);

// Fit with ARD — automatically discovers irrelevant features
let config = ArdConfig {
    max_iter: 200,
    tol: 1e-6,
    ..ArdConfig::default()
};
let fitted = fit(&phi, &y, n, d, &config)?;

// Inspect results
println!("Noise std:        {:.4}", fitted.noise_std());
println!("Relevant features: {}/{}", 
    fitted.relevant_features(None).iter().filter(|&&x| x).count(), d);
println!("Active weights:   {:?}", &fitted.posterior.mean[..d]);

// Predict with uncertainty on new data
let x_test = vec![-1.0, 0.0, 1.0];
let (phi_test, _) = features::polynomial(&x_test, 3);
let pred = fitted.predict(&phi_test, 3, d);
println!("Predictions (mean ± std):");
for (i, &mu) in pred.mean.iter().enumerate() {
    println!("  x={:5.1}:  {: .3} ± {:.3}", x_test[i], mu, pred.total_std[i]);
}
# Ok::<(), blr_core::BLRError>(())
```

## How It Works

BLR assumes a Gaussian likelihood over observations: each target `y_n` equals
the dot product of a feature vector `φ_n` with a weight vector `w`, plus Gaussian
noise with precision β.

ARD places an independent Gaussian prior on each weight `w_d` with its own precision
hyperparameter `α_d`. The EM algorithm iteratively updates both the posterior over
weights and the hyperparameters. Features with low signal drive `α_d → ∞`, effectively
removing those weights from the model.

The result is a **sparse, interpretable model** with calibrated uncertainty estimates
— ideal for sensor calibration where only a few physics-based features truly explain
the sensor's behaviour.

## Core Modules

| Module | Purpose |
|--------|---------|
| [`gaussian`](https://docs.rs/blr-core/latest/blr_core/gaussian/) | Multivariate Gaussian utilities |
| [`ard`](https://docs.rs/blr-core/latest/blr_core/ard/) | BLR+ARD fitting, predictions, evidence computation |
| [`noise_estimation`](https://docs.rs/blr-core/latest/blr_core/noise_estimation/) | Automated noise characterisation |
| [`features`](https://docs.rs/blr-core/latest/blr_core/features/) | Polynomial, RBF, and custom basis functions |
| [`synthetic_data`](https://docs.rs/blr-core/latest/blr_core/synthetic_data/) | Benchmark datasets and physics simulators |

## Examples

Run examples directly:

```bash
cargo run --example quick_start -p blr-core
cargo run --example noise_estimation_workflow -p blr-core
cargo run --example hall_sensor -p blr-core
```

## Performance

Quick reference on a single core (Intel i7-12700K, `--release`):

| N (obs) | D (features) | Fit time |
|---------|--------------|----------|
| 30      | 6            | ~3 ms   |
| 60      | 11           | ~12 ms  |
| 500     | 30           | ~240 ms |

See [BENCHMARK_GUIDE.md](./BENCHMARK_GUIDE.md) for detailed benchmarks, hardware
requirements, and how to reproduce and interpret results.

## Running Benchmarks

```bash
# Run all benchmarks (~2 minutes)
cargo bench -p blr-core

# Run a specific benchmark
cargo bench -p blr-core -- "fit_medium"
```

## WASM Deployment

```bash
cargo build -p blr-core --target wasm32-wasip2 --release
```

See [`PYTHON_BINDINGS.md`](../../PYTHON_BINDINGS.md) for integration with Python via Wasmtime.

## When NOT to Use blr-core

- **High-dimensional features**: Covariance matrix inversion scales O(D³); becomes prohibitive for D > ~30–50.
- **Deep non-linear functions**: BLR is a linear model in the feature space.
- **Multi-output regression**: Current implementation is single-output only.
- **Hard real-time deadlines**: EM convergence is iterative; worst-case runtime is
  bounded by `max_iter` but not cycle-exact.

## Compatibility

- **Rust**: 1.70+
- **Platforms**: Linux, macOS, Windows, WebAssembly (WASI Preview 2)
- **License**: Apache 2.0
- **Key dependency**: [`faer`](https://crates.io/crates/faer) — pure-Rust linear algebra

## References

- Tipping, M. E. (2001). "Sparse Bayesian learning and the relevance vector machine."
  *Journal of Machine Learning Research*, 1, [211–244](https://www.jmlr.org/papers/volume1/tipping01a/tipping01a.pdf).
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Hennig, P. (2025). [*Probabilistic Machine Learning* | 2025](https://www.youtube.com/playlist?list=PL05umP7R6ij0hPfU7Yuz8J9WXjlb3MFjm) .
  University of Tübingen.
- MacKay, D. J. C. (1992). "Bayesian Interpolation."
  *Neural Computation*, 4(3), 415–447.

## License

Licensed under the Apache License, Version 2.0. See [`LICENSE`](./LICENSE) for details.

## Acknowledgments

Part of the [**Wamli**](https://wamli.github.io/) initiative - __*WAMLI - "WASM Machine Learning Inference"*__. Special thanks to the
[ByteCode Alliance](https://bytecodealliance.org/) for WASI standardisation and the
[Wasmtime](https://wasmtime.dev/) project for runtime support.

