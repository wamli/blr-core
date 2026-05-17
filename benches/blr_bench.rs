use blr_core::{fit, ArdConfig};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

/// Build a deterministic N×D feature matrix and N-vector of targets.
/// Uses a simple linear congruential pattern so benchmarks are reproducible.
fn make_dataset(n: usize, d: usize) -> (Vec<f64>, Vec<f64>) {
    // Pseudo-random features via LCG
    let mut state: u64 = 12345;
    let lcg = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // map [0, 2^64) → (-1, 1)
        (*s as i64 as f64) / (i64::MAX as f64)
    };

    let phi: Vec<f64> = (0..n * d).map(|_| lcg(&mut state)).collect();

    // True weights: first d/3 components = 1.0, rest = 0.0; noise std ≈ 0.1
    let n_relevant = (d / 3).max(1);
    let true_w: Vec<f64> = (0..d)
        .map(|j| if j < n_relevant { 1.0 } else { 0.0 })
        .collect();

    let y: Vec<f64> = (0..n)
        .map(|i| {
            let dot: f64 = (0..d).map(|j| phi[i * d + j] * true_w[j]).sum();
            dot + 0.1 * lcg(&mut state)
        })
        .collect();

    (phi, y)
}

fn bench_fit_small(c: &mut Criterion) {
    let (phi, y) = make_dataset(30, 6);
    let config = ArdConfig {
        max_iter: 200,
        tol: 1e-5,
        ..ArdConfig::default()
    };
    c.bench_function("fit_small (N=30 D=6)", |b| {
        b.iter(|| fit(black_box(&phi), black_box(&y), 30, 6, black_box(&config)).unwrap())
    });
}

fn bench_fit_medium(c: &mut Criterion) {
    let (phi, y) = make_dataset(60, 11);
    let config = ArdConfig {
        max_iter: 200,
        tol: 1e-5,
        ..ArdConfig::default()
    };
    c.bench_function("fit_medium (N=60 D=11)", |b| {
        b.iter(|| fit(black_box(&phi), black_box(&y), 60, 11, black_box(&config)).unwrap())
    });
}

fn bench_fit_large(c: &mut Criterion) {
    let (phi, y) = make_dataset(500, 30);
    let config = ArdConfig {
        max_iter: 200,
        tol: 1e-5,
        ..ArdConfig::default()
    };
    c.bench_function("fit_large (N=500 D=30)", |b| {
        b.iter(|| fit(black_box(&phi), black_box(&y), 500, 30, black_box(&config)).unwrap())
    });
}

fn bench_predict_medium(c: &mut Criterion) {
    let (phi, y) = make_dataset(60, 11);
    let config = ArdConfig {
        max_iter: 200,
        tol: 1e-5,
        ..ArdConfig::default()
    };
    let fitted = fit(&phi, &y, 60, 11, &config).unwrap();

    let (phi_test, _) = make_dataset(300, 11);
    c.bench_function("predict_medium (M=300 D=11)", |b| {
        b.iter(|| fitted.predict(black_box(&phi_test), 300, 11))
    });
}

criterion_group!(
    benches,
    bench_fit_small,
    bench_fit_medium,
    bench_fit_large,
    bench_predict_medium
);
criterion_main!(benches);
