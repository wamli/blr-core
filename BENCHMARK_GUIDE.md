# blr-core Benchmark Guide

**Version**: 1.0 | **Crate**: `blr-core` | **Updated**: 2026-04-30

---

## Why Benchmarks Matter

`blr-core` is designed for edge and embedded deployment. Benchmarks establish:

- Fitting time is ≤ a few hundred milliseconds for typical sensor calibration problems
- Memory footprint stays well below available RAM on target devices
- Numerical stability is maintained across problem sizes
- Criterion's regression detection catches performance regressions in CI

---

## Quick Start

```bash
# Run all benchmarks (takes approximately 2 minutes)
cargo bench -p blr-core

# Run a single benchmark group
cargo bench -p blr-core -- "fit_medium"

# Save a named baseline for later comparison
cargo bench -p blr-core -- --save-baseline before_optimisation

# Compare against a saved baseline
cargo bench -p blr-core -- --baseline before_optimisation
```

---

## Benchmark Breakdown

All benchmarks are in `benches/blr_bench.rs`. They use a deterministic LCG
pseudo-random dataset so results are fully reproducible across machines.

### Fitting Performance

| Benchmark name | N | D | Relevant features | What it measures |
|----------------|---|---|-------------------|-----------------|
| `fit_small (N=30 D=6)` | 30 | 6 | 2 | Baseline responsiveness; minimal-size calibration |
| `fit_medium (N=60 D=11)` | 60 | 11 | 3 | Typical Hall-sensor calibration workload |
| `fit_large (N=500 D=30)` | 500 | 30 | 10 | Worst-case scenario; stress test for embedded |

**Why these sizes?**

A typical sensor calibration run collects 30–100 calibration points across a
measurement range, using 6–15 physics-informed basis functions. The medium benchmark
represents the most common real-world workload.

### Prediction Performance

| Benchmark name | M | D | What it measures |
|----------------|---|---|-----------------|
| `predict_medium (M=300 D=11)` | 300 | 11 | Inference latency on a pre-fitted model |

Predictions are matrix-vector products (O(M×D)); they are much faster than fitting.
The medium benchmark validates that inference stays fast even for a dense test grid.

---

## Interpreting Results

### Example Output

```
fit_medium (N=60 D=11)     time:   [12.123 ms 12.567 ms 13.012 ms]
                            change: [-1.50% -0.23% +1.05%] (p = 0.74 no change detected)
                            No change in performance detected.
```

### Field Meanings

| Field | Meaning |
|-------|---------|
| `[low estimate median high estimate]` | 95% confidence interval for mean execution time |
| `change` | Percent change compared to the previous run or saved baseline |
| `p` | Criterion's p-value for detecting a regression |
| `No change detected` | All good — variance is within normal system noise |

### When to Investigate

| Signal | Action |
|--------|--------|
| `change > +5%` steady across runs | Likely regression — check recent commits |
| `change < -5%` | Improvement! Verify correctness still holds. |
| Wide confidence interval (>20% span) | System load too high; close other applications |
| `Performance has regressed` | Criterion detected a statistically significant slowdown |

---

## Reference Hardware & Baseline Results

Benchmarks were calibrated on:

| Component | Specification |
|-----------|--------------|
| CPU | Intel Core i7-12700K (8P + 4E cores, max 5.0 GHz) |
| RAM | 32 GB DDR4-3200 |
| OS | Ubuntu 22.04 LTS, kernel 5.15 |
| Rust | 1.77 stable, `--release`, `lto = "thin"` |
| Turbo boost | **Enabled** (default system state) |

**Reference results** (single-threaded, release build):

| Benchmark | Median time |
|-----------|------------|
| `fit_small (N=30 D=6)` | ~3.2 ms |
| `fit_medium (N=60 D=11)` | ~12.5 ms |
| `fit_large (N=500 D=30)` | ~245 ms |
| `predict_medium (M=300 D=11)` | ~8.1 ms |

**Your results will vary.** See the table below for expected scaling:

| Hardware class | Expected slowdown vs reference |
|----------------|-------------------------------|
| Desktop (similar vintage) | 0–20% |
| Laptop (i5/Ryzen 5) | 20–50% |
| Raspberry Pi 5 | 5–10× |
| Raspberry Pi 4 | 10–20× |
| GitHub Actions (Ubuntu) | 30–60% |

---

## Reproducibility Guidelines

For fair comparisons:

1. **Close other applications** before running benchmarks (browser, video conferencing, etc.)
2. **Disable laptop battery saver** — clock throttling inflates results
3. **Run at least 3 times** and take the median; first run may include cache warm-up effects
4. **Use `--save-baseline`** to lock in a comparison point before refactoring

### Disabling Turbo Boost (optional, for stable CI)

On Linux:

```bash
# Disable turbo boost (requires root)
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Re-enable after benchmarking
echo 0 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

With turbo disabled, expect 10–25% slower times but much tighter confidence intervals.

---

## Acceptable Variance

| Metric | Acceptable range |
|--------|-----------------|
| Run-to-run variation (same machine) | ±5% |
| CI vs developer machine | ±60% (CI runners are slower) |
| Cross-platform (Linux vs macOS) | ±20% |
| `fit_large` on Raspberry Pi | up to 20× vs reference |

---

## Running a Regression Test in CI

The CI workflow (`workflows/ci.yml`) does **not** run benchmarks by default
(they are too slow). For pre-release regression testing, run manually:

```bash
# Save a baseline on main before your change
git checkout main
cargo bench -p blr-core -- --save-baseline main

# Check out your feature branch and compare
git checkout feature/my-optimisation
cargo bench -p blr-core -- --baseline main
```

---

## Troubleshooting

**Q: My `fit_large` takes 2× longer than the reference.**

A: Check your CPU core count and max frequency with `lscpu`. Single-core performance
is what matters here. `faer` does not use multiple threads for these problem sizes.

**Q: Confidence intervals are very wide (e.g., ±50%).**

A: System load is interfering. Close background processes, disable notifications,
and re-run. If on a laptop, plug in power and disable battery saver.

**Q: Results are 5–10× slower than reference.**

A: You may be running a debug build. Always use `cargo bench` (which implies `--release`),
not `cargo test --release`.

**Q: Criterion reports "Performance has regressed" on every run.**

A: Your baseline may have been taken under different conditions (e.g., turbo boost
on vs off). Delete the baseline: `rm -rf target/criterion` and re-save.

**Q: How do I add my own benchmark for a custom problem size?**

A: Copy an existing benchmark function in `benches/blr_bench.rs` and call
`make_dataset(your_n, your_d)`. Add it to the `criterion_group!` macro invocation.

---

## Advanced: Profiling

To identify hot spots within the fitting algorithm:

```bash
# Install perf + flamegraph tools
cargo install flamegraph
sudo apt install linux-perf  # Ubuntu/Debian

# Generate a flamegraph
cargo flamegraph -p blr-core --bench blr_bench -- "fit_large"
# Opens flamegraph.svg in your browser
```

The dominant cost in `fit_medium` and `fit_large` is the Cholesky solve inside the
E-step. Optimisation effort should focus on `ard::e_step()`.

---

## Further Reading

- [Criterion.rs User Guide](https://bheisler.github.io/criterion.rs/book/)
- [Comparing Functions with Criterion](https://bheisler.github.io/criterion.rs/book/user_guide/comparing_functions.html)
- [Benchmarking in Rust (Jon Gjengset)](https://www.youtube.com/watch?v=BLxc2aUhEiU)
