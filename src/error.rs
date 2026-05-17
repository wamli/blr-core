use thiserror::Error;

/// Errors that can arise during BLR+ARD operations.
#[derive(Debug, Error)]
pub enum BLRError {
    /// A matrix required to be positive-definite is singular.
    #[error("matrix is singular or not positive-definite")]
    SingularMatrix,

    /// Dimension mismatch in an input or intermediate calculation.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimMismatch { expected: usize, got: usize },
}
