//! Solana Turbine Hardware-Vectorized Reed-Solomon Erasure Coding Engine.
//!
//! Provides 100% wire-compatible bit-for-bit parity with `reed-solomon-erasure`
//! in Solana Agave master, accelerated using 128-bit ARM NEON `vqtbl1q_u8`
//! and 256-bit x86 AVX2 `_mm256_shuffle_epi8`.

pub mod galois;
pub mod matrix;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx2;

pub mod engine;

pub use engine::{Error, SimdReedSolomon};
