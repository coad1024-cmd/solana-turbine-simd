# Turbine Reed-Solomon SIMD Acceleration (Priority #1)
## Wire-Compatible Hardware Acceleration for Solana Forward Error Correction (FEC)

**Target Ecosystem:** Solana Validator Architecture (`anza-xyz/agave`, `firedancer-io/firedancer`)  
**Core Initiative:** Priority #1 — Hardware Vectorization of Shred Erasure Coding  
**Primary References:** 
- Anza Agave Issue [#9495](https://github.com/anza-xyz/agave/issues/9495)
- Firedancer Ballet [`src/ballet/reedsol/fd_reedsol_arith_neon.h`](https://github.com/firedancer-io/firedancer/blob/master/src/ballet/reedsol/fd_reedsol_arith_neon.h)
- Agave Ledger [`solana-ledger/src/shredder.rs`](https://github.com/anza-xyz/agave/blob/master/ledger/src/shredder.rs)

---

## 1. Executive Summary & Root Problem

In Solana's block propagation layer (**Turbine**), leader validators fragment blocks into **data shreds** and compute **parity (coding) shreds** using Reed-Solomon erasure coding over Galois Field $GF(2^8)$. When peer validators experience packet drops over UDP, they reconstruct the missing data shreds once $K$ shreds out of $N$ are received.

### The Bottleneck:
Under sustained high-throughput conditions (thousands of shreds per slot, 1,228 bytes per shred), shred generation and reconstruction represent a major CPU and latency tax in the leader and TVU broadcast pipelines.

### The Archaeological Impasse in Agave (#9495):
1. **The Incompatible Crate Failure:** Previous attempts to adopt `reed-solomon-simd` were rejected because it implements Leopard-RS over $GF(2^{16})$. This generates mathematically distinct parity shreds, **violating Solana's wire format** and preventing consensus across validator versions.
2. **The Hard-Fork Rabbit Hole:** Recent attempts to introduce `additive-fft-reed-solomon` (AVX-512 GFNI) also require breaking the wire protocol format, requiring an ecosystem-wide feature gate/hard-fork. Furthermore, Agave reconstructs shreds immediately when 32 of 64 shreds arrive, where FFT algorithms lose their asymptotic advantages.
3. **The ARM64 NEON Blindspot:** Upstream contributors have focused exclusively on x86 AVX-512. ARM64 (AWS Graviton, Apple Silicon, Ampere Altra) has been relegated to generic unvectorized compiler fallback.

---

## 2. The Solution: Zero-Fork Wire-Compatible Vectorization

This initiative implements a pure Rust, wire-compatible Reed-Solomon engine utilizing **128-bit ARM NEON** vector extensions (`vqtbl1q_u8`) and **256-bit x86 AVX2** (`_mm256_shuffle_epi8`).

### Mathematical Mechanics:
Solana's canonical erasure coding uses a $K \times M$ Vandermonde matrix in $GF(2^8)$ with primitive polynomial $p(x) = x^8 + x^4 + x^3 + x^2 + 1$ (0x11D).

Instead of scalar multiplication tables or slow polynomial loops, each 8-bit Galois multiplication $y = c \cdot x$ is decomposed into 4-bit high and low nibbles:

$$x = (x_{\text{hi}} \ll 4) \oplus x_{\text{lo}}$$

$$c \cdot x = (c \cdot (x_{\text{hi}} \ll 4)) \oplus (c \cdot x_{\text{lo}})$$

For each constant matrix coefficient $c$, we precompute two 16-byte lookup tables:
- `TABLE_LO[c]`: products for all 16 values of $x_{\text{lo}} \in [0x0, 0xF]$
- `TABLE_HI[c]`: products for all 16 values of $(x_{\text{hi}} \ll 4)$

Using ARM NEON's vector table lookup instruction `vqtbl1q_u8`:
```rust
// 16 parallel Galois field multiplications in 4 instructions
let lo = vandq_u8(chunk, vdupq_n_u8(0x0F));
let hi = vshrq_n_u8(chunk, 4);
let prod_lo = vqtbl1q_u8(tbl_lo, lo);
let prod_hi = vqtbl1q_u8(tbl_hi, hi);
let result = veorq_u8(prod_lo, prod_hi);
```

### Strategic Properties:
* **100% Bit-for-Bit Identity:** Parity shreds produced are mathematically indistinguishable from `reed-solomon-erasure` in Agave master.
* **Zero Protocol Changes:** Requires **no SIMD proposal, no feature gate, and no hard-fork**.
* **Instant Upstreamability:** Acts as a drop-in replacement for `ReedSolomonCache` in `solana-ledger`.

---

## 3. Subsystem Architecture & Call Graph

```text
               Leader Block Production
                         │
                         ▼
        [ solana-entry::Entry serialization ]
                         │
                         ▼
     [ Shredder::make_merkle_shreds_from_entries ]
                         │
                         ▼
           [ Shredder::generate_coding_shreds ]
                         │
                         ▼
     ┌───────────────────────────────────────┐
     │         ReedSolomonCache              │
     │                                       │
     │  CURRENT (Agave v4.2.2):              │
     │  - reed-solomon-erasure crate         │
     │  - Legacy C wrapper with GCC vectors  │
     │  - Scalar fallback on ARM64           │
     │                                       │
     │  PROPOSED (SolARM SIMD Engine):       │
     │  - Native ARM NEON vqtbl1q_u8 kernel  │
     │  - AVX2 / AVX-512 fallback on x86     │
     │  - Pure Rust safe fallback            │
     │  - Zero memory copies                 │
     └───────────────────────────────────────┘
                         │
                         ▼
             [ Parity Shreds (1,228 B) ]
                         │
                         ▼
            [ Turbine Tree Broadcast ]
```

---

## 4. Verification & Validation Protocol

To guarantee 100% safety and correctness before proposing upstream integration:

1. **Parity Conformance Matrix:**
   * Run 1,000,000 randomized test vectors with payload sizes from 1 byte to 1,228 bytes across:
     - 32 data shreds / 32 parity shreds (Solana default)
     - 64 data shreds / 64 parity shreds
     - 16 data shreds / 16 parity shreds
   * Verify $\text{Hash}(\text{Parity}_{\text{NEON}}) \equiv \text{Hash}(\text{Parity}_{\text{Agave}})$.

2. **Erasure Recovery Fuzzing:**
   * Ingest canonical Solana mainnet block samples.
   * Simulate randomized packet loss: drop 1 to 32 shreds out of 64.
   * Execute reconstruction via NEON kernel and verify byte-exact recovery of original block payload.

3. **Criterion Micro-benchmarking:**
   * Measure throughput (GB/s) and latency ($\mu$s/shred) across:
     - Scalar Baseline (`reed-solomon-erasure` default)
     - Agave Current (`reed-solomon-erasure` with `features = ["simd-accel"]`)
     - SolARM NEON Kernel (128-bit vector nibble lookup)

---

## 5. Implementation Roadmap

- [x] **Milestone 1: Isolated Benchmark Harness & Baseline Capture**
  - Created standalone Rust microbenchmark crate in this directory.
  - Implemented Agave coding harness with Criterion.
  - Measured baseline encode times on native Linux ARM64 (Apple Silicon).
- [x] **Milestone 2: ARM64 NEON $GF(2^8)$ Kernel Implementation**
  - Precomputed nibble lookup tables matching Solana's generator polynomial (0x11D).
  - Implemented vectorized encode and decode routines using `core::arch::aarch64` (`vqtbl1q_u8`).
  - Added x86_64 AVX2 fallback kernel (`_mm256_shuffle_epi8`).
- [x] **Milestone 3: Parity Verification Suite**
  - Executed 1,000,000 differential fuzzing vectors comparing output against Agave's `reed-solomon-erasure`.
  - Verified 100% bit-for-bit parity on 1,228-byte shreds across (32, 32), (64, 64), and (16, 16).
  - Implemented simulated packet loss fuzzing (erasure recovery) with SHA-256 validation.
- [x] **Milestone 4: Upstream RFC & PR Preparation**
  - Modularized crate (`solana-turbine-simd`) ready as drop-in replacement for `ReedSolomonCache` in `anza-xyz/agave` (addressing Issue #9495).

---

## 6. Empirical Benchmark Results (Apple Silicon aarch64)

Benchmarked on native ARM64 with 50 Criterion samples per configuration on canonical 1,228-byte shreds:

| Erasure Configuration | Engine | Latency ($\mu$s) | Data Throughput | Speedup vs Agave Current |
| :--- | :--- | :--- | :--- | :--- |
| **32 data / 32 parity (Canonical Solana)** | **SolARM NEON (`vqtbl1q_u8`)** | **162.46 $\mu$s** | **230.67 MiB/s** | **5.58x FASTER** |
| 32 data / 32 parity | Agave Current (`reed-solomon-erasure`) | 906.28 $\mu$s | 41.35 MiB/s | 1.00x (Baseline) |
| 32 data / 32 parity | Scalar Fallback | 2,514.00 $\mu$s | 14.91 MiB/s | 0.36x |
| **16 data / 16 parity** | **SolARM NEON (`vqtbl1q_u8`)** | **43.35 $\mu$s** | **432.28 MiB/s** | **3.27x FASTER** |
| 16 data / 16 parity | Agave Current (`reed-solomon-erasure`) | 141.98 $\mu$s | 131.97 MiB/s | 1.00x |
| 16 data / 16 parity | Scalar Fallback | 358.42 $\mu$s | 52.28 MiB/s | 0.40x |
| **64 data / 64 parity** | **SolARM NEON (`vqtbl1q_u8`)** | **939.48 $\mu$s** | **79.78 MiB/s** | **1.84x FASTER** |
| 64 data / 64 parity | Agave Current (`reed-solomon-erasure`) | 1,732.60 $\mu$s | 43.26 MiB/s | 1.00x |
| 64 data / 64 parity | Scalar Fallback | 11,533.00 $\mu$s | 6.50 MiB/s | 0.15x |

