use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::RngCore;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use reed_solomon_erasure::galois_8::ReedSolomon as RefReedSolomon;
use solana_turbine_simd::SimdReedSolomon;

const SHRED_SIZE: usize = 1228;

fn bench_shred_encode(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(0x42_1337_BEEF);

    let test_configs = [
        (16, 16, "16_data_16_parity"),
        (32, 32, "32_data_32_parity_canonical"),
        (64, 64, "64_data_64_parity"),
    ];

    for &(k, m, name) in &test_configs {
        let mut group = c.benchmark_group(format!("shred_encode_{}", name));

        // Throughput measured as data payload bytes processed (K * 1,228 bytes)
        let data_bytes = (k * SHRED_SIZE) as u64;
        group.throughput(Throughput::Bytes(data_bytes));

        // Prepare test data shreds
        let data_buffers: Vec<Vec<u8>> = (0..k)
            .map(|_| {
                let mut b = vec![0u8; SHRED_SIZE];
                rng.fill_bytes(&mut b);
                b
            })
            .collect();
        let data_slices: Vec<&[u8]> = data_buffers.iter().map(|v| v.as_slice()).collect();

        // 1. Benchmark: Agave Current (reed-solomon-erasure with simd-accel)
        let ref_rs = RefReedSolomon::new(k, m).unwrap();
        let mut ref_parity = vec![vec![0u8; SHRED_SIZE]; m];
        group.bench_function(BenchmarkId::new("reed-solomon-erasure (Agave Current)", k), |b| {
            b.iter(|| {
                let mut p_slices: Vec<&mut [u8]> =
                    ref_parity.iter_mut().map(|v| v.as_mut_slice()).collect();
                ref_rs
                    .encode_sep(black_box(&data_slices), black_box(&mut p_slices))
                    .unwrap();
            })
        });

        // 2. Benchmark: Pure Scalar Baseline
        let our_rs = SimdReedSolomon::new(k, m).unwrap();
        let mut scalar_parity = vec![vec![0u8; SHRED_SIZE]; m];
        group.bench_function(BenchmarkId::new("SolARM Scalar Baseline", k), |b| {
            b.iter(|| {
                let mut p_slices: Vec<&mut [u8]> =
                    scalar_parity.iter_mut().map(|v| v.as_mut_slice()).collect();
                our_rs
                    .encode_sep_scalar(black_box(&data_slices), black_box(&mut p_slices))
                    .unwrap();
            })
        });

        // 3. Benchmark: ARM NEON Slice-by-Slice (Drop-in replacement loop)
        #[cfg(target_arch = "aarch64")]
        {
            let mut neon_slice_parity = vec![vec![0u8; SHRED_SIZE]; m];
            group.bench_function(
                BenchmarkId::new("SolARM NEON (Slice-by-Slice vqtbl1q_u8)", k),
                |b| {
                    b.iter(|| {
                        let mut p_slices: Vec<&mut [u8]> =
                            neon_slice_parity.iter_mut().map(|v| v.as_mut_slice()).collect();
                        our_rs
                            .encode_sep_neon_slice_by_slice(
                                black_box(&data_slices),
                                black_box(&mut p_slices),
                            )
                            .unwrap();
                    })
                },
            );
        }

        // 4. Benchmark: SolARM NEON Direct Register Accumulation Kernel
        #[cfg(target_arch = "aarch64")]
        {
            let mut neon_direct_parity = vec![vec![0u8; SHRED_SIZE]; m];
            group.bench_function(
                BenchmarkId::new("SolARM NEON (Direct Register Accumulation)", k),
                |b| {
                    b.iter(|| {
                        let mut p_slices: Vec<&mut [u8]> =
                            neon_direct_parity.iter_mut().map(|v| v.as_mut_slice()).collect();
                        our_rs
                            .encode_sep_neon(
                                black_box(&data_slices),
                                black_box(&mut p_slices),
                            )
                            .unwrap();
                    })
                },
            );
        }

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = bench_shred_encode
}
criterion_main!(benches);
