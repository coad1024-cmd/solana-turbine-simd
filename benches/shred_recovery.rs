use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::seq::SliceRandom;
use rand::RngCore;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use reed_solomon_erasure::galois_8::ReedSolomon as RefReedSolomon;
use solana_turbine_simd::SimdReedSolomon;

const SHRED_SIZE: usize = 1228;

fn bench_shred_recovery(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(0xDEAD_BEEF_C001_CAFE);

    let data_shards = 32;
    let parity_shards = 32;
    let total_shards = data_shards + parity_shards;

    let our_rs = SimdReedSolomon::new(data_shards, parity_shards).unwrap();
    let ref_rs = RefReedSolomon::new(data_shards, parity_shards).unwrap();

    // Generate test data shreds
    let data_buffers: Vec<Vec<u8>> = (0..data_shards)
        .map(|_| {
            let mut b = vec![0u8; SHRED_SIZE];
            rng.fill_bytes(&mut b);
            b
        })
        .collect();

    // Generate parity shreds
    let mut all_shards: Vec<Vec<u8>> = Vec::with_capacity(total_shards);
    for d in &data_buffers {
        all_shards.push(d.clone());
    }
    for _ in 0..parity_shards {
        all_shards.push(vec![0u8; SHRED_SIZE]);
    }
    {
        let (d_part, p_part) = all_shards.split_at_mut(data_shards);
        our_rs.encode_sep(d_part, p_part).unwrap();
    }

    // Packet loss scenarios: 3 missing (10%), 8 missing (25%), 16 missing (50%)
    let loss_scenarios = [
        (3, "10pct_loss_3_missing"),
        (8, "25pct_loss_8_missing"),
        (16, "50pct_loss_16_missing"),
    ];

    for &(num_dropped, scenario_name) in &loss_scenarios {
        let mut group = c.benchmark_group(format!("shred_recovery_{}", scenario_name));

        // Throughput measured as reconstructed data bytes (num_dropped * 1,228 bytes)
        let recovered_bytes = (num_dropped * SHRED_SIZE) as u64;
        group.throughput(Throughput::Bytes(recovered_bytes));

        // Deterministically pick which data shreds to drop
        let mut indices: Vec<usize> = (0..data_shards).collect();
        indices.shuffle(&mut rng);
        let dropped_indices: Vec<usize> = indices.into_iter().take(num_dropped).collect();

        // 1. Benchmark: Agave Current (reed-solomon-erasure)
        group.bench_function(
            BenchmarkId::new("reed-solomon-erasure (Agave Current)", num_dropped),
            |b| {
                b.iter_batched(
                    || {
                        // Prepare (Vec<u8>, bool) shards
                        let mut shards: Vec<(Vec<u8>, bool)> =
                            all_shards.clone().into_iter().map(|v| (v, true)).collect();
                        for &idx in &dropped_indices {
                            shards[idx].1 = false;
                        }
                        shards
                    },
                    |mut shards| {
                        ref_rs.reconstruct_data(black_box(&mut shards)).unwrap();
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        // 2. Benchmark: SolARM NEON (SimdReedSolomon)
        group.bench_function(
            BenchmarkId::new("SolARM NEON (SimdReedSolomon)", num_dropped),
            |b| {
                b.iter_batched(
                    || {
                        let mut shards = all_shards.clone();
                        let mut present = vec![true; total_shards];
                        for &idx in &dropped_indices {
                            present[idx] = false;
                            shards[idx].fill(0);
                        }
                        (shards, present)
                    },
                    |(mut shards, present)| {
                        our_rs
                            .reconstruct_data(black_box(&mut shards), black_box(&present))
                            .unwrap();
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = bench_shred_recovery
}
criterion_main!(benches);
