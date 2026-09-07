//! Erasure Recovery Fuzzing Suite:
//! Simulates randomized packet loss (dropping 1 to 32 shreds out of 64)
//! and reconstructs the original payload using the NEON kernel,
//! verifying byte-exact recovery and SHA-256 hash match.

use rand::seq::SliceRandom;
use rand::RngCore;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use solana_turbine_simd::SimdReedSolomon;

const SHRED_SIZE: usize = 1228;

#[test]
fn test_erasure_recovery_simulated_packet_loss() {
    let mut rng = ChaCha20Rng::seed_from_u64(0xFEED_FACE_CAFE_BEEF);

    let data_shards = 32;
    let parity_shards = 32;
    let total_shards = data_shards + parity_shards;
    let rs = SimdReedSolomon::new(data_shards, parity_shards).unwrap();

    let iterations = 100;
    println!("Running {} iterations of randomized packet loss recovery...", iterations);

    for iter in 0..iterations {
        // 1. Generate original data shreds
        let original_data: Vec<Vec<u8>> = (0..data_shards)
            .map(|_| {
                let mut buf = vec![0u8; SHRED_SIZE];
                rng.fill_bytes(&mut buf);
                buf
            })
            .collect();

        // Compute original block SHA-256
        let mut original_hasher = Sha256::new();
        for d in &original_data {
            original_hasher.update(d);
        }
        let original_hash = original_hasher.finalize();

        // 2. Encode parity shreds
        let mut all_shards: Vec<Vec<u8>> = Vec::with_capacity(total_shards);
        for d in &original_data {
            all_shards.push(d.clone());
        }
        for _ in 0..parity_shards {
            all_shards.push(vec![0u8; SHRED_SIZE]);
        }

        let (data_part, parity_part) = all_shards.split_at_mut(data_shards);
        rs.encode_sep(data_part, parity_part).unwrap();

        // 3. Simulate random packet loss: drop between 1 and 32 data shreds
        // We can drop any subset of shreds as long as at least 32 shards survive
        let mut shard_present = vec![true; total_shards];

        // Pick how many data shreds to drop: between 1 and data_shards (up to 32)
        let num_data_drops = (rng.next_u32() as usize % data_shards) + 1;
        let mut data_indices: Vec<usize> = (0..data_shards).collect();
        data_indices.shuffle(&mut rng);

        for &idx in &data_indices[..num_data_drops] {
            shard_present[idx] = false;
            // Wipe the data to simulate packet loss
            all_shards[idx].fill(0xFF);
        }

        // Also randomly drop some parity shreds, keeping total surviving >= data_shards
        let max_additional_parity_drops = data_shards - num_data_drops;
        let num_parity_drops = if max_additional_parity_drops > 0 {
            rng.next_u32() as usize % (max_additional_parity_drops + 1)
        } else {
            0
        };

        let mut parity_indices: Vec<usize> = (data_shards..total_shards).collect();
        parity_indices.shuffle(&mut rng);
        for &idx in &parity_indices[..num_parity_drops] {
            shard_present[idx] = false;
            all_shards[idx].fill(0xAA);
        }

        let total_surviving = shard_present.iter().filter(|&&p| p).count();
        assert!(
            total_surviving >= data_shards,
            "Must have at least K shreds surviving"
        );

        // 4. Reconstruct missing data using SIMD kernel
        rs.reconstruct_data(&mut all_shards, &shard_present).expect("Reconstruction should succeed");

        // 5. Verify byte-exact equality and SHA256 match
        for (d, (actual, expected)) in all_shards.iter().zip(original_data.iter()).enumerate().take(data_shards) {
            assert_eq!(
                actual, expected,
                "Iteration {}: Reconstructed data shred {} differs from original!",
                iter, d
            );
        }

        let mut recovered_hasher = Sha256::new();
        for shard in all_shards.iter().take(data_shards) {
            recovered_hasher.update(shard);
        }
        let recovered_hash = recovered_hasher.finalize();

        assert_eq!(
            original_hash, recovered_hash,
            "Iteration {}: Recovered block SHA-256 does not match original!",
            iter
        );
    }

    println!("✓ 100% of {} randomized erasure recovery scenarios recovered byte-exact payloads!", iterations);
}
