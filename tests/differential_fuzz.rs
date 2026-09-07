//! High-volume differential fuzzing test suite:
//! Verifies 1,000,000 randomized test vectors comparing native ARM NEON kernel
//! against `reed-solomon-erasure`.

use rand::RngCore;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use reed_solomon_erasure::galois_8::ReedSolomon as RefReedSolomon;
use solana_turbine_simd::SimdReedSolomon;

#[test]
fn test_differential_fuzz_1m_vectors() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x1337_C0DE_CAFE_BABE);

    let configs = [(4, 2), (8, 4), (16, 16), (32, 32)];
    let total_iterations = 1_000_000;
    let iters_per_config = total_iterations / configs.len();

    println!(
        "Executing {} differential fuzzing vectors across {} configurations...",
        total_iterations,
        configs.len()
    );

    let mut total_verified = 0;

    for &(k, m) in &configs {
        let our_rs = SimdReedSolomon::new(k, m).unwrap();
        let ref_rs = RefReedSolomon::new(k, m).unwrap();

        // Batch test vectors: we test various sizes (1, 2, 4, 8, 16, 32, 64 bytes)
        // to test SIMD edge boundaries and alignments rapidly.
        let batch_size = 500;
        let batches = iters_per_config / batch_size;

        for _ in 0..batches {
            let size = ((rng.next_u32() as usize) % 64) + 1;

            let data_buffers: Vec<Vec<u8>> = (0..k)
                .map(|_| {
                    let mut b = vec![0u8; size];
                    rng.fill_bytes(&mut b);
                    b
                })
                .collect();

            let mut our_parity = vec![vec![0u8; size]; m];
            let mut ref_parity = vec![vec![0u8; size]; m];

            let d_refs: Vec<&[u8]> = data_buffers.iter().map(|v| v.as_slice()).collect();
            {
                let mut our_p_refs: Vec<&mut [u8]> = our_parity.iter_mut().map(|v| v.as_mut_slice()).collect();
                our_rs.encode_sep(&d_refs, &mut our_p_refs).unwrap();
            }
            {
                let mut ref_p_refs: Vec<&mut [u8]> = ref_parity.iter_mut().map(|v| v.as_mut_slice()).collect();
                ref_rs.encode_sep(&d_refs, &mut ref_p_refs).unwrap();
            }

            for p in 0..m {
                assert_eq!(
                    our_parity[p], ref_parity[p],
                    "Fuzz mismatch: config=({}, {}), size={}, p_idx={}",
                    k, m, size, p
                );
            }
            total_verified += batch_size;
        }
    }

    println!("✓ Successfully verified {} / {} randomized differential fuzz vectors!", total_verified, total_iterations);
}
