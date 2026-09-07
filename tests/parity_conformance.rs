//! Parity Conformance Test Suite: verifies 100% bit-for-bit parity
//! between our native ARM NEON kernel and `reed-solomon-erasure` in Agave master.

use rand::RngCore;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha20Rng;
use reed_solomon_erasure::galois_8::ReedSolomon as RefReedSolomon;
use sha2::{Digest, Sha256};
use solana_turbine_simd::SimdReedSolomon;

const SHRED_SIZE: usize = 1228;

fn run_parity_test_for_config(data_shards: usize, parity_shards: usize, shard_sizes: &[usize]) {
    let mut rng = ChaCha20Rng::seed_from_u64(0x42_DE_AD_BE_EF);

    let our_rs = SimdReedSolomon::new(data_shards, parity_shards).expect("SimdReedSolomon::new failed");
    let ref_rs = RefReedSolomon::new(data_shards, parity_shards).expect("RefReedSolomon::new failed");

    for &size in shard_sizes {
        // Generate random data shreds
        let data_buffers: Vec<Vec<u8>> = (0..data_shards)
            .map(|_| {
                let mut buf = vec![0u8; size];
                rng.fill_bytes(&mut buf);
                buf
            })
            .collect();

        // Parity buffers for our NEON kernel
        let mut our_parity_direct: Vec<Vec<u8>> = vec![vec![0u8; size]; parity_shards];
        let mut our_parity_slice: Vec<Vec<u8>> = vec![vec![0u8; size]; parity_shards];
        let mut our_parity_scalar: Vec<Vec<u8>> = vec![vec![0u8; size]; parity_shards];

        // Parity buffers for reference reed-solomon-erasure
        let mut ref_parity: Vec<Vec<u8>> = vec![vec![0u8; size]; parity_shards];

        let data_slices: Vec<&[u8]> = data_buffers.iter().map(|v| v.as_slice()).collect();

        // 1. Reference Agave encode
        {
            let mut ref_p_slices: Vec<&mut [u8]> = ref_parity.iter_mut().map(|v| v.as_mut_slice()).collect();
            ref_rs.encode_sep(&data_slices, &mut ref_p_slices).unwrap();
        }

        // 2. Our NEON direct register accumulation encode
        {
            let mut our_p_slices: Vec<&mut [u8]> = our_parity_direct.iter_mut().map(|v| v.as_mut_slice()).collect();
            our_rs.encode_sep(&data_slices, &mut our_p_slices).unwrap();
        }

        // 3. Our NEON slice-by-slice encode
        #[cfg(target_arch = "aarch64")]
        {
            let mut our_p_slices: Vec<&mut [u8]> = our_parity_slice.iter_mut().map(|v| v.as_mut_slice()).collect();
            our_rs.encode_sep_neon_slice_by_slice(&data_slices, &mut our_p_slices).unwrap();
        }

        // 4. Our scalar reference encode
        {
            let mut our_p_slices: Vec<&mut [u8]> = our_parity_scalar.iter_mut().map(|v| v.as_mut_slice()).collect();
            our_rs.encode_sep_scalar(&data_slices, &mut our_p_slices).unwrap();
        }

        // Verify byte-for-byte equality and SHA-256 hash equality
        for p in 0..parity_shards {
            let ref_hash = Sha256::digest(&ref_parity[p]);
            let our_direct_hash = Sha256::digest(&our_parity_direct[p]);
            let our_scalar_hash = Sha256::digest(&our_parity_scalar[p]);

            assert_eq!(
                ref_hash, our_direct_hash,
                "Direct NEON SHA256 mismatch: config=({}, {}), size={}, parity_idx={}",
                data_shards, parity_shards, size, p
            );
            assert_eq!(
                ref_parity[p], our_parity_direct[p],
                "Direct NEON byte mismatch: config=({}, {}), size={}, parity_idx={}",
                data_shards, parity_shards, size, p
            );

            assert_eq!(
                ref_hash, our_scalar_hash,
                "Scalar SHA256 mismatch: config=({}, {}), size={}, parity_idx={}",
                data_shards, parity_shards, size, p
            );

            #[cfg(target_arch = "aarch64")]
            {
                let our_slice_hash = Sha256::digest(&our_parity_slice[p]);
                assert_eq!(
                    ref_hash, our_slice_hash,
                    "Slice-by-slice NEON SHA256 mismatch: config=({}, {}), size={}, parity_idx={}",
                    data_shards, parity_shards, size, p
                );
                assert_eq!(
                    ref_parity[p], our_parity_slice[p],
                    "Slice-by-slice NEON byte mismatch: config=({}, {}), size={}, parity_idx={}",
                    data_shards, parity_shards, size, p
                );
            }
        }
    }
}

#[test]
fn test_parity_conformance_canonical_32_32_shreds() {
    println!("Testing canonical Solana (32 data, 32 parity) with 1,228-byte shreds...");
    run_parity_test_for_config(32, 32, &[SHRED_SIZE]);
    println!("✓ Canonical (32, 32) @ 1,228B bit-for-bit parity verified!");
}

#[test]
fn test_parity_conformance_64_64_shreds() {
    println!("Testing (64 data, 64 parity) with 1,228-byte shreds...");
    run_parity_test_for_config(64, 64, &[SHRED_SIZE]);
    println!("✓ (64, 64) @ 1,228B bit-for-bit parity verified!");
}

#[test]
fn test_parity_conformance_16_16_shreds() {
    println!("Testing (16 data, 16 parity) with 1,228-byte shreds...");
    run_parity_test_for_config(16, 16, &[SHRED_SIZE]);
    println!("✓ (16, 16) @ 1,228B bit-for-bit parity verified!");
}

#[test]
fn test_parity_conformance_various_payload_sizes() {
    println!("Testing various shred sizes from 1 byte to 1,228 bytes across configurations...");
    let sizes = [1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 512, 1024, 1216, 1228];
    run_parity_test_for_config(32, 32, &sizes);
    run_parity_test_for_config(16, 16, &sizes);
    println!("✓ Variable sizes bit-for-bit parity verified across all edge boundaries!");
}
