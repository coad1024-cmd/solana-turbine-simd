//! x86_64 AVX2 vector routines for Galois field GF(2^8) arithmetic.
//!
//! Uses 256-bit `_mm256_shuffle_epi8` nibble table lookups matching Solana's canonical
//! Vandermonde Reed-Solomon wire format.

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use core::arch::x86_64::*;

use crate::galois::{MUL_TABLE, TABLE_HI, TABLE_LO};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_mul_slice(c: u8, input: &[u8], out: &mut [u8]) {
    assert_eq!(input.len(), out.len());
    let len = input.len();
    if len == 0 {
        return;
    }
    if c == 0 {
        out.fill(0);
        return;
    }
    if c == 1 {
        out.copy_from_slice(input);
        return;
    }

    let mask_0f = _mm256_set1_epi8(0x0F);
    let tbl_lo_128 = _mm_loadu_si128(TABLE_LO[c as usize].as_ptr() as *const __m128i);
    let tbl_hi_128 = _mm_loadu_si128(TABLE_HI[c as usize].as_ptr() as *const __m128i);
    let tbl_lo = _mm256_broadcastsi128_si256(tbl_lo_128);
    let tbl_hi = _mm256_broadcastsi128_si256(tbl_hi_128);

    let mut in_ptr = input.as_ptr();
    let mut out_ptr = out.as_mut_ptr();
    let mut offset = 0;

    // 64 bytes per iteration (2x 256-bit AVX2 vectors)
    while offset + 64 <= len {
        let in0 = _mm256_loadu_si256(in_ptr.add(offset) as *const __m256i);
        let in1 = _mm256_loadu_si256(in_ptr.add(offset + 32) as *const __m256i);

        let lo0 = _mm256_and_si256(in0, mask_0f);
        let hi0 = _mm256_and_si256(_mm256_srli_epi16(in0, 4), mask_0f);
        let lo1 = _mm256_and_si256(in1, mask_0f);
        let hi1 = _mm256_and_si256(_mm256_srli_epi16(in1, 4), mask_0f);

        let res0 = _mm256_xor_si256(
            _mm256_shuffle_epi8(tbl_lo, lo0),
            _mm256_shuffle_epi8(tbl_hi, hi0),
        );
        let res1 = _mm256_xor_si256(
            _mm256_shuffle_epi8(tbl_lo, lo1),
            _mm256_shuffle_epi8(tbl_hi, hi1),
        );

        _mm256_storeu_si256(out_ptr.add(offset) as *mut __m256i, res0);
        _mm256_storeu_si256(out_ptr.add(offset + 32) as *mut __m256i, res1);

        offset += 64;
    }

    while offset + 32 <= len {
        let in0 = _mm256_loadu_si256(in_ptr.add(offset) as *const __m256i);
        let lo0 = _mm256_and_si256(in0, mask_0f);
        let hi0 = _mm256_and_si256(_mm256_srli_epi16(in0, 4), mask_0f);
        let res0 = _mm256_xor_si256(
            _mm256_shuffle_epi8(tbl_lo, lo0),
            _mm256_shuffle_epi8(tbl_hi, hi0),
        );
        _mm256_storeu_si256(out_ptr.add(offset) as *mut __m256i, res0);
        offset += 32;
    }

    let mt = &MUL_TABLE[c as usize];
    while offset < len {
        *out_ptr.add(offset) = mt[*in_ptr.add(offset) as usize];
        offset += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_mul_slice_xor(c: u8, input: &[u8], out: &mut [u8]) {
    assert_eq!(input.len(), out.len());
    let len = input.len();
    if len == 0 || c == 0 {
        return;
    }
    if c == 1 {
        let mut offset = 0;
        let in_ptr = input.as_ptr();
        let out_ptr = out.as_mut_ptr();
        while offset + 32 <= len {
            let i0 = _mm256_loadu_si256(in_ptr.add(offset) as *const __m256i);
            let o0 = _mm256_loadu_si256(out_ptr.add(offset) as *const __m256i);
            _mm256_storeu_si256(out_ptr.add(offset) as *mut __m256i, _mm256_xor_si256(o0, i0));
            offset += 32;
        }
        while offset < len {
            *out_ptr.add(offset) ^= *in_ptr.add(offset);
            offset += 1;
        }
        return;
    }

    let mask_0f = _mm256_set1_epi8(0x0F);
    let tbl_lo_128 = _mm_loadu_si128(TABLE_LO[c as usize].as_ptr() as *const __m128i);
    let tbl_hi_128 = _mm_loadu_si128(TABLE_HI[c as usize].as_ptr() as *const __m128i);
    let tbl_lo = _mm256_broadcastsi128_si256(tbl_lo_128);
    let tbl_hi = _mm256_broadcastsi128_si256(tbl_hi_128);

    let mut in_ptr = input.as_ptr();
    let mut out_ptr = out.as_mut_ptr();
    let mut offset = 0;

    while offset + 64 <= len {
        let in0 = _mm256_loadu_si256(in_ptr.add(offset) as *const __m256i);
        let in1 = _mm256_loadu_si256(in_ptr.add(offset + 32) as *const __m256i);

        let lo0 = _mm256_and_si256(in0, mask_0f);
        let hi0 = _mm256_and_si256(_mm256_srli_epi16(in0, 4), mask_0f);
        let lo1 = _mm256_and_si256(in1, mask_0f);
        let hi1 = _mm256_and_si256(_mm256_srli_epi16(in1, 4), mask_0f);

        let res0 = _mm256_xor_si256(
            _mm256_shuffle_epi8(tbl_lo, lo0),
            _mm256_shuffle_epi8(tbl_hi, hi0),
        );
        let res1 = _mm256_xor_si256(
            _mm256_shuffle_epi8(tbl_lo, lo1),
            _mm256_shuffle_epi8(tbl_hi, hi1),
        );

        let cur0 = _mm256_loadu_si256(out_ptr.add(offset) as *const __m256i);
        let cur1 = _mm256_loadu_si256(out_ptr.add(offset + 32) as *const __m256i);

        _mm256_storeu_si256(
            out_ptr.add(offset) as *mut __m256i,
            _mm256_xor_si256(cur0, res0),
        );
        _mm256_storeu_si256(
            out_ptr.add(offset + 32) as *mut __m256i,
            _mm256_xor_si256(cur1, res1),
        );

        offset += 64;
    }

    while offset + 32 <= len {
        let in0 = _mm256_loadu_si256(in_ptr.add(offset) as *const __m256i);
        let lo0 = _mm256_and_si256(in0, mask_0f);
        let hi0 = _mm256_and_si256(_mm256_srli_epi16(in0, 4), mask_0f);
        let res0 = _mm256_xor_si256(
            _mm256_shuffle_epi8(tbl_lo, lo0),
            _mm256_shuffle_epi8(tbl_hi, hi0),
        );
        let cur0 = _mm256_loadu_si256(out_ptr.add(offset) as *const __m256i);
        _mm256_storeu_si256(
            out_ptr.add(offset) as *mut __m256i,
            _mm256_xor_si256(cur0, res0),
        );
        offset += 32;
    }

    let mt = &MUL_TABLE[c as usize];
    while offset < len {
        *out_ptr.add(offset) ^= mt[*in_ptr.add(offset) as usize];
        offset += 1;
    }
}
