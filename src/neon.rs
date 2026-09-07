//! Native ARM64 NEON vector routines for Galois field GF(2^8) arithmetic.
//!
//! Uses 128-bit `vqtbl1q_u8` nibble table lookups matching Solana's canonical
//! Vandermonde Reed-Solomon wire format.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use crate::galois::{MUL_TABLE, TABLE_HI, TABLE_LO};

/// Vectorized multiplication of a byte slice by a Galois field constant `c`:
/// `out[i] = c * input[i]`
///
/// Handles arbitrary slice lengths with 64-byte unrolling and scalar tail.
#[cfg(target_arch = "aarch64")]
pub fn neon_mul_slice(c: u8, input: &[u8], out: &mut [u8]) {
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

    unsafe {
        let tbl_lo = vld1q_u8(TABLE_LO[c as usize].as_ptr());
        let tbl_hi = vld1q_u8(TABLE_HI[c as usize].as_ptr());
        let mask_0f = vdupq_n_u8(0x0F);

        let in_ptr = input.as_ptr();
        let out_ptr = out.as_mut_ptr();
        let mut offset = 0;

        // 64-byte unrolled loop (4x 128-bit NEON vectors)
        while offset + 64 <= len {
            let in0 = vld1q_u8(in_ptr.add(offset));
            let in1 = vld1q_u8(in_ptr.add(offset + 16));
            let in2 = vld1q_u8(in_ptr.add(offset + 32));
            let in3 = vld1q_u8(in_ptr.add(offset + 48));

            let lo0 = vandq_u8(in0, mask_0f);
            let hi0 = vshrq_n_u8(in0, 4);
            let lo1 = vandq_u8(in1, mask_0f);
            let hi1 = vshrq_n_u8(in1, 4);
            let lo2 = vandq_u8(in2, mask_0f);
            let hi2 = vshrq_n_u8(in2, 4);
            let lo3 = vandq_u8(in3, mask_0f);
            let hi3 = vshrq_n_u8(in3, 4);

            let res0 = veorq_u8(vqtbl1q_u8(tbl_lo, lo0), vqtbl1q_u8(tbl_hi, hi0));
            let res1 = veorq_u8(vqtbl1q_u8(tbl_lo, lo1), vqtbl1q_u8(tbl_hi, hi1));
            let res2 = veorq_u8(vqtbl1q_u8(tbl_lo, lo2), vqtbl1q_u8(tbl_hi, hi2));
            let res3 = veorq_u8(vqtbl1q_u8(tbl_lo, lo3), vqtbl1q_u8(tbl_hi, hi3));

            vst1q_u8(out_ptr.add(offset), res0);
            vst1q_u8(out_ptr.add(offset + 16), res1);
            vst1q_u8(out_ptr.add(offset + 32), res2);
            vst1q_u8(out_ptr.add(offset + 48), res3);

            offset += 64;
        }

        // 16-byte remainder loop
        while offset + 16 <= len {
            let in0 = vld1q_u8(in_ptr.add(offset));
            let lo0 = vandq_u8(in0, mask_0f);
            let hi0 = vshrq_n_u8(in0, 4);
            let res0 = veorq_u8(vqtbl1q_u8(tbl_lo, lo0), vqtbl1q_u8(tbl_hi, hi0));
            vst1q_u8(out_ptr.add(offset), res0);
            offset += 16;
        }

        // Scalar tail (< 16 bytes, e.g. 12 bytes for 1228-byte shreds)
        let mt = &MUL_TABLE[c as usize];
        while offset < len {
            *out_ptr.add(offset) = mt[*in_ptr.add(offset) as usize];
            offset += 1;
        }
    }
}

/// Vectorized multiplication and XOR accumulation into a byte slice:
/// `out[i] ^= c * input[i]`
///
/// Handles arbitrary slice lengths with 64-byte unrolling and scalar tail.
#[cfg(target_arch = "aarch64")]
pub fn neon_mul_slice_xor(c: u8, input: &[u8], out: &mut [u8]) {
    assert_eq!(input.len(), out.len());
    let len = input.len();
    if len == 0 || c == 0 {
        return;
    }
    if c == 1 {
        neon_slice_xor(input, out);
        return;
    }

    unsafe {
        let tbl_lo = vld1q_u8(TABLE_LO[c as usize].as_ptr());
        let tbl_hi = vld1q_u8(TABLE_HI[c as usize].as_ptr());
        let mask_0f = vdupq_n_u8(0x0F);

        let in_ptr = input.as_ptr();
        let out_ptr = out.as_mut_ptr();
        let mut offset = 0;

        // 64-byte unrolled loop (4x 128-bit NEON vectors)
        while offset + 64 <= len {
            let in0 = vld1q_u8(in_ptr.add(offset));
            let in1 = vld1q_u8(in_ptr.add(offset + 16));
            let in2 = vld1q_u8(in_ptr.add(offset + 32));
            let in3 = vld1q_u8(in_ptr.add(offset + 48));

            let lo0 = vandq_u8(in0, mask_0f);
            let hi0 = vshrq_n_u8(in0, 4);
            let lo1 = vandq_u8(in1, mask_0f);
            let hi1 = vshrq_n_u8(in1, 4);
            let lo2 = vandq_u8(in2, mask_0f);
            let hi2 = vshrq_n_u8(in2, 4);
            let lo3 = vandq_u8(in3, mask_0f);
            let hi3 = vshrq_n_u8(in3, 4);

            let res0 = veorq_u8(vqtbl1q_u8(tbl_lo, lo0), vqtbl1q_u8(tbl_hi, hi0));
            let res1 = veorq_u8(vqtbl1q_u8(tbl_lo, lo1), vqtbl1q_u8(tbl_hi, hi1));
            let res2 = veorq_u8(vqtbl1q_u8(tbl_lo, lo2), vqtbl1q_u8(tbl_hi, hi2));
            let res3 = veorq_u8(vqtbl1q_u8(tbl_lo, lo3), vqtbl1q_u8(tbl_hi, hi3));

            let cur0 = vld1q_u8(out_ptr.add(offset));
            let cur1 = vld1q_u8(out_ptr.add(offset + 16));
            let cur2 = vld1q_u8(out_ptr.add(offset + 32));
            let cur3 = vld1q_u8(out_ptr.add(offset + 48));

            vst1q_u8(out_ptr.add(offset), veorq_u8(cur0, res0));
            vst1q_u8(out_ptr.add(offset + 16), veorq_u8(cur1, res1));
            vst1q_u8(out_ptr.add(offset + 32), veorq_u8(cur2, res2));
            vst1q_u8(out_ptr.add(offset + 48), veorq_u8(cur3, res3));

            offset += 64;
        }

        // 16-byte remainder loop
        while offset + 16 <= len {
            let in0 = vld1q_u8(in_ptr.add(offset));
            let lo0 = vandq_u8(in0, mask_0f);
            let hi0 = vshrq_n_u8(in0, 4);
            let res0 = veorq_u8(vqtbl1q_u8(tbl_lo, lo0), vqtbl1q_u8(tbl_hi, hi0));

            let cur0 = vld1q_u8(out_ptr.add(offset));
            vst1q_u8(out_ptr.add(offset), veorq_u8(cur0, res0));
            offset += 16;
        }

        // Scalar tail (< 16 bytes)
        let mt = &MUL_TABLE[c as usize];
        while offset < len {
            *out_ptr.add(offset) ^= mt[*in_ptr.add(offset) as usize];
            offset += 1;
        }
    }
}

/// Vectorized XOR: out[i] ^= input[i]
#[cfg(target_arch = "aarch64")]
#[inline]
pub fn neon_slice_xor(input: &[u8], out: &mut [u8]) {
    assert_eq!(input.len(), out.len());
    let len = input.len();
    unsafe {
        let in_ptr = input.as_ptr();
        let out_ptr = out.as_mut_ptr();
        let mut offset = 0;

        while offset + 64 <= len {
            let i0 = vld1q_u8(in_ptr.add(offset));
            let i1 = vld1q_u8(in_ptr.add(offset + 16));
            let i2 = vld1q_u8(in_ptr.add(offset + 32));
            let i3 = vld1q_u8(in_ptr.add(offset + 48));

            let o0 = vld1q_u8(out_ptr.add(offset));
            let o1 = vld1q_u8(out_ptr.add(offset + 16));
            let o2 = vld1q_u8(out_ptr.add(offset + 32));
            let o3 = vld1q_u8(out_ptr.add(offset + 48));

            vst1q_u8(out_ptr.add(offset), veorq_u8(o0, i0));
            vst1q_u8(out_ptr.add(offset + 16), veorq_u8(o1, i1));
            vst1q_u8(out_ptr.add(offset + 32), veorq_u8(o2, i2));
            vst1q_u8(out_ptr.add(offset + 48), veorq_u8(o3, i3));

            offset += 64;
        }

        while offset + 16 <= len {
            let i0 = vld1q_u8(in_ptr.add(offset));
            let o0 = vld1q_u8(out_ptr.add(offset));
            vst1q_u8(out_ptr.add(offset), veorq_u8(o0, i0));
            offset += 16;
        }

        while offset < len {
            *out_ptr.add(offset) ^= *in_ptr.add(offset);
            offset += 1;
        }
    }
}

/// Computes a full parity shred directly into `out_parity` by keeping accumulation
/// in vector registers over chunks, eliminating repetitive RAM/cache roundtrips.
///
/// `parity_row` has length `K` (number of data shreds).
/// `data_shreds` is an array of slices, each of length `len`.
#[cfg(target_arch = "aarch64")]
pub fn neon_encode_parity_shred(
    parity_row: &[u8],
    data_shreds: &[&[u8]],
    out_parity: &mut [u8],
) {
    let k = parity_row.len();
    assert_eq!(k, data_shreds.len());
    let len = out_parity.len();
    if len == 0 || k == 0 {
        return;
    }

    unsafe {
        let mask_0f = vdupq_n_u8(0x0F);
        let out_ptr = out_parity.as_mut_ptr();
        let mut offset = 0;

        // Process 64 bytes of output across all K inputs
        while offset + 64 <= len {
            let mut acc0 = vdupq_n_u8(0);
            let mut acc1 = vdupq_n_u8(0);
            let mut acc2 = vdupq_n_u8(0);
            let mut acc3 = vdupq_n_u8(0);

            for i in 0..k {
                let c = parity_row[i];
                if c == 0 {
                    continue;
                }
                let in_ptr = data_shreds[i].as_ptr();
                let in0 = vld1q_u8(in_ptr.add(offset));
                let in1 = vld1q_u8(in_ptr.add(offset + 16));
                let in2 = vld1q_u8(in_ptr.add(offset + 32));
                let in3 = vld1q_u8(in_ptr.add(offset + 48));

                if c == 1 {
                    acc0 = veorq_u8(acc0, in0);
                    acc1 = veorq_u8(acc1, in1);
                    acc2 = veorq_u8(acc2, in2);
                    acc3 = veorq_u8(acc3, in3);
                } else {
                    let tbl_lo = vld1q_u8(TABLE_LO[c as usize].as_ptr());
                    let tbl_hi = vld1q_u8(TABLE_HI[c as usize].as_ptr());

                    let lo0 = vandq_u8(in0, mask_0f);
                    let hi0 = vshrq_n_u8(in0, 4);
                    let lo1 = vandq_u8(in1, mask_0f);
                    let hi1 = vshrq_n_u8(in1, 4);
                    let lo2 = vandq_u8(in2, mask_0f);
                    let hi2 = vshrq_n_u8(in2, 4);
                    let lo3 = vandq_u8(in3, mask_0f);
                    let hi3 = vshrq_n_u8(in3, 4);

                    let res0 = veorq_u8(vqtbl1q_u8(tbl_lo, lo0), vqtbl1q_u8(tbl_hi, hi0));
                    let res1 = veorq_u8(vqtbl1q_u8(tbl_lo, lo1), vqtbl1q_u8(tbl_hi, hi1));
                    let res2 = veorq_u8(vqtbl1q_u8(tbl_lo, lo2), vqtbl1q_u8(tbl_hi, hi2));
                    let res3 = veorq_u8(vqtbl1q_u8(tbl_lo, lo3), vqtbl1q_u8(tbl_hi, hi3));

                    acc0 = veorq_u8(acc0, res0);
                    acc1 = veorq_u8(acc1, res1);
                    acc2 = veorq_u8(acc2, res2);
                    acc3 = veorq_u8(acc3, res3);
                }
            }

            vst1q_u8(out_ptr.add(offset), acc0);
            vst1q_u8(out_ptr.add(offset + 16), acc1);
            vst1q_u8(out_ptr.add(offset + 32), acc2);
            vst1q_u8(out_ptr.add(offset + 48), acc3);

            offset += 64;
        }

        // 16-byte remainder loop
        while offset + 16 <= len {
            let mut acc0 = vdupq_n_u8(0);

            for i in 0..k {
                let c = parity_row[i];
                if c == 0 {
                    continue;
                }
                let in_ptr = data_shreds[i].as_ptr();
                let in0 = vld1q_u8(in_ptr.add(offset));

                if c == 1 {
                    acc0 = veorq_u8(acc0, in0);
                } else {
                    let tbl_lo = vld1q_u8(TABLE_LO[c as usize].as_ptr());
                    let tbl_hi = vld1q_u8(TABLE_HI[c as usize].as_ptr());
                    let lo0 = vandq_u8(in0, mask_0f);
                    let hi0 = vshrq_n_u8(in0, 4);
                    let res0 = veorq_u8(vqtbl1q_u8(tbl_lo, lo0), vqtbl1q_u8(tbl_hi, hi0));
                    acc0 = veorq_u8(acc0, res0);
                }
            }

            vst1q_u8(out_ptr.add(offset), acc0);
            offset += 16;
        }

        // Scalar tail (< 16 bytes)
        while offset < len {
            let mut acc = 0u8;
            for i in 0..k {
                let c = parity_row[i];
                if c != 0 {
                    let b = data_shreds[i][offset];
                    acc ^= MUL_TABLE[c as usize][b as usize];
                }
            }
            *out_ptr.add(offset) = acc;
            offset += 1;
        }
    }
}
