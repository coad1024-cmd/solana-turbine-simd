//! Galois Field GF(2^8) arithmetic matching Solana Turbine canonical wire format.
//!
//! Polynomial: p(x) = x^8 + x^4 + x^3 + x^2 + 1 (0x11D, generator 29)

pub const FIELD_SIZE: usize = 256;
pub const GENERATING_POLYNOMIAL: usize = 29;

/// Computes the discrete logarithm table for GF(2^8).
pub const fn gen_log_table() -> [u8; FIELD_SIZE] {
    let mut log = [0u8; FIELD_SIZE];
    let mut b: usize = 1;
    let mut i = 0;
    while i < FIELD_SIZE - 1 {
        log[b] = i as u8;
        b <<= 1;
        if b >= FIELD_SIZE {
            b = (b - FIELD_SIZE) ^ GENERATING_POLYNOMIAL;
        }
        i += 1;
    }
    log
}

pub const EXP_TABLE_SIZE: usize = FIELD_SIZE * 2 - 2;

/// Computes the exponential (anti-log) table for GF(2^8).
pub const fn gen_exp_table(log_table: &[u8; FIELD_SIZE]) -> [u8; EXP_TABLE_SIZE] {
    let mut exp = [0u8; EXP_TABLE_SIZE];
    let mut i = 1;
    while i < FIELD_SIZE {
        let log = log_table[i] as usize;
        exp[log] = i as u8;
        exp[log + FIELD_SIZE - 1] = i as u8;
        i += 1;
    }
    exp
}

/// Computes the full 256x256 multiplication table for GF(2^8).
pub const fn gen_mul_table(
    log_table: &[u8; FIELD_SIZE],
    exp_table: &[u8; EXP_TABLE_SIZE],
) -> [[u8; FIELD_SIZE]; FIELD_SIZE] {
    let mut mul = [[0u8; FIELD_SIZE]; FIELD_SIZE];
    let mut a = 0;
    while a < FIELD_SIZE {
        let mut b = 0;
        while b < FIELD_SIZE {
            if a == 0 || b == 0 {
                mul[a][b] = 0;
            } else {
                let log_a = log_table[a] as usize;
                let log_b = log_table[b] as usize;
                mul[a][b] = exp_table[log_a + log_b];
            }
            b += 1;
        }
        a += 1;
    }
    mul
}

/// Generates 16-byte low and high nibble tables for SIMD vector lookup.
///
/// For each constant coefficient `c` and nibble `x_lo` in 0..16:
///   `TABLE_LO[c][x_lo] = c * x_lo`
/// For each constant coefficient `c` and nibble `x_hi` in 0..16:
///   `TABLE_HI[c][x_hi] = c * (x_hi << 4)`
pub const fn gen_nibble_tables(
    mul_table: &[[u8; FIELD_SIZE]; FIELD_SIZE],
) -> ([[u8; 16]; FIELD_SIZE], [[u8; 16]; FIELD_SIZE]) {
    let mut low = [[0u8; 16]; FIELD_SIZE];
    let mut high = [[0u8; 16]; FIELD_SIZE];

    let mut c = 0;
    while c < FIELD_SIZE {
        let mut nibble = 0;
        while nibble < 16 {
            low[c][nibble] = mul_table[c][nibble];
            high[c][nibble] = mul_table[c][nibble << 4];
            nibble += 1;
        }
        c += 1;
    }
    (low, high)
}

pub static LOG_TABLE: [u8; FIELD_SIZE] = gen_log_table();
pub static EXP_TABLE: [u8; EXP_TABLE_SIZE] = gen_exp_table(&LOG_TABLE);
pub static MUL_TABLE: [[u8; FIELD_SIZE]; FIELD_SIZE] = gen_mul_table(&LOG_TABLE, &EXP_TABLE);

// Precomputed 16-byte nibble tables for 128-bit NEON vqtbl1q_u8 and AVX2 shuffle
pub static NIBBLE_TABLES: ([[u8; 16]; FIELD_SIZE], [[u8; 16]; FIELD_SIZE]) =
    gen_nibble_tables(&MUL_TABLE);

pub static TABLE_LO: [[u8; 16]; FIELD_SIZE] = NIBBLE_TABLES.0;
pub static TABLE_HI: [[u8; 16]; FIELD_SIZE] = NIBBLE_TABLES.1;

/// Multiply two elements in GF(2^8).
#[inline(always)]
pub fn gf_mul(a: u8, b: u8) -> u8 {
    MUL_TABLE[a as usize][b as usize]
}

/// Add two elements in GF(2^8).
#[inline(always)]
pub fn gf_add(a: u8, b: u8) -> u8 {
    a ^ b
}

/// Subtract two elements in GF(2^8).
#[inline(always)]
pub fn gf_sub(a: u8, b: u8) -> u8 {
    a ^ b
}

/// Divide two elements in GF(2^8).
#[inline(always)]
pub fn gf_div(a: u8, b: u8) -> u8 {
    if a == 0 {
        0
    } else if b == 0 {
        panic!("Divisor is 0 in GF(2^8)");
    } else {
        let log_a = LOG_TABLE[a as usize] as isize;
        let log_b = LOG_TABLE[b as usize] as isize;
        let mut diff = log_a - log_b;
        if diff < 0 {
            diff += 255;
        }
        EXP_TABLE[diff as usize]
    }
}

/// Exponentiation in GF(2^8): a^n.
#[inline(always)]
pub fn gf_exp(a: u8, n: usize) -> u8 {
    if n == 0 {
        1
    } else if a == 0 {
        0
    } else {
        let log_a = LOG_TABLE[a as usize] as usize;
        let mut log_res = log_a * n;
        while log_res >= 255 {
            log_res -= 255;
        }
        EXP_TABLE[log_res]
    }
}

/// Invert an element in GF(2^8).
#[inline(always)]
pub fn gf_inv(a: u8) -> u8 {
    if a == 0 {
        panic!("Cannot invert 0 in GF(2^8)");
    }
    let log_a = LOG_TABLE[a as usize] as usize;
    EXP_TABLE[255 - log_a]
}

/// Scalar fallback multiplication of a slice: out = c * input
pub fn scalar_mul_slice(c: u8, input: &[u8], out: &mut [u8]) {
    assert_eq!(input.len(), out.len());
    if c == 0 {
        out.fill(0);
        return;
    }
    if c == 1 {
        out.copy_from_slice(input);
        return;
    }
    let mt = &MUL_TABLE[c as usize];
    for (i, &b) in input.iter().enumerate() {
        out[i] = mt[b as usize];
    }
}

/// Scalar fallback multiplication and XOR: out ^= c * input
pub fn scalar_mul_slice_xor(c: u8, input: &[u8], out: &mut [u8]) {
    assert_eq!(input.len(), out.len());
    if c == 0 {
        return;
    }
    if c == 1 {
        for (i, &b) in input.iter().enumerate() {
            out[i] ^= b;
        }
        return;
    }
    let mt = &MUL_TABLE[c as usize];
    for (i, &b) in input.iter().enumerate() {
        out[i] ^= mt[b as usize];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reed_solomon_erasure::galois_8;

    #[test]
    fn test_galois_tables_parity() {
        for a in 0..256 {
            for b in 0..256 {
                let our_mul = gf_mul(a as u8, b as u8);
                let ref_mul = galois_8::mul(a as u8, b as u8);
                assert_eq!(
                    our_mul, ref_mul,
                    "Multiplication mismatch for a={}, b={}",
                    a, b
                );
            }
            if a > 0 {
                let our_inv = gf_inv(a as u8);
                let ref_div = galois_8::div(1, a as u8);
                assert_eq!(our_inv, ref_div, "Inverse mismatch for a={}", a);
            }
        }
    }

    #[test]
    fn test_nibble_decomposition() {
        for c in 0..256 {
            for x in 0..256 {
                let x_u8 = x as u8;
                let lo = (x_u8 & 0x0F) as usize;
                let hi = (x_u8 >> 4) as usize;

                let prod_lo = TABLE_LO[c][lo];
                let prod_hi = TABLE_HI[c][hi];
                let combined = prod_lo ^ prod_hi;

                let expected = gf_mul(c as u8, x_u8);
                assert_eq!(
                    combined, expected,
                    "Nibble decomposition failed for c={}, x={}",
                    c, x
                );
            }
        }
    }
}
