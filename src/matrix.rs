//! Galois field matrix arithmetic and Vandermonde matrix generation for Reed-Solomon.
//!
//! Matches reed-solomon-erasure and Solana Agave wire-format specification exactly.

use crate::galois::{gf_add, gf_div, gf_exp, gf_mul};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u8>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0u8; rows * cols],
        }
    }

    #[inline(always)]
    pub fn get(&self, r: usize, c: usize) -> u8 {
        self.data[r * self.cols + c]
    }

    #[inline(always)]
    pub fn set(&mut self, r: usize, c: usize, val: u8) {
        self.data[r * self.cols + c] = val;
    }

    pub fn identity(size: usize) -> Self {
        let mut m = Self::new(size, size);
        for i in 0..size {
            m.set(i, i, 1);
        }
        m
    }

    pub fn sub_matrix(&self, rmin: usize, cmin: usize, rmax: usize, cmax: usize) -> Self {
        let rows = rmax - rmin;
        let cols = cmax - cmin;
        let mut res = Self::new(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                res.set(r, c, self.get(rmin + r, cmin + c));
            }
        }
        res
    }

    pub fn multiply(&self, rhs: &Matrix) -> Self {
        assert_eq!(self.cols, rhs.rows);
        let mut res = Self::new(self.rows, rhs.cols);
        for r in 0..self.rows {
            for c in 0..rhs.cols {
                let mut acc = 0u8;
                for k in 0..self.cols {
                    acc = gf_add(acc, gf_mul(self.get(r, k), rhs.get(k, c)));
                }
                res.set(r, c, acc);
            }
        }
        res
    }

    /// Generates a Vandermonde matrix in GF(2^8) where V[r, c] = r^c.
    pub fn vandermonde(rows: usize, cols: usize) -> Self {
        let mut m = Self::new(rows, cols);
        for r in 0..rows {
            let r_u8 = r as u8;
            for c in 0..cols {
                m.set(r, c, gf_exp(r_u8, c));
            }
        }
        m
    }

    /// Inverts a square matrix using Gaussian elimination over GF(2^8).
    pub fn invert(&self) -> Result<Self, &'static str> {
        if self.rows != self.cols {
            return Err("Matrix is not square");
        }
        let n = self.rows;
        // Augmented matrix [A | I]
        let mut aug = Matrix::new(n, 2 * n);
        for r in 0..n {
            for c in 0..n {
                aug.set(r, c, self.get(r, c));
            }
            aug.set(r, n + r, 1);
        }

        // Forward elimination
        for r in 0..n {
            // Find pivot
            let mut pivot = r;
            while pivot < n && aug.get(pivot, r) == 0 {
                pivot += 1;
            }
            if pivot == n {
                return Err("Singular matrix");
            }
            // Swap rows if needed
            if pivot != r {
                for c in 0..2 * n {
                    let tmp = aug.get(r, c);
                    aug.set(r, c, aug.get(pivot, c));
                    aug.set(pivot, c, tmp);
                }
            }

            // Scale pivot row so aug[r, r] == 1
            let pivot_val = aug.get(r, r);
            if pivot_val != 1 {
                for c in 0..2 * n {
                    let v = aug.get(r, c);
                    aug.set(r, c, gf_div(v, pivot_val));
                }
            }

            // Eliminate column r in other rows
            for other in 0..n {
                if other != r {
                    let factor = aug.get(other, r);
                    if factor != 0 {
                        for c in 0..2 * n {
                            let sub = gf_mul(factor, aug.get(r, c));
                            let old = aug.get(other, c);
                            aug.set(other, c, gf_add(old, sub));
                        }
                    }
                }
            }
        }

        // Extract right half of augmented matrix
        let mut inv = Matrix::new(n, n);
        for r in 0..n {
            for c in 0..n {
                inv.set(r, c, aug.get(r, n + c));
            }
        }
        Ok(inv)
    }

    /// Builds the canonical Reed-Solomon coding matrix:
    /// Vandermonde(data + parity, data) * (Vandermonde_top)^-1
    pub fn build_coding_matrix(data_shards: usize, parity_shards: usize) -> Result<Self, &'static str> {
        let total_shards = data_shards + parity_shards;
        let v = Self::vandermonde(total_shards, data_shards);
        let top = v.sub_matrix(0, 0, data_shards, data_shards);
        let top_inv = top.invert()?;
        Ok(v.multiply(&top_inv))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reed_solomon_erasure::galois_8::ReedSolomon;

    #[test]
    fn test_matrix_inversion() {
        let mut m = Matrix::new(3, 3);
        m.set(0, 0, 1); m.set(0, 1, 2); m.set(0, 2, 3);
        m.set(1, 0, 4); m.set(1, 1, 5); m.set(1, 2, 6);
        m.set(2, 0, 7); m.set(2, 1, 8); m.set(2, 2, 10);

        let inv = m.invert().expect("Should invert");
        let prod = m.multiply(&inv);
        let id = Matrix::identity(3);
        assert_eq!(prod, id);
    }

    #[test]
    fn test_coding_matrix_matches_reed_solomon_erasure() {
        for &(data, parity) in &[(4, 2), (16, 16), (32, 32), (64, 64)] {
            let our_mat = Matrix::build_coding_matrix(data, parity).unwrap();
            let ref_rs = ReedSolomon::new(data, parity).unwrap();

            // Check that identity is at the top
            for r in 0..data {
                for c in 0..data {
                    let expected = if r == c { 1 } else { 0 };
                    assert_eq!(our_mat.get(r, c), expected);
                }
            }

            // Encode a dummy vector of 1 byte per shard to compare parity outputs
            let data_bytes: Vec<Vec<u8>> = (0..data).map(|i| vec![(i * 7 + 3) as u8]).collect();
            let mut ref_parity: Vec<Vec<u8>> = vec![vec![0u8; 1]; parity];

            let d_slices: Vec<&[u8]> = data_bytes.iter().map(|v| v.as_slice()).collect();
            let mut p_slices: Vec<&mut [u8]> = ref_parity.iter_mut().map(|v| v.as_mut_slice()).collect();
            ref_rs.encode_sep(&d_slices, &mut p_slices).unwrap();

            for (p, ref_p_shard) in ref_parity.iter().enumerate().take(parity) {
                let mut our_p = 0u8;
                for (d, data_byte) in data_bytes.iter().enumerate().take(data) {
                    let coeff = our_mat.get(data + p, d);
                    our_p ^= gf_mul(coeff, data_byte[0]);
                }
                assert_eq!(
                    our_p, ref_p_shard[0],
                    "Mismatch for configuration ({}, {}) at parity shard {}",
                    data, parity, p
                );
            }
        }
    }
}
