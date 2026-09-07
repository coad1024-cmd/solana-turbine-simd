//! Wire-compatible Reed-Solomon Erasure Engine with native ARM NEON and AVX2 acceleration.

use crate::galois::{scalar_mul_slice, scalar_mul_slice_xor};
use crate::matrix::Matrix;

#[cfg(target_arch = "aarch64")]
use crate::neon::{neon_encode_parity_shred, neon_mul_slice, neon_mul_slice_xor};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::avx2::{avx2_mul_slice, avx2_mul_slice_xor};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    TooFewDataShards,
    TooFewParityShards,
    TooManyShards,
    EmptyShard,
    IncorrectShardSize,
    TooFewShardsPresent,
    SingularMatrix,
    InvalidIndex,
}

#[derive(Debug, Clone)]
pub struct SimdReedSolomon {
    data_shards: usize,
    parity_shards: usize,
    total_shards: usize,
    coding_matrix: Matrix,
    parity_rows: Vec<Vec<u8>>,
}

impl SimdReedSolomon {
    /// Creates a new Reed-Solomon instance.
    pub fn new(data_shards: usize, parity_shards: usize) -> Result<Self, Error> {
        if data_shards == 0 {
            return Err(Error::TooFewDataShards);
        }
        if parity_shards == 0 {
            return Err(Error::TooFewParityShards);
        }
        if data_shards + parity_shards > 256 {
            return Err(Error::TooManyShards);
        }

        let total_shards = data_shards + parity_shards;
        let coding_matrix = Matrix::build_coding_matrix(data_shards, parity_shards)
            .map_err(|_| Error::SingularMatrix)?;

        let mut parity_rows = Vec::with_capacity(parity_shards);
        for p in 0..parity_shards {
            let row_idx = data_shards + p;
            let mut row = Vec::with_capacity(data_shards);
            for d in 0..data_shards {
                row.push(coding_matrix.get(row_idx, d));
            }
            parity_rows.push(row);
        }

        Ok(Self {
            data_shards,
            parity_shards,
            total_shards,
            coding_matrix,
            parity_rows,
        })
    }

    #[inline(always)]
    pub fn data_shard_count(&self) -> usize {
        self.data_shards
    }

    #[inline(always)]
    pub fn parity_shard_count(&self) -> usize {
        self.parity_shards
    }

    #[inline(always)]
    pub fn total_shard_count(&self) -> usize {
        self.total_shards
    }

    pub fn parity_rows(&self) -> &[Vec<u8>] {
        &self.parity_rows
    }

    fn validate_shards<T: AsRef<[u8]>, U: AsMut<[u8]>>(
        &self,
        data: &[T],
        parity: &mut [U],
    ) -> Result<usize, Error> {
        if data.len() != self.data_shards {
            return Err(Error::TooFewDataShards);
        }
        if parity.len() != self.parity_shards {
            return Err(Error::TooFewParityShards);
        }

        let shard_len = data[0].as_ref().len();
        if shard_len == 0 {
            return Err(Error::EmptyShard);
        }
        for d in data.iter() {
            if d.as_ref().len() != shard_len {
                return Err(Error::IncorrectShardSize);
            }
        }
        for p in parity.iter_mut() {
            if p.as_mut().len() != shard_len {
                return Err(Error::IncorrectShardSize);
            }
        }
        Ok(shard_len)
    }

    /// Primary high-throughput encode entrypoint for separate data and parity buffers.
    /// Uses the fastest hardware-vectorized path available for the host CPU.
    pub fn encode_sep<T: AsRef<[u8]>, U: AsMut<[u8]>>(
        &self,
        data: &[T],
        parity: &mut [U],
    ) -> Result<(), Error> {
        self.validate_shards(data, parity)?;

        #[cfg(target_arch = "aarch64")]
        {
            self.encode_sep_neon_slice_by_slice(data, parity)
        }

        #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(target_arch = "aarch64")))]
        {
            if is_x86_feature_detected!("avx2") {
                self.encode_sep_avx2(data, parity)
            } else {
                self.encode_sep_scalar(data, parity)
            }
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
        {
            self.encode_sep_scalar(data, parity)
        }
    }

    /// Pure scalar encoder (portable reference).
    pub fn encode_sep_scalar<T: AsRef<[u8]>, U: AsMut<[u8]>>(
        &self,
        data: &[T],
        parity: &mut [U],
    ) -> Result<(), Error> {
        self.validate_shards(data, parity)?;
        for (p_idx, p_buf) in parity.iter_mut().enumerate() {
            let out = p_buf.as_mut();
            let row = &self.parity_rows[p_idx];
            scalar_mul_slice(row[0], data[0].as_ref(), out);
            for d in 1..self.data_shards {
                scalar_mul_slice_xor(row[d], data[d].as_ref(), out);
            }
        }
        Ok(())
    }

    /// Direct register accumulation NEON encoder (fastest on ARM64).
    #[cfg(target_arch = "aarch64")]
    pub fn encode_sep_neon<T: AsRef<[u8]>, U: AsMut<[u8]>>(
        &self,
        data: &[T],
        parity: &mut [U],
    ) -> Result<(), Error> {
        self.validate_shards(data, parity)?;
        let data_refs: Vec<&[u8]> = data.iter().map(|d| d.as_ref()).collect();
        for (p_idx, p_buf) in parity.iter_mut().enumerate() {
            neon_encode_parity_shred(&self.parity_rows[p_idx], &data_refs, p_buf.as_mut());
        }
        Ok(())
    }

    /// Slice-by-slice NEON encoder (exact analogue to Agave's loop using vqtbl1q_u8).
    #[cfg(target_arch = "aarch64")]
    pub fn encode_sep_neon_slice_by_slice<T: AsRef<[u8]>, U: AsMut<[u8]>>(
        &self,
        data: &[T],
        parity: &mut [U],
    ) -> Result<(), Error> {
        self.validate_shards(data, parity)?;
        for (p_idx, p_buf) in parity.iter_mut().enumerate() {
            let out = p_buf.as_mut();
            let row = &self.parity_rows[p_idx];
            neon_mul_slice(row[0], data[0].as_ref(), out);
            for d in 1..self.data_shards {
                neon_mul_slice_xor(row[d], data[d].as_ref(), out);
            }
        }
        Ok(())
    }

    /// AVX2 encoder for x86_64.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    pub fn encode_sep_avx2<T: AsRef<[u8]>, U: AsMut<[u8]>>(
        &self,
        data: &[T],
        parity: &mut [U],
    ) -> Result<(), Error> {
        self.validate_shards(data, parity)?;
        for (p_idx, p_buf) in parity.iter_mut().enumerate() {
            let out = p_buf.as_mut();
            let row = &self.parity_rows[p_idx];
            unsafe {
                avx2_mul_slice(row[0], data[0].as_ref(), out);
                for d in 1..self.data_shards {
                    avx2_mul_slice_xor(row[d], data[d].as_ref(), out);
                }
            }
        }
        Ok(())
    }

    /// Reconstructs missing data shards in-place from available data and parity shards.
    pub fn reconstruct_data<T: AsRef<[u8]> + AsMut<[u8]>>(
        &self,
        shards: &mut [T],
        shard_present: &[bool],
    ) -> Result<(), Error> {
        if shards.len() != self.total_shards || shard_present.len() != self.total_shards {
            return Err(Error::IncorrectShardSize);
        }

        let shard_len = shards[0].as_ref().len();
        if shard_len == 0 {
            return Err(Error::EmptyShard);
        }
        for s in shards.iter() {
            if s.as_ref().len() != shard_len {
                return Err(Error::IncorrectShardSize);
            }
        }

        // Check if all data shards are already present
        let mut missing_data = Vec::new();
        for (i, &present) in shard_present.iter().enumerate().take(self.data_shards) {
            if !present {
                missing_data.push(i);
            }
        }
        if missing_data.is_empty() {
            return Ok(());
        }

        // Count available shards
        let mut available_indices = Vec::new();
        for (i, &present) in shard_present.iter().enumerate() {
            if present {
                available_indices.push(i);
                if available_indices.len() == self.data_shards {
                    break;
                }
            }
        }
        if available_indices.len() < self.data_shards {
            return Err(Error::TooFewShardsPresent);
        }

        // Build decode matrix from the available rows of the coding matrix
        let mut sub_mat = Matrix::new(self.data_shards, self.data_shards);
        for (r, &src_row) in available_indices.iter().enumerate() {
            for c in 0..self.data_shards {
                sub_mat.set(r, c, self.coding_matrix.get(src_row, c));
            }
        }
        let inv_decode_mat = sub_mat.invert().map_err(|_| Error::SingularMatrix)?;

        // Prepare available shard buffers
        let available_shards: Vec<Vec<u8>> = available_indices
            .iter()
            .map(|&idx| shards[idx].as_ref().to_vec())
            .collect();
        let avail_refs: Vec<&[u8]> = available_shards.iter().map(|v| v.as_slice()).collect();

        // For each missing data shard, compute its value from available shards
        for &missing_idx in &missing_data {
            let mut row = Vec::with_capacity(self.data_shards);
            for c in 0..self.data_shards {
                row.push(inv_decode_mat.get(missing_idx, c));
            }

            let out_buf = shards[missing_idx].as_mut();

            #[cfg(target_arch = "aarch64")]
            {
                neon_encode_parity_shred(&row, &avail_refs, out_buf);
            }

            #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), not(target_arch = "aarch64")))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        avx2_mul_slice(row[0], avail_refs[0], out_buf);
                        for d in 1..self.data_shards {
                            avx2_mul_slice_xor(row[d], avail_refs[d], out_buf);
                        }
                    }
                } else {
                    scalar_mul_slice(row[0], avail_refs[0], out_buf);
                    for d in 1..self.data_shards {
                        scalar_mul_slice_xor(row[d], avail_refs[d], out_buf);
                    }
                }
            }

            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
            {
                scalar_mul_slice(row[0], avail_refs[0], out_buf);
                for d in 1..self.data_shards {
                    scalar_mul_slice_xor(row[d], avail_refs[d], out_buf);
                }
            }
        }

        Ok(())
    }
}
