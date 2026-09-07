//! Drop-in `ReedSolomonCache` replacement for `solana-ledger::shredder::ReedSolomonCache`.
//!
//! Provides thread-safe, LRU-cached instances of `SimdReedSolomon` keyed by
//! `(data_shards, parity_shards)`.

use std::sync::Arc;
use lru::LruCache;
use parking_lot::Mutex;

use crate::engine::{Error, SimdReedSolomon};

const DEFAULT_CACHE_CAPACITY: usize = 64;

/// Thread-safe LRU cache of hardware-accelerated Reed-Solomon codec instances.
/// Drop-in compatible with `solana-ledger::shredder::ReedSolomonCache`.
#[derive(Debug)]
pub struct ReedSolomonCache {
    cache: Mutex<LruCache<(usize, usize), Arc<SimdReedSolomon>>>,
}

impl Default for ReedSolomonCache {
    fn default() -> Self {
        Self::new(DEFAULT_CACHE_CAPACITY)
    }
}

impl ReedSolomonCache {
    /// Creates a new cache with the specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Mutex::new(LruCache::new(capacity)),
        }
    }

    /// Retrieves an existing codec from the cache or instantiates a new one.
    pub fn get(
        &self,
        data_shards: usize,
        parity_shards: usize,
    ) -> Result<Arc<SimdReedSolomon>, Error> {
        let key = (data_shards, parity_shards);
        let mut cache = self.cache.lock();
        if let Some(entry) = cache.get(&key) {
            return Ok(Arc::clone(entry));
        }

        let rs = Arc::new(SimdReedSolomon::new(data_shards, parity_shards)?);
        cache.put(key, Arc::clone(&rs));
        Ok(rs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reed_solomon_cache() {
        let cache = ReedSolomonCache::default();
        let rs1 = cache.get(32, 32).expect("Get (32, 32)");
        let rs2 = cache.get(32, 32).expect("Get (32, 32) cached");
        assert!(Arc::ptr_eq(&rs1, &rs2));

        let rs3 = cache.get(16, 16).expect("Get (16, 16)");
        assert!(!Arc::ptr_eq(&rs1, &rs3));
        assert_eq!(rs3.data_shard_count(), 16);
        assert_eq!(rs3.parity_shard_count(), 16);
    }
}
