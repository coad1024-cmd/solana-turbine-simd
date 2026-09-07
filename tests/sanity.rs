use reed_solomon_erasure::galois_8::ReedSolomon;

#[test]
fn test_reed_solomon_erasure_basic() {
    let r = ReedSolomon::new(32, 32).unwrap();
    let data: Vec<Vec<u8>> = (0..32).map(|i| vec![i as u8; 1228]).collect();
    let mut parity: Vec<Vec<u8>> = vec![vec![0u8; 1228]; 32];

    let data_slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
    let mut parity_slices: Vec<&mut [u8]> = parity.iter_mut().map(|v| v.as_mut_slice()).collect();

    r.encode_sep(&data_slices, &mut parity_slices).unwrap();

    println!("Parity shred 0 first 8 bytes: {:?}", &parity[0][..8]);
    println!("Parity shred 31 first 8 bytes: {:?}", &parity[31][..8]);
}
