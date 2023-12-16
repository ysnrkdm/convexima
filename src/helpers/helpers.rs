use sprs::{CsVecBase, CsVecView};
use std::ops::Deref;

pub(crate) fn to_dense<IStorage, DStorage>(vec: &CsVecBase<IStorage, DStorage, f64>) -> Vec<f64>
where
    IStorage: Deref<Target = [usize]>,
    DStorage: Deref<Target = [f64]>,
{
    let mut dense = vec![0.0; vec.dim()];
    vec.scatter(&mut dense);
    dense
}

#[cfg(test)]
use sprs::{CsMat, CsVec};

#[cfg(test)]
pub(crate) fn to_sparse(slice: &[f64]) -> CsVec<f64> {
    let mut res = CsVec::empty(slice.len());
    for (i, &val) in slice.iter().enumerate() {
        if val != 0.0 {
            res.append(i, val);
        }
    }
    res
}

#[cfg(test)]
pub(crate) fn assert_matrix_eq(mat: &CsMat<f64>, reference: &[Vec<f64>]) {
    let mat = mat.to_csr();
    assert_eq!(mat.rows(), reference.len());
    for (r, row) in mat.outer_iterator().enumerate() {
        assert_eq!(to_dense(&row), reference[r], "matrices differ in row {}", r);
    }
}
