pub mod gplu;

use std::ops::Deref;

use sprs::{CsMat, CsVecBase};

use crate::sparse::SparseMat;

#[derive(Clone)]
pub struct LUFactors {
    // L of PA=LU
    lower: TriangleMat,
    // U of PA=LU
    upper: TriangleMat,
    // P of PA=LU where P is the permutation matrix
    row_permutation: Option<Permutation>,
    col_permutation: Option<Permutation>,
}

impl std::fmt::Debug for LUFactors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "L:\n{:?}", self.lower)?;
        writeln!(f, "U:\n{:?}", self.upper)?;
        writeln!(
            f,
            "row_perm.orig_from_new: {:?}",
            self.row_permutation.as_ref().map(|p| &p.orig_from_new)
        )?;
        writeln!(
            f,
            "col_perm.orig_from_new: {:?}",
            self.col_permutation.as_ref().map(|p| &p.orig_from_new)
        )?;
        Ok(())
    }
}

impl LUFactors {
    pub fn transpose(&self) -> LUFactors {
        LUFactors {
            lower: self.upper.transpose(),
            upper: self.lower.transpose(),
            row_permutation: self.col_permutation.clone(),
            col_permutation: self.row_permutation.clone(),
        }
    }

    pub fn nnz(&self) -> usize {
        self.lower.nondiag.nnz() + self.upper.nondiag.nnz() + self.lower.cols()
    }
}

#[derive(Clone)]
pub(crate) struct TriangleMat {
    nondiag: SparseMat,
    /// Diag elements, None if all 1's
    diag: Option<Vec<f64>>,
}

impl TriangleMat {
    fn rows(&self) -> usize {
        self.nondiag.rows()
    }

    fn cols(&self) -> usize {
        self.nondiag.cols()
    }

    pub(crate) fn transpose(&self) -> TriangleMat {
        TriangleMat {
            nondiag: self.nondiag.transpose(),
            diag: self.diag.clone(),
        }
    }

    #[cfg(test)]
    pub(crate) fn to_csmat(&self) -> CsMat<f64> {
        let mut tri_mat = sprs::TriMat::new((self.rows(), self.cols()));
        if let Some(diag) = self.diag.as_ref() {
            for (i, &val) in diag.iter().enumerate() {
                tri_mat.add_triplet(i, i, val);
            }
        } else {
            for i in 0..self.rows() {
                tri_mat.add_triplet(i, i, 1.0);
            }
        }

        for c in 0..self.nondiag.cols() {
            for (r, &val) in self.nondiag.col_iter(c) {
                tri_mat.add_triplet(r, c, val);
            }
        }

        tri_mat.to_csc()
    }
}

fn to_dense<IStorage, DStorage>(vec: &CsVecBase<IStorage, DStorage, f64>) -> Vec<f64>
where
    IStorage: Deref<Target = [usize]>,
    DStorage: Deref<Target = [f64]>,
{
    let mut dense = vec![0.0; vec.dim()];
    vec.scatter(&mut dense);
    dense
}

pub fn pretty_print_csmat(mat: &CsMat<f64>) {
    for row in mat.to_csr().outer_iterator() {
        println!("{:?}", to_dense(&row))
    }
}

impl std::fmt::Debug for TriangleMat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "nondiag:")?;
        for row in self.nondiag.to_csmat().to_csr().outer_iterator() {
            writeln!(f, "{:?}", to_dense(&row))?
        }
        writeln!(f, "diag: {:?}", self.diag)?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Permutation {
    new_from_orig: Vec<usize>,
    orig_from_new: Vec<usize>,
}

pub trait LUFactorizer {
    fn lu_factorize<'a>(
        self,
        col_size: usize,
        get_col: impl Fn(usize) -> (&'a [usize], &'a [f64]),
    ) -> Result<LUFactors, Error>;
}

#[derive(Debug)]
pub enum Error {
    SingularMatrix,
}
