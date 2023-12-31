pub mod gplu;

use std::ops::Deref;

use sprs::{CsMat, CsVecBase};

use crate::sparse::{ScatteredVec, SparseMat};

use self::gplu::topo_sorted_reachables;

#[derive(Clone)]
pub struct LUFactors {
    // L of PA=LU
    pub lower: TriangleMat,
    // U of PA=LU
    pub upper: TriangleMat,
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

    // Solve LUx = b, where b is the rhs
    pub fn solve(&self, rhs: &ScatteredVec) -> ScatteredVec {
        // Prepare the temp vector which permutates the rows
        let mut tmp;
        if let Some(row_permutation) = &self.row_permutation {
            tmp = ScatteredVec::empty(rhs.len());
            for &orig_i in &rhs.nonzero {
                let new_i = row_permutation.new_from_orig[orig_i];
                tmp.nonzero.push(new_i);
                tmp.is_nonzero[new_i] = true;
                tmp.values[new_i] = rhs.values[orig_i];
            }
        } else {
            tmp = rhs.clone();
        };

        // First, Ly = b, where y = Ux. b is given as tmp, and y is returned by tmp
        tri_solve_sparse_inplace(&self.lower, &mut tmp);
        // Then, Ux = y. y is given as tmp, and x is returned by tmp
        tri_solve_sparse_inplace(&self.upper, &mut tmp);
        // Ok, now we have x in tmp

        // Get result vector which permutates the columns
        let mut ans;
        if let Some(col_permutation) = &self.col_permutation {
            ans = ScatteredVec::empty(tmp.len());
            for &new_i in &tmp.nonzero {
                let orig_i = col_permutation.orig_from_new[new_i];
                ans.nonzero.push(orig_i);
                ans.is_nonzero[orig_i] = true;
                ans.values[orig_i] = tmp.values[orig_i];
            }
        } else {
            ans = tmp.clone();
        };
        ans
    }
}

fn tri_solve_sparse_inplace(triangle_mat: &TriangleMat, rhs: &mut ScatteredVec) {
    let reachables = topo_sorted_reachables(
        rhs.len(),
        &rhs.nonzero,
        |col| triangle_mat.nondiag.col_rows(col),
        |_| true,
        |i| i,
    );

    for &i in &reachables.visited {
        if !rhs.is_nonzero[i] {
            rhs.is_nonzero[i] = true;
            rhs.nonzero.push(i);
        }
    }

    for &col in reachables.visited.iter().rev() {
        tri_solve_process_col(triangle_mat, col, &mut rhs.values);
    }
}

fn tri_solve_process_col(triangle_mat: &TriangleMat, col: usize, rhs: &mut [f64]) {
    // all other variables in this row (multiplied by their coeffs)
    // are already subtracted from rhs[col].
    let x_val = if let Some(diag) = triangle_mat.diag.as_ref() {
        rhs[col] / diag[col]
    } else {
        rhs[col]
    };

    rhs[col] = x_val;
    for (r, &coeff) in triangle_mat.nondiag.col_iter(col) {
        rhs[r] -= x_val * coeff;
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
