pub mod gplu;
pub mod supernodal_lu;

use std::{cell::RefCell, ops::Deref};

use sprs::{CsMat, CsVecBase};

use crate::sparse::{ScatteredVec, SparseMat};

use self::gplu::topo_sorted_reachables;

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

    // Solve LUx = b, where b is the rhs
    pub fn solve(&self, rhs: &ScatteredVec) -> ScatteredVec {
        let mut tmp = rhs.clone();
        self.solve_inplace(&mut tmp);
        tmp
    }

    // Inplace version. Takes rhs from the argument, and the answer is populated to the rhs
    pub fn solve_inplace(&self, rhs: &mut ScatteredVec) {
        thread_local! {
            static SCRATCH_RHS:RefCell<ScatteredVec> = RefCell::new(ScatteredVec::empty(1))
        }
        SCRATCH_RHS.with(|scratch_rhs| {
            let mut tmp = scratch_rhs.borrow_mut().to_owned();
            let n = rhs.len();
            if tmp.len() != n {
                dbg!(
                    "solve_inplace: clearing and resizing the scratch rhs with {}, from {}",
                    n,
                    tmp.len()
                );
                tmp.clear_and_resize(n)
            }

            // Prepare the temp vector which permutates the rows
            if let Some(row_permutation) = &self.row_permutation {
                tmp.clear();
                for &orig_i in &rhs.nonzero {
                    let new_i = row_permutation.new_from_orig[orig_i];
                    tmp.nonzero.push(new_i);
                    tmp.is_nonzero[new_i] = true;
                    tmp.values[new_i] = rhs.values[orig_i];
                }
            } else {
                std::mem::swap(&mut tmp, rhs);
            };

            // First, Ly = b, where y = Ux. b is given as tmp, and y is returned by tmp
            tri_solve_sparse_inplace(&self.lower, &mut tmp);
            // Then, Ux = y. y is given as tmp, and x is returned by tmp
            tri_solve_sparse_inplace(&self.upper, &mut tmp);
            // Ok, now we have x in tmp

            // Get result vector which permutates the columns
            if let Some(col_permutation) = &self.col_permutation {
                rhs.clear();
                for &new_i in &tmp.nonzero {
                    let orig_i = col_permutation.orig_from_new[new_i];
                    rhs.nonzero.push(orig_i);
                    rhs.is_nonzero[orig_i] = true;
                    rhs.values[orig_i] = tmp.values[new_i];
                }
            } else {
                std::mem::swap(rhs, &mut tmp);
            };

            scratch_rhs.replace(tmp);
        })
    }

    pub fn solve_dense(&self, dense_rhs: &Vec<f64>) -> Vec<f64> {
        let mut tmp = vec![0.0; dense_rhs.len()];
        if let Some(row_permutation) = &self.row_permutation {
            for orig in 0..dense_rhs.len() {
                tmp[row_permutation.new_from_orig[orig]] = dense_rhs[orig];
            }
        } else {
            tmp.copy_from_slice(dense_rhs);
        }

        tri_solve_dense_inplace(&self.lower, TriangleSide::Lower, &mut tmp);
        tri_solve_dense_inplace(&self.upper, TriangleSide::Upper, &mut tmp);

        let mut ans = vec![0.0; dense_rhs.len()];
        if let Some(col_permutation) = &self.col_permutation {
            for new in 0..dense_rhs.len() {
                ans[col_permutation.orig_from_new[new]] = tmp[new];
            }
        } else {
            ans.copy_from_slice(&tmp);
        }
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

enum TriangleSide {
    Lower,
    Upper,
}

fn tri_solve_dense_inplace(triangle_mat: &TriangleMat, triangle: TriangleSide, rhs: &mut [f64]) {
    match triangle {
        TriangleSide::Lower => {
            for col in 0..triangle_mat.cols() {
                tri_solve_process_col(triangle_mat, col, rhs);
            }
        }

        TriangleSide::Upper => {
            for col in (0..triangle_mat.cols()).rev() {
                tri_solve_process_col(triangle_mat, col, rhs);
            }
        }
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
pub struct Permutation {
    pub new_from_orig: Vec<usize>,
    pub orig_from_new: Vec<usize>,
}

impl Permutation {
    pub fn identity(n: usize) -> Permutation {
        Permutation {
            new_from_orig: (0..n).collect(),
            orig_from_new: (0..n).collect(),
        }
    }

    pub fn reorder(&mut self, ordering: Vec<usize>) {
        let size = self.new_from_orig.len();

        assert_eq!(size, ordering.len());

        for (new_n, &old_n) in ordering.iter().enumerate() {
            self.new_from_orig[old_n] = new_n;
        }

        for (orig, &new) in self.new_from_orig.iter().enumerate() {
            self.orig_from_new[new] = orig;
        }
    }
}

pub trait LUFactorizer {
    fn lu_factorize<'a>(
        self,
        col_size: usize,
        get_col: impl Fn(usize) -> (&'a [usize], &'a [f64]), // (Index, and Value in f64)
    ) -> Result<LUFactors, Error>;
}

#[derive(Debug)]
pub enum Error {
    SingularMatrix,
}

#[cfg(test)]
mod tests {
    use crate::{
        helpers::helpers::mat_from_triplets,
        solvers::revised_dual_simplex::lu::{gplu::GPLUFactorizer, pretty_print_csmat},
    };

    use super::*;

    #[test]
    fn test_solve_base() {
        let gplu = GPLUFactorizer::new(0.9);

        let test_mat = mat_from_triplets(
            3,
            3,
            &[
                (0, 0, 3.0),
                (0, 1, -2.0),
                (0, 2, 4.0),
                (1, 0, 2.0),
                (1, 1, 1.0),
                (1, 2, -3.0),
                (2, 0, 4.0),
                (2, 1, -3.0),
                (2, 2, 2.0),
            ],
        );

        let factorized_lu = gplu.lu_factorize(test_mat.rows(), |c| {
            test_mat.outer_view(c).unwrap().into_raw_storage()
        });

        pretty_print_csmat(&test_mat);
        assert!(factorized_lu.is_ok());
        let mat = factorized_lu.unwrap();
        println!("{:?}", mat);
        let multiplied = &mat.lower.to_csmat() * &mat.upper.to_csmat();
        pretty_print_csmat(&multiplied);

        let mut rhs = ScatteredVec::empty(3);
        // rhs.set(vec![(0, &(1.0)), (1, &(2.0)), (2, &(-3.0))]);
        rhs.set(vec![(0, &(11.0)), (1, &(-5.0)), (2, &(4.0))]);
        let d = mat.solve(&rhs);
        assert_eq!(vec![1.0, 2.0, 3.0], d.values);
        // println!("Answer: {:?}", d);

        rhs.set(vec![(0, &(1.0)), (1, &(2.0)), (2, &(-3.0))]);
        let d2 = mat.solve(&rhs);
        assert_eq!(vec![1.0, 3.0, 1.0], d2.values);
        // println!("Answer: {:?}", d2);
    }
}
