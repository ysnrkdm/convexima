use std::borrow::BorrowMut;

use crate::{
    consts::STABILITY_COEFF,
    datatype::CsMat,
    sparse::{ScatteredVec, SparseVec},
};

use super::{
    eta_matrices::EtaMatrices,
    lu::{gplu::GPLUFactorizer, LUFactorizer, LUFactors},
};

#[derive(Debug)]
pub struct BasisSolver {
    lu_factors: LUFactors,
    lu_factors_transpose: LUFactors,
    eta_matrices: EtaMatrices,
}

impl BasisSolver {
    pub fn new(lu_factors: LUFactors, num_constraints: usize) -> BasisSolver {
        let lu_factors_transpose = lu_factors.transpose();
        BasisSolver {
            lu_factors,
            lu_factors_transpose,
            eta_matrices: EtaMatrices::new(num_constraints),
        }
    }

    fn push_eta_matrix(&mut self, col_coeffs: &SparseVec, r_leaving: usize, pivot_coeff: f64) {
        let coeffs = col_coeffs.iter().map(|(r, &coeff)| {
            let val = if r == r_leaving {
                1.0 - 1.0 / pivot_coeff
            } else {
                coeff / pivot_coeff
            };
            (r, val)
        });
        self.eta_matrices.push(r_leaving, coeffs);
    }

    fn reset(&mut self, orig_constraints_csc: &CsMat, basic_vars: &[usize]) {
        // self.scratch.clear_sparse(basic_vars.len());
        self.eta_matrices.clear_and_resize(basic_vars.len());
        // self.rhs.clear_and_resize(basic_vars.len());

        let gplu = GPLUFactorizer::new(STABILITY_COEFF);
        self.lu_factors = gplu
            .lu_factorize(basic_vars.len(), |c| {
                orig_constraints_csc
                    .outer_view(basic_vars[c])
                    .unwrap()
                    .into_raw_storage()
            })
            .unwrap(); // TODO: When is singular basis matrix possible? Report as a proper error.
        self.lu_factors_transpose = self.lu_factors.transpose();
    }

    pub fn push_eta_matrix_or_reset(
        &mut self,
        col_coeffs: &SparseVec,
        r_leaving: usize,
        pivot_coeff: f64,
        orig_constraints_csc: &CsMat,
        basic_vars: &[usize],
    ) {
        let eta_matrices_nnz = self.eta_matrices.nnz();
        if eta_matrices_nnz < self.lu_factors.nnz() {
            self.push_eta_matrix(col_coeffs, r_leaving, pivot_coeff)
        } else {
            self.reset(orig_constraints_csc, basic_vars)
        }
    }

    pub fn reset_if_eta_matrices_too_long(
        &mut self,
        orig_constraints_csc: &CsMat,
        basic_vars: &[usize],
    ) -> bool {
        if self.eta_matrices.len() > 8 {
            self.reset(orig_constraints_csc, basic_vars);
            true
        } else {
            false
        }
    }

    pub fn solve_for_vector<'a>(
        &mut self,
        rhs_vec: impl Iterator<Item = (usize, &'a f64)>,
    ) -> ScatteredVec {
        thread_local! {
            pub static SCRATCH_RHS:ScatteredVec = ScatteredVec::empty(1)
        }
        SCRATCH_RHS.with(|mut scratch_rhs| {
            let mut rhs = scratch_rhs.borrow_mut().to_owned();
            let n = self.eta_matrices.coeff_cols.rows();

            if rhs.len() != n {
                rhs.clear_and_resize(n)
            } else {
                rhs.clear()
            }
            rhs.set(rhs_vec);

            self.solve(&rhs)
        })
    }

    // Solve Bd=a for d where a = rhs
    fn solve(&mut self, rhs: &ScatteredVec) -> ScatteredVec {
        let mut d = self.lu_factors.solve(rhs);

        // apply eta matrices (Vanderbei p.139)
        for idx in 0..self.eta_matrices.len() {
            let row_leaving = self.eta_matrices.leaving_rows[idx];
            let coeff = *d.get(row_leaving);
            for (r, &val) in self.eta_matrices.coeff_cols.col_iter(idx) {
                *d.get_mut(r) -= coeff * val;
            }
        }

        d
    }

    pub fn solve_transpose_for_vector<'a>(
        &mut self,
        rhs_vec: impl Iterator<Item = (usize, &'a f64)>,
    ) -> ScatteredVec {
        thread_local! {
            pub static SCRATCH_RHS:ScatteredVec = ScatteredVec::empty(1)
        }
        SCRATCH_RHS.with(|mut scratch_rhs| {
            let mut rhs = scratch_rhs.borrow_mut().to_owned();
            let n = self.eta_matrices.coeff_cols.rows();

            if rhs.len() != n {
                rhs.clear_and_resize(n)
            } else {
                rhs.clear()
            }
            rhs.set(rhs_vec);

            self.solve_transpose(&rhs)
        })
    }

    fn solve_transpose(&self, rhs: &ScatteredVec) -> ScatteredVec {
        let mut d = rhs.clone();

        // apply eta matrices in reverse (Vanderbei p.139)
        for idx in (0..self.eta_matrices.len()).rev() {
            let mut coeff = 0.0;
            // eta col `dot` rhs_transposed
            for (i, &val) in self.eta_matrices.coeff_cols.col_iter(idx) {
                coeff += val * d.get(i);
            }
            let row_leaving = self.eta_matrices.leaving_rows[idx];
            *d.get_mut(row_leaving) -= coeff;
        }

        self.lu_factors_transpose.solve(&d)
    }

    pub fn solve_dense_lu_factors_transpose(&self, dense_rhs: &Vec<f64>) -> Vec<f64> {
        self.lu_factors_transpose.solve_dense(dense_rhs)
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_solve_base() {}
}
