use crate::{consts::STABILITY_COEFF, datatype::CsMat, sparse::SparseVec};

use super::{
    eta_matrices::{self, EtaMatrices},
    lu::{self, gplu::GPLUFactorizer, LUFactorizer, LUFactors},
};

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
}
