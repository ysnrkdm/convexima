use std::borrow::Borrow;

use crate::{
    consts::STABILITY_COEFF,
    datatype::CsMat,
    sparse::{ScatteredVec, SparseVec},
};

use super::{
    eta_matrices::EtaMatrices,
    lu::{gplu::GPLUFactorizer, LUFactorizer, LUFactors},
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

    pub fn solve_for_vector<'a>(
        &mut self,
        rhs_vec: impl Iterator<Item = (usize, &'a f64)>,
    ) -> ScatteredVec {
        let mut rhs = ScatteredVec::empty(self.eta_matrices.coeff_cols.rows());
        rhs.set(rhs_vec);

        self.solve(&rhs)
    }

    // Solve Bd=a for d where a = rhs
    pub fn solve(&mut self, rhs: &ScatteredVec) -> ScatteredVec {
        let mut d = self.lu_factors.solve(rhs);

        for idx in 0..self.eta_matrices.len() {
            let row_leaving = self.eta_matrices.leaving_rows[idx];
            let coeff = *d.get(row_leaving);
            for (r, &val) in self.eta_matrices.coeff_cols.col_iter(idx) {
                *d.get_mut(r) -= coeff * val;
            }
        }

        d
    }

    pub fn solve_transpose<'a>(
        &mut self,
        rhs: impl Iterator<Item = (usize, &'a f64)>,
    ) -> &ScatteredVec {
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        helpers::helpers::mat_from_triplets, solvers::revised_dual_simplex::lu::pretty_print_csmat,
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
