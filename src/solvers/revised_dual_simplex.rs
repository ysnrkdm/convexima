mod eta_matrices;
mod lu;

use log::debug;

use crate::{
    datatype::{CsMat, CsVec},
    problem::{ComparisonOp, Problem},
    solution::Solution,
    solver::{Error, Solver, SolverTryNew},
    sparse::{ScatteredVec, SparseVec},
};

#[derive(Clone, Debug)]
enum VarState {
    Basic(usize),
    NonBasic(usize),
}

#[derive(Clone, Debug)]
struct NonBasicVarState {
    at_min: bool,
    at_max: bool,
}

pub struct SimpleSolver {
    orig_problem: Problem,

    pub num_vars: usize,

    is_primal_feasible: bool,
    is_dual_feasible: bool,

    pub cur_obj_val: f64,

    // Original inputs
    orig_obj_coeffs: Vec<f64>,
    orig_var_mins: Vec<f64>,
    orig_var_maxs: Vec<f64>,
    orig_constraints: CsMat, // excluding rhs
    orig_rhs: Vec<f64>,

    // Variables
    var_states: Vec<VarState>,

    // Non-basic variables
    nb_vars: Vec<usize>,
    nb_var_obj_coeffs: Vec<f64>,
    nb_var_vals: Vec<f64>,
    nb_var_states: Vec<NonBasicVarState>,
    nb_var_is_fixed: Vec<bool>,

    // Basic variables
    basic_vars: Vec<usize>,
    basic_var_vals: Vec<f64>,
    basic_var_mins: Vec<f64>,
    basic_var_maxs: Vec<f64>,

    // Recomputed on each pivot
    col_coeffs: SparseVec,
    row_coeffs: ScatteredVec,
}

impl SolverTryNew<SimpleSolver> for SimpleSolver {
    fn try_new(problem: &Problem) -> Result<SimpleSolver, Error> {
        let obj_coeffs = &problem.obj_coeffs;

        let num_vars = obj_coeffs.len();

        assert_eq!(num_vars, problem.var_mins.len());
        assert_eq!(num_vars, problem.var_maxs.len());
        let mut orig_var_mins = problem.var_mins.to_vec();
        let mut orig_var_maxs = problem.var_maxs.to_vec();

        let mut var_states = vec![];

        let mut nb_vars = vec![];
        let mut nb_var_vals = vec![];
        let mut nb_var_states = vec![];

        let mut obj_val = 0.0;

        let mut is_dual_feasible = true;

        for v_idx in 0..num_vars {
            let min = orig_var_mins[v_idx];
            let max = orig_var_maxs[v_idx];
            if min > max {
                return Err(Error::Infeasible);
            }

            var_states.push(VarState::NonBasic(nb_vars.len()));
            nb_vars.push(v_idx);

            let (base_sol_val, is_dual_feasible_by) = match () {
                _ if min == max => (min, true),
                _ if min.is_infinite() && max.is_infinite() => (0.0, obj_coeffs[v_idx] != 0.0),
                // Below min or max is (either or both) finite
                _ if obj_coeffs[v_idx] > 0.0 && min.is_finite() => (min, true),
                _ if obj_coeffs[v_idx] > 0.0 => (max, false),
                _ if obj_coeffs[v_idx] < 0.0 && max.is_finite() => (max, true),
                _ if obj_coeffs[v_idx] < 0.0 => (min, false),
                _ if min.is_infinite() => (min, true),
                _ => (max, true),
            };
            is_dual_feasible &= is_dual_feasible_by;

            nb_var_vals.push(base_sol_val);
            obj_val += base_sol_val * obj_coeffs[v_idx];

            nb_var_states.push(NonBasicVarState {
                at_min: base_sol_val == min,
                at_max: base_sol_val == max,
            })
        }

        let mut constraint_coeffs = vec![];
        let mut orig_rhs = vec![];

        let mut basic_vars = vec![];
        let mut basic_var_vals = vec![];
        let mut basic_var_mins = vec![];
        let mut basic_var_maxs = vec![];

        for (coeffs, cmp_op, rhs) in problem.constraints.iter() {
            let rhs = *rhs;

            if coeffs.indices().is_empty() {
                let is_tautological = cmp_op.evaluate(0.0, rhs);
                if is_tautological {
                    continue;
                } else {
                    return Err(Error::Infeasible);
                }
            }

            constraint_coeffs.push(coeffs.clone());
            orig_rhs.push(rhs);

            let (slack_var_min, slack_var_max) = match cmp_op {
                ComparisonOp::Eq => (0.0, f64::INFINITY),
                ComparisonOp::Le => (f64::INFINITY, 0.0),
                ComparisonOp::Ge => (0.0, 0.0),
            };

            orig_var_mins.push(slack_var_min);
            orig_var_maxs.push(slack_var_max);

            basic_var_mins.push(slack_var_min);
            basic_var_maxs.push(slack_var_max);

            let cur_slack_var = var_states.len();
            var_states.push(VarState::Basic(basic_vars.len()));
            basic_vars.push(cur_slack_var);

            let lhs_val: f64 = coeffs
                .iter()
                .map(|(var, &coeff)| coeff * nb_var_vals[var])
                .sum();
            basic_var_vals.push(rhs - lhs_val);
        }

        let is_primal_feasible = basic_var_vals
            .iter()
            .zip(&basic_var_mins)
            .zip(&basic_var_maxs)
            .all(|((&val, &min), &max)| min <= val && val <= max);

        let num_constraints = constraint_coeffs.len();
        let num_total_vars = num_vars + num_constraints;

        let orig_obj_coeffs = {
            let mut r = obj_coeffs.to_vec();
            r.resize(num_total_vars, 0.0);
            r
        };

        let orig_constraints = {
            let mut r = CsMat::empty(sprs::CompressedStorage::CSR, num_total_vars);
            for (cur_slack_var, coeffs) in constraint_coeffs.into_iter().enumerate() {
                let mut coeffs = into_resized(coeffs, num_total_vars);
                coeffs.append(num_vars + cur_slack_var, 1.0);
                r = r.append_outer_csvec(coeffs.view());
            }
            r
        };

        // let orig_constraints_csc = orig_constraints.to_csc();

        let need_artificial_obj = !is_primal_feasible && !is_dual_feasible;

        let nb_var_obj_coeffs = {
            let mut r = vec![];
            for (&var, state) in nb_vars.iter().zip(&nb_var_states) {
                // let col = orig_constraints_csc.outer_view(var).unwrap();

                if need_artificial_obj {
                    let coeff = if state.at_min && !state.at_max {
                        1.0
                    } else if !state.at_min && state.at_max {
                        -1.0
                    } else {
                        0.0
                    };
                    r.push(coeff);
                } else {
                    r.push(orig_obj_coeffs[var]);
                }
            }
            r
        };

        let nb_var_is_fixed = vec![false; nb_vars.len()];

        let cur_obj_val = if need_artificial_obj { 0.0 } else { obj_val };

        // let mut scratch = ScratchSpace::with_capacity(num_constraints);

        // let orig_constraints_csc = orig_constraints.to_csc();
        // let lu_factors = lu_factorize(
        //     basic_vars.len(),
        //     |c| {
        //         orig_constraints_csc
        //             .outer_view(basic_vars[c])
        //             .unwrap()
        //             .into_raw_storage()
        //     },
        //     0.1,
        //     &mut scratch,
        // )
        // .unwrap();

        // let lu_factors_transp = lu_factors.transpose();

        let res = Self {
            orig_problem: problem.clone(),

            num_vars,

            is_primal_feasible,
            is_dual_feasible,

            cur_obj_val,

            // Original inputs
            orig_obj_coeffs,
            orig_var_mins,
            orig_var_maxs,
            orig_constraints,
            orig_rhs,

            // Variables
            var_states,

            // Non-basic variables
            nb_vars,
            nb_var_obj_coeffs,
            nb_var_vals,
            nb_var_states,
            nb_var_is_fixed,

            // Basic variables
            basic_vars,
            basic_var_vals,
            basic_var_mins,
            basic_var_maxs,

            col_coeffs: SparseVec::new(),
            row_coeffs: ScatteredVec::empty(num_total_vars - num_constraints),
            // inv_basis_matrix: InvertedBasisMatrix {
            //     lu_factors,
            //     lu_factors_transp,
            //     scratch,
            //     eta_matrices: EtaMatrices::new(num_constraints),
            //     rhs: ScatteredVec::empty(num_constraints),
            // },
        };

        debug!(
            "initialized solver: vars: {}, constraints: {}, primal feasible: {}, dual feasible: {}, nnz: {}",
            res.num_vars,
            res.orig_constraints.rows(),
            res.is_primal_feasible,
            res.is_dual_feasible,
            res.orig_constraints.nnz(),
        );

        Ok(res)
    }
}

impl Solver for SimpleSolver {
    fn solve(&mut self) -> Result<Solution, Error> {
        if !self.is_primal_feasible {
            self.restore_primal_feasibility()?;
        }

        if !self.is_dual_feasible {
            self.recalc_obj_coeffs();
            self.optimize()?;
        }

        Ok(Solution::new(
            self.orig_problem.direction,
            self.orig_obj_coeffs.len(),
            self.cur_obj_val,
            self.get_optimal_variables(),
        ))
    }
}

#[derive(Debug)]
struct PivotInfo {
    col: usize,
    entering_new_val: f64,
    entering_diff: f64,

    /// Contains info about the intersection between pivot row and column.
    /// If it is None, objective can be decreased without changing the basis
    /// (simply by changing the value of non-basic variable chosen as entering)
    elem: Option<PivotElem>,
}

#[derive(Debug)]
struct PivotElem {
    row: usize,
    coeff: f64,
    leaving_new_val: f64,
}

impl SimpleSolver {
    fn restore_primal_feasibility(&mut self) -> Result<(), Error> {
        for iter in 0.. {
            if let Some((row, leaving_new_value)) = self.choose_pivot_row_dual() {
                self.calc_row_coeffs(row);
                let pivot_info = self.choose_entering_col_dual(row, leaving_new_value)?;
                self.calc_col_coeffs(pivot_info.col);
                self.pivot(&pivot_info);
            }
        }

        self.is_primal_feasible = true;
        Ok(())
    }

    fn recalc_obj_coeffs(&mut self) {}

    fn optimize(&mut self) -> Result<(), Error> {
        todo!();
    }

    fn get_optimal_variables(&self) -> Vec<f64> {
        todo!();
    }

    fn choose_pivot_row_dual(&self) -> Option<(usize, f64)> {
        todo!();
    }

    fn calc_row_coeffs(&mut self, r_constr: usize) {
        todo!();
    }

    fn calc_col_coeffs(&mut self, c_var: usize) {
        todo!();
    }

    fn choose_entering_col_dual(
        &self,
        row: usize,
        leaving_new_val: f64,
    ) -> Result<PivotInfo, Error> {
        todo!();
    }

    fn pivot(&mut self, pivot_info: &PivotInfo) {
        self.cur_obj_val += self.nb_var_obj_coeffs[pivot_info.col] * pivot_info.entering_diff;

        if pivot_info.elem.is_none() {
            self._pivot_nb_and_nb(pivot_info)
        } else {
            self._pivot_nb_and_b(pivot_info)
        }
    }

    fn _pivot_nb_and_nb(&mut self, pivot_info: &PivotInfo) {
        let entering_var = self.nb_vars[pivot_info.col];

        // "entering" var is still non-basic, it just changes value from one limit
        // to the other.
        self.nb_var_vals[pivot_info.col] = pivot_info.entering_new_val;
        for (r, coeff) in self.col_coeffs.iter() {
            self.basic_var_vals[r] -= pivot_info.entering_diff * coeff;
        }
        let var_state = &mut self.nb_var_states[pivot_info.col];
        var_state.at_min = pivot_info.entering_new_val == self.orig_var_mins[entering_var];
        var_state.at_max = pivot_info.entering_new_val == self.orig_var_maxs[entering_var];
    }

    fn _pivot_nb_and_b(&mut self, pivot_info: &PivotInfo) {
        let pivot_elem = pivot_info.elem.as_ref().unwrap();
        let pivot_coeff = pivot_elem.coeff;

        // Update basic variables related data (vals, min and max)
        let entering_var = self.nb_vars[pivot_info.col];

        for (r, coeff) in self.col_coeffs.iter() {
            if r == pivot_elem.row {
                self.basic_var_vals[r] = pivot_info.entering_new_val;
            } else {
                self.basic_var_vals[r] -= pivot_info.entering_diff * coeff;
            }
        }

        self.basic_var_mins[pivot_elem.row] = self.orig_var_mins[entering_var];
        self.basic_var_maxs[pivot_elem.row] = self.orig_var_maxs[entering_var];

        // Update non-basic variables related data (vals, min and max)

        let leaving_var = self.basic_vars[pivot_elem.row];

        self.nb_var_vals[pivot_info.col] = pivot_elem.leaving_new_val;
        let leaving_var_state = &mut self.nb_var_states[pivot_info.col];
        leaving_var_state.at_min = pivot_elem.leaving_new_val == self.orig_var_mins[leaving_var];
        leaving_var_state.at_max = pivot_elem.leaving_new_val == self.orig_var_maxs[leaving_var];

        let pivot_obj = self.nb_var_obj_coeffs[pivot_info.col] / pivot_coeff;
        for (c, &coeff) in self.row_coeffs.iter() {
            if c == pivot_info.col {
                self.nb_var_obj_coeffs[c] = -pivot_obj;
            } else {
                self.nb_var_obj_coeffs[c] -= pivot_obj * coeff;
            }
        }

        // Update Basic/Non-baisc Variables
        self.basic_vars[pivot_elem.row] = entering_var;
        self.var_states[entering_var] = VarState::Basic(pivot_elem.row);
        self.nb_vars[pivot_info.col] = leaving_var;
        self.var_states[leaving_var] = VarState::NonBasic(pivot_info.col);
    }
}

fn into_resized(vec: CsVec, len: usize) -> CsVec {
    let (mut indices, mut data) = vec.into_raw_storage();

    while let Some(&i) = indices.last() {
        if i < len {
            // TODO: binary search
            break;
        }

        indices.pop();
        data.pop();
    }

    CsVec::new(len, indices, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {}
}
