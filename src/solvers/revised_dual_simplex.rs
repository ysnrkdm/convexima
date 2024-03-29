mod basis_solver;
mod eta_matrices;
mod lu;
mod pivot;

use std::io::{stdout, Write};

use log::debug;
use stopwatch::Stopwatch;

use crate::{
    consts::{EPS, STABILITY_COEFF},
    datatype::{CsMat, CsVec},
    problem::{ComparisonOp, Problem},
    solution::Solution,
    solver::{Error, Solver, SolverTryNew},
    solvers::revised_dual_simplex::lu::{gplu::GPLUFactorizer, LUFactorizer},
    sparse::{ScatteredVec, SparseVec},
};

use self::{
    basis_solver::BasisSolver,
    pivot::{harris_devex::HarrisDevex, PivotChooser},
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

#[derive(Debug)]
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
    orig_constraints_csc: CsMat,
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
    inv_basis_row_coeffs: SparseVec,

    //
    basis_solver: BasisSolver,
    pivot_chooser: HarrisDevex,
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
                _ if min.is_finite() => (min, true),
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
                ComparisonOp::Le => (0.0, f64::INFINITY),
                ComparisonOp::Ge => (f64::NEG_INFINITY, 0.0),
                ComparisonOp::Eq => (0.0, 0.0),
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
            // dbg!(basic_var_vals.len(), coeffs, cmp_op, rhs, lhs_val);
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

        let gplu = GPLUFactorizer::new(STABILITY_COEFF);

        let orig_constraints_csc = orig_constraints.to_csc();
        let lu_factors = gplu
            .lu_factorize(basic_vars.len(), |c| {
                orig_constraints_csc
                    .outer_view(basic_vars[c])
                    .unwrap()
                    .into_raw_storage()
            })
            .unwrap();

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
            orig_constraints_csc,
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
            inv_basis_row_coeffs: SparseVec::new(),
            basis_solver: BasisSolver::new(lu_factors, num_constraints),
            pivot_chooser: HarrisDevex {},
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
            println!("Primal infeasibility to be solved...");
            self.restore_primal_feasibility()?;
        }

        if !self.is_dual_feasible {
            println!("Preparing obj coeffs...");
            self.recalc_obj_coeffs();
            println!("Starting optimization...");
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
        let mut sw = Stopwatch::new();

        sw.restart();
        for iter in 0.. {
            if iter % 100 == 0 {
                if iter % 10000 == 0 {
                    let (num_vars, infeasibility) = self.calc_primal_infeasibility();
                    dbg!(
                        "restore feasibility iter {}: {}: {}, infeas. vars: {} ({})",
                        iter,
                        if self.is_dual_feasible {
                            "obj."
                        } else {
                            "artificial obj."
                        },
                        self.cur_obj_val,
                        num_vars,
                        infeasibility,
                    );
                }

                let elapsed = sw.elapsed().as_secs() + 1;
                print!(
                    "{}-th iteration, Elapsed {} secs, {} iters per sec\r",
                    iter,
                    elapsed,
                    iter / elapsed
                );
                let _ = stdout().flush();
            }

            if iter >= 20000 {
                return Err(Error::Infeasible);
            }

            if let Some(row) = self.pivot_chooser.choose_pivot_row_dual(self) {
                // dbg!(row, leaving_new_value);
                self.calc_row_coeffs(row);
                // dbg!(&self.row_coeffs);
                let pivot_info = {
                    if let Some(col) = self.pivot_chooser.choose_pivot_col_dual(self, row) {
                        let pivot_coeff = self.row_coeffs.get(col);
                        let leaving_new_val = {
                            let val = self.basic_var_vals[row];
                            let min = self.basic_var_mins[row];
                            let max = self.basic_var_maxs[row];

                            if val < min {
                                min
                            } else if val > max {
                                max
                            } else {
                                unreachable!();
                            }
                        };
                        let entering_diff =
                            (self.basic_var_vals[row] - leaving_new_val) / pivot_coeff;
                        let entering_new_val = self.nb_var_vals[col] + entering_diff;

                        PivotInfo {
                            col,
                            entering_new_val,
                            entering_diff,
                            elem: Some(PivotElem {
                                row,
                                coeff: *pivot_coeff,
                                leaving_new_val,
                            }),
                        }
                    } else {
                        return Err(Error::Infeasible);
                    }
                };
                // debug!("picked pivot_info: {:?}", pivot_info);
                // dbg!(&pivot_info);
                self.calc_col_coeffs(pivot_info.col);
                // debug!("col coeffs updated to {:?}", self.col_coeffs);
                self.pivot(&pivot_info);
            } else {
                dbg!(
                    "Restored feasibility at {}-th iteration, {}: {}",
                    iter,
                    if self.is_dual_feasible {
                        "obj."
                    } else {
                        "artificial obj."
                    },
                    self.cur_obj_val,
                );
                break;
            }
        }

        self.is_primal_feasible = true;
        Ok(())
    }

    /// Number of infeasible basic vars and sum of their infeasibilities.
    fn calc_primal_infeasibility(&self) -> (usize, f64) {
        let mut num_vars = 0;
        let mut infeasibility = 0.0;
        for ((&val, &min), &max) in self
            .basic_var_vals
            .iter()
            .zip(&self.basic_var_mins)
            .zip(&self.basic_var_maxs)
        {
            if val < min - EPS {
                num_vars += 1;
                infeasibility += min - val;
            } else if val > max + EPS {
                num_vars += 1;
                infeasibility += val - max;
            }
        }
        (num_vars, infeasibility)
    }

    fn recalc_obj_coeffs(&mut self) {
        self.basis_solver
            .reset_if_eta_matrices_too_long(&self.orig_constraints_csc, &self.basic_vars);

        let multipliers = {
            let mut rhs = vec![0.0; self.orig_constraints.rows()];
            for (c, &var) in self.basic_vars.iter().enumerate() {
                rhs[c] = self.orig_obj_coeffs[var];
            }
            self.basis_solver.solve_dense_lu_factors_transpose(&rhs)
        };

        self.nb_var_obj_coeffs.clear();
        for &var in &self.nb_vars {
            let col = self.orig_constraints_csc.outer_view(var).unwrap();
            let dot_prod: f64 = col.iter().map(|(r, val)| val * multipliers[r]).sum();
            self.nb_var_obj_coeffs
                .push(self.orig_obj_coeffs[var] - dot_prod);
        }

        self.cur_obj_val = 0.0;
        for (r, &var) in self.basic_vars.iter().enumerate() {
            self.cur_obj_val += self.orig_obj_coeffs[var] * self.basic_var_vals[r];
        }
        for (c, &var) in self.nb_vars.iter().enumerate() {
            self.cur_obj_val += self.orig_obj_coeffs[var] * self.nb_var_vals[c];
        }
    }

    fn optimize(&mut self) -> Result<(), Error> {
        for iter in 0.. {
            if iter % 10000 == 0 {
                let (num_vars, infeasibility) = self.calc_dual_infeasibility();
                dbg!(
                    "optimize iter {}: obj.: {}, non-optimal coeffs: {} ({})",
                    iter,
                    self.cur_obj_val,
                    num_vars,
                    infeasibility,
                );
            }

            if let Some(pivot_info) = self.choose_pivot()? {
                // dbg!(&pivot_info);
                self.pivot(&pivot_info);
            } else {
                debug!(
                    "Found optimum in {} iterations, obj.: {}",
                    iter + 1,
                    self.cur_obj_val,
                );
                break;
            }
        }

        self.is_dual_feasible = true;
        Ok(())
    }

    /// Number of infeasible obj. coeffs and sum of their infeasibilities.
    /// For debug print purpose
    fn calc_dual_infeasibility(&self) -> (usize, f64) {
        let mut num_vars = 0;
        let mut infeasibility = 0.0;
        for (&obj_coeff, var_state) in self.nb_var_obj_coeffs.iter().zip(&self.nb_var_states) {
            if !(var_state.at_min && obj_coeff > -EPS) && !(var_state.at_max && obj_coeff < EPS) {
                num_vars += 1;
                infeasibility += obj_coeff.abs();
            }
        }
        (num_vars, infeasibility)
    }

    fn get_optimal_variables(&self) -> Vec<f64> {
        (0..self.num_vars)
            .map(|var| self.get_optimal_variable(var))
            .collect::<Vec<f64>>()
    }

    fn get_optimal_variable(&self, var: usize) -> f64 {
        match self.var_states[var] {
            VarState::Basic(idx) => self.basic_var_vals[idx],
            VarState::NonBasic(idx) => self.nb_var_vals[idx],
        }
    }

    fn choose_pivot(&mut self) -> Result<Option<PivotInfo>, Error> {
        let maybe_entering_col = self.pivot_chooser.choose_pivot_col(self);
        if maybe_entering_col.is_none() {
            return Ok(None);
        }
        let entering_col = maybe_entering_col.unwrap();

        let entering_col_val = self.nb_var_vals[entering_col];
        let entering_diff_positive = self.nb_var_obj_coeffs[entering_col] < 0.0;
        let entering_other_val = if entering_diff_positive {
            self.orig_var_maxs[self.nb_vars[entering_col]]
        } else {
            self.orig_var_mins[self.nb_vars[entering_col]]
        };

        self.calc_col_coeffs(entering_col);

        let leaving_row = self.pivot_chooser.choose_pivot_row(&self, entering_col);

        if let Some((row, pivot_coeff)) = leaving_row {
            let leaving_new_val = if (entering_diff_positive && pivot_coeff < 0.0)
                || (!entering_diff_positive && pivot_coeff > 0.0)
            {
                self.basic_var_maxs[row]
            } else {
                self.basic_var_mins[row]
            };

            self.calc_row_coeffs(row);

            let entering_diff = (self.basic_var_vals[row] - leaving_new_val) / pivot_coeff;
            let entering_new_val = entering_col_val + entering_diff;

            Ok(Some(PivotInfo {
                col: entering_col,
                entering_new_val,
                entering_diff,
                elem: Some(PivotElem {
                    row,
                    coeff: pivot_coeff,
                    leaving_new_val,
                }),
            }))
        } else {
            if entering_other_val.is_infinite() {
                return Err(Error::Unbounded);
            }

            Ok(Some(PivotInfo {
                col: entering_col,
                entering_new_val: entering_other_val,
                entering_diff: entering_other_val - entering_col_val,
                elem: None,
            }))
        }
    }

    fn calc_row_coeffs(&mut self, r_constr: usize) {
        self.basis_solver
            .solve_transpose_for_vector(std::iter::once((r_constr, &1.0)))
            .to_sparse_vec(&mut self.inv_basis_row_coeffs);

        // dbg!(&self.inv_basis_row_coeffs);

        self.row_coeffs.clear_and_resize(self.nb_vars.len());

        for (r, &coeff) in self.inv_basis_row_coeffs.iter() {
            for (v, &val) in self.orig_constraints.outer_view(r).unwrap().iter() {
                if let VarState::NonBasic(idx) = self.var_states[v] {
                    *self.row_coeffs.get_mut(idx) += val * coeff;
                }
            }
        }
    }

    fn calc_col_coeffs(&mut self, c_var: usize) {
        let var = self.nb_vars[c_var];
        let orig_col = self.orig_constraints_csc.outer_view(var).unwrap();
        self.basis_solver
            .solve_for_vector(orig_col.iter())
            .to_sparse_vec(&mut self.col_coeffs);
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

        //
        self.basis_solver.push_eta_matrix_or_reset(
            &self.col_coeffs,
            pivot_elem.row,
            pivot_coeff,
            &self.orig_constraints_csc,
            &self.basic_vars,
        );
    }
}

fn into_resized(vec: CsVec, len: usize) -> CsVec {
    let (mut indices, mut data) = vec.into_raw_storage();

    // TODO: binary search
    while let Some(&i) = indices.last() {
        if i < len {
            break;
        }

        indices.pop();
        data.pop();
    }

    CsVec::new(len, indices, data)
}

#[cfg(test)]
mod tests {
    use std::f64::{INFINITY, NEG_INFINITY};

    use super::*;
    use crate::problem::OptimizationDirection::*;

    #[test]
    fn initialize_test() {
        let mut problem = Problem::new(Minimize);
        let x1 = problem.add_var(2.0, (NEG_INFINITY, 0.0));
        let x2 = problem.add_var(1.0, (5.0, INFINITY));
        problem.add_constraint([(x1, 1.0), (x2, 1.0)], ComparisonOp::Le, 6.0);
        problem.add_constraint([(x1, 1.0), (x2, 2.0)], ComparisonOp::Le, 6.0);
        problem.add_constraint([(x1, 1.0), (x2, 1.0)], ComparisonOp::Ge, 2.0);
        problem.add_constraint([(x1, 0.0), (x2, 1.0)], ComparisonOp::Eq, 3.0);
        let res_solver = SimpleSolver::try_new(&problem);
        assert!(res_solver.is_ok());
        let solver = res_solver.unwrap();
        assert_eq!(solver.num_vars, 2);
        assert!(!solver.is_primal_feasible);
        assert!(!solver.is_dual_feasible);
        assert_eq!(&solver.orig_obj_coeffs, &[2.0, 1.0, 0.0, 0.0, 0.0, 0.0]);

        assert_eq!(
            &solver.orig_var_mins,
            &[NEG_INFINITY, 5.0, 0.0, 0.0, NEG_INFINITY, 0.0],
        );
        assert_eq!(
            &solver.orig_var_maxs,
            &[0.0, INFINITY, INFINITY, INFINITY, 0.0, 0.0],
        );

        // TODO: add more assertions to inspect the object
    }

    #[test]
    fn restore_primal_feasibility_test() {
        let mut problem = Problem::new(Minimize);
        let x1 = problem.add_var(-3.0, (NEG_INFINITY, 20.0));
        let x2 = problem.add_var(-4.0, (5.0, INFINITY));
        problem.add_constraint([(x1, 1.0), (x2, 1.0)], ComparisonOp::Le, 20.0);
        problem.add_constraint([(x1, -1.0), (x2, 4.0)], ComparisonOp::Le, 20.0);

        let res_solver = SimpleSolver::try_new(&problem);
        assert!(res_solver.is_ok());
        let mut solver = res_solver.unwrap();
        assert_eq!(solver.num_vars, 2);
        assert!(!solver.is_primal_feasible);
        assert!(!solver.is_dual_feasible);

        println!("{:?}", solver);

        let solved = solver.restore_primal_feasibility();
        assert!(solved.is_ok(), "Err: {:?}", solved.err());

        println!("{:?}", solver);
    }

    #[test]
    fn solve_base_test() {
        let mut problem = Problem::new(Maximize);
        let x1 = problem.add_var(5.0, (0.0, INFINITY));
        let x2 = problem.add_var(5.0, (0.0, INFINITY));
        let x3 = problem.add_var(3.0, (0.0, INFINITY));
        problem.add_constraint([(x1, 1.0), (x2, 3.0), (x3, 1.0)], ComparisonOp::Le, 3.0);
        problem.add_constraint([(x1, -1.0), (x3, 3.0)], ComparisonOp::Le, 2.0);
        problem.add_constraint([(x1, 2.0), (x2, -1.0), (x3, 2.0)], ComparisonOp::Le, 4.0);
        problem.add_constraint([(x1, 2.0), (x2, 3.0), (x3, -1.0)], ComparisonOp::Le, 2.0);
        let res_solver = SimpleSolver::try_new(&problem);
        assert!(res_solver.is_ok());
        let mut solver = res_solver.unwrap();
        let res = solver.solve();
        assert!(res.is_ok());
        let solution = res.ok().unwrap();
        println!("{:?}", solution);

        assert_eq!(solution.objective_value(), 10.0);
        assert_eq!(solution.var_value(x1), &1.103448275862069);
        assert_eq!(solution.var_value(x2), &0.2758620689655173);
        assert_eq!(solution.var_value(x3), &1.0344827586206897);
    }

    #[test]
    fn solve_base_test2() {
        let mut problem = Problem::new(Maximize);
        let x = problem.add_var(1.0, (0.0, INFINITY));
        let y = problem.add_var(2.0, (0.0, 3.0));

        problem.add_constraint([(x, 1.0), (y, 1.0)], ComparisonOp::Le, 4.0);
        problem.add_constraint([(x, 2.0), (y, 1.0)], ComparisonOp::Ge, 2.0);

        let res_solver = SimpleSolver::try_new(&problem);
        assert!(res_solver.is_ok());
        let mut solver = res_solver.unwrap();
        let res = solver.solve();
        assert!(res.is_ok());
        let solution = res.ok().unwrap();
        println!("{:?}", solution);

        assert_eq!(solution.objective_value(), 7.0);
        assert_eq!(*solution.var_value(x), 1.0);
        assert_eq!(*solution.var_value(y), 3.0);
    }

    #[test]
    fn solve_base_test3() {
        let mut problem = Problem::new(Minimize);
        let x = problem.add_var(64.0, (0.0, INFINITY));
        let y = problem.add_var(26.0, (0.0, INFINITY));

        problem.add_constraint([(x, 2.6), (y, 3.4)], ComparisonOp::Ge, 12.0);
        problem.add_constraint([(x, 1.1), (y, 3.9)], ComparisonOp::Ge, 10.2);
        problem.add_constraint([(x, 33.6), (y, 4.8)], ComparisonOp::Ge, 57.5);
        problem.add_constraint([(x, 1.0), (y, -0.25)], ComparisonOp::Ge, 0.0);
        problem.add_constraint([(x, 1.0), (y, -2.0)], ComparisonOp::Le, 0.0);

        let res_solver = SimpleSolver::try_new(&problem);
        assert!(res_solver.is_ok());
        let mut solver = res_solver.unwrap();
        let res = solver.solve();
        assert!(res.is_ok(), "Err: {:?}", res.err());
        let solution = res.ok().unwrap();
        println!("{:?}", solution);

        assert_eq!(solution.objective_value(), 151.5507075471698);
        assert_eq!(*solution.var_value(x), 1.3551493710691815);
        assert_eq!(*solution.var_value(y), 2.49312106918239);
    }

    #[test]
    fn solve_base_test4a() {
        let mut problem = Problem::new(Maximize);
        let x1 = problem.add_var(3.0, (0.0, INFINITY));
        let x2 = problem.add_var(1.0, (0.0, INFINITY));
        let x3 = problem.add_var(2.0, (0.0, INFINITY));

        problem.add_constraint([(x1, -3.0), (x2, 2.0), (x3, -2.0)], ComparisonOp::Le, 1.0);
        problem.add_constraint([(x2, 1.0), (x3, 1.0)], ComparisonOp::Le, 3.0);
        problem.add_constraint([(x1, 2.0), (x3, 1.0)], ComparisonOp::Le, 2.0);

        let res_solver = SimpleSolver::try_new(&problem);
        assert!(res_solver.is_ok());
        let mut solver = res_solver.unwrap();
        let res = solver.solve();
        assert!(res.is_ok(), "Err: {:?}", res.err());
        let solution = res.ok().unwrap();
        println!("{:?}", solution);

        assert_eq!(solution.objective_value(), 5.6);
        assert_eq!(*solution.var_value(x1), 0.6);
        assert_eq!(*solution.var_value(x2), 2.2);
        assert_eq!(*solution.var_value(x3), 0.8);
    }

    #[test]
    fn solve_base_test4b() {
        let mut problem = Problem::new(Maximize);
        let x1 = problem.add_var(1.0, (0.0, INFINITY));
        let x2 = problem.add_var(1.0, (0.0, INFINITY));
        let x3 = problem.add_var(2.0, (0.0, INFINITY));

        problem.add_constraint([(x1, -3.0), (x2, 2.0), (x3, -2.0)], ComparisonOp::Le, 1.0);
        problem.add_constraint([(x2, 1.0), (x3, 1.0)], ComparisonOp::Le, 3.0);
        problem.add_constraint([(x1, 2.0), (x3, 1.0)], ComparisonOp::Le, 2.0);

        let res_solver = SimpleSolver::try_new(&problem);
        assert!(res_solver.is_ok());
        let mut solver = res_solver.unwrap();
        let res = solver.solve();
        assert!(res.is_ok(), "Err: {:?}", res.err());
        let solution = res.ok().unwrap();
        println!("{:?}", solution);

        assert_eq!(solution.objective_value(), 5.0);
        assert_eq!(*solution.var_value(x1), 0.0);
        assert_eq!(*solution.var_value(x2), 1.0);
        assert_eq!(*solution.var_value(x3), 2.0);
    }
}
