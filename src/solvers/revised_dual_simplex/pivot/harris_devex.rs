use crate::{consts::EPS, solvers::revised_dual_simplex::NonBasicVarState};

use super::PivotChooser;

#[derive(Debug)]
pub struct HarrisDevex {}

impl PivotChooser for HarrisDevex {
    fn choose_pivot_col(
        &self,
        simplex_tabular: &crate::solvers::revised_dual_simplex::SimpleSolver,
    ) -> Option<usize> {
        let filtered_obj_coeffs = simplex_tabular
            .nb_var_obj_coeffs
            .iter()
            .zip(&simplex_tabular.nb_var_states)
            .enumerate()
            .filter_map(|(col, (&obj_coeff, var_state))| match () {
                _ if var_state.at_min && obj_coeff > -EPS => None,
                _ if var_state.at_max && obj_coeff < EPS => None,
                _ => Some((col, obj_coeff)),
            })
            .rev();

        let best_col = filtered_obj_coeffs
            .map(|(col, obj_coeff)| (col, obj_coeff.abs()))
            .max_by(|(_col1, score1), (_col2, score2)| score1.partial_cmp(score2).unwrap());

        if let Some((col, _score)) = best_col {
            Some(col)
        } else {
            None
        }
    }

    fn choose_pivot_row(
        &self,
        simplex_tabular: &crate::solvers::revised_dual_simplex::SimpleSolver,
        entering_col: usize,
    ) -> Option<(usize, f64)> {
        let entering_col_val = simplex_tabular.nb_var_vals[entering_col];
        let entering_diff_positive = simplex_tabular.nb_var_obj_coeffs[entering_col] < 0.0;
        let entering_other_val = if entering_diff_positive {
            simplex_tabular.orig_var_maxs[simplex_tabular.nb_vars[entering_col]]
        } else {
            simplex_tabular.orig_var_mins[simplex_tabular.nb_vars[entering_col]]
        };

        let get_leaving_var_step = |r: usize, coeff: f64| {
            let val = simplex_tabular.basic_var_vals[r];
            if (entering_diff_positive && coeff < 0.0) || (!entering_diff_positive && coeff > 0.0) {
                let max = simplex_tabular.basic_var_maxs[r];
                if val < max {
                    max - val
                } else {
                    0.0
                }
            } else {
                let min = simplex_tabular.basic_var_mins[r];
                if val > min {
                    val - min
                } else {
                    0.0
                }
            }
        };

        // Harris rule. See e.g.
        // Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1989).
        // A practical anti-cycling procedure for linearly constrained optimization.
        // Mathematical Programming, 45(1-3), 437-474.
        //
        // https://link.springer.com/content/pdf/10.1007/BF01589114.pdf

        // First, we determine the max change in entering variable so that basic variables
        // remain feasible using relaxed bounds.
        let mut max_step = (entering_other_val - entering_col_val).abs();
        for (r, &coeff) in simplex_tabular.col_coeffs.iter() {
            let coeff_abs = coeff.abs();
            if coeff_abs < EPS {
                continue;
            }

            // By which amount can we change the entering variable so that the limit on this
            // basic var is not violated. The var with the minimum such amount becomes leaving.
            let cur_step = (get_leaving_var_step(r, coeff) + EPS) / coeff_abs;
            if cur_step < max_step {
                max_step = cur_step;
            }
        }

        // Second, we choose among variables with steps less than max_step a variable with the biggest
        // abs. coefficient as the leaving variable. This means that we get numerically more stable
        // basis at the price of slight infeasibility of some basic variables.
        let mut leaving_row = None;
        let mut pivot_coeff_abs = f64::NEG_INFINITY;
        let mut pivot_coeff = 0.0;
        for (r, &coeff) in simplex_tabular.col_coeffs.iter() {
            let coeff_abs = coeff.abs();
            if coeff_abs < EPS {
                continue;
            }

            let cur_step = get_leaving_var_step(r, coeff) / coeff_abs;
            if cur_step <= max_step && coeff_abs > pivot_coeff_abs {
                leaving_row = Some(r);
                pivot_coeff = coeff;
                pivot_coeff_abs = coeff_abs;
            }
        }

        leaving_row.map(|r| (r, pivot_coeff))
    }

    fn choose_pivot_col_dual(
        &self,
        simplex_tabular: &crate::solvers::revised_dual_simplex::SimpleSolver,
        row: usize,
    ) -> Option<usize> {
        let leaving_new_val = {
            let val = simplex_tabular.basic_var_vals[row];
            let min = simplex_tabular.basic_var_mins[row];
            let max = simplex_tabular.basic_var_maxs[row];

            if val < min {
                min
            } else if val > max {
                max
            } else {
                unreachable!();
            }
        };

        // True if the new obj. coeff. must be nonnegative in a dual-feasible configuration.
        let leaving_diff_positive = leaving_new_val > simplex_tabular.basic_var_vals[row];

        fn clamp_obj_coeff(mut obj_coeff: f64, var_state: &NonBasicVarState) -> f64 {
            if var_state.at_min && obj_coeff < 0.0 {
                obj_coeff = 0.0;
            }
            if var_state.at_max && obj_coeff > 0.0 {
                obj_coeff = 0.0;
            }
            obj_coeff
        }

        let is_eligible_var = |coeff: f64, var_state: &NonBasicVarState| -> bool {
            let entering_diff_positive = if coeff >= EPS {
                !leaving_diff_positive
            } else if coeff <= -EPS {
                leaving_diff_positive
            } else {
                return false;
            };

            if entering_diff_positive {
                !var_state.at_max
            } else {
                !var_state.at_min
            }
        };

        // Harris rule. See e.g.
        // Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1989).
        // A practical anti-cycling procedure for linearly constrained optimization.
        // Mathematical Programming, 45(1-3), 437-474.
        //
        // https://link.springer.com/content/pdf/10.1007/BF01589114.pdf

        // First, we determine the max step (change in the leaving variable obj. coeff that still
        // leaves us with a dual-feasible state) using relaxed bounds.
        let mut max_step = f64::INFINITY;
        for (c, &coeff) in simplex_tabular.row_coeffs.iter() {
            let var_state = &simplex_tabular.nb_var_states[c];
            if !is_eligible_var(coeff, var_state) {
                continue;
            }

            let obj_coeff = clamp_obj_coeff(simplex_tabular.nb_var_obj_coeffs[c], var_state);
            let cur_step = (obj_coeff.abs() + EPS) / coeff.abs();
            if cur_step < max_step {
                max_step = cur_step;
            }
        }

        // Second, we choose among the variables satisfying the relaxed step bound
        // the one with the biggest pivot coefficient. This allows for a much more
        // numerically stable basis at the price of slight infeasibility in dual variables.
        let mut entering_col = None;
        let mut pivot_coeff_abs = f64::NEG_INFINITY;
        for (c, &coeff) in simplex_tabular.row_coeffs.iter() {
            let var_state = &simplex_tabular.nb_var_states[c];
            if !is_eligible_var(coeff, var_state) {
                continue;
            }

            let obj_coeff = clamp_obj_coeff(simplex_tabular.nb_var_obj_coeffs[c], var_state);

            // If we change obj. coeff of the leaving variable by this amount,
            // obj. coeff if the current variable will reach the bound of dual infeasibility.
            // Variable with the tightest such bound is the entering variable.
            let cur_step = obj_coeff.abs() / coeff.abs();
            if cur_step <= max_step {
                let coeff_abs = coeff.abs();
                if coeff_abs > pivot_coeff_abs {
                    entering_col = Some(c);
                    pivot_coeff_abs = coeff_abs;
                }
            }
        }

        entering_col
    }

    fn choose_pivot_row_dual(
        &self,
        simplex_tabular: &crate::solvers::revised_dual_simplex::SimpleSolver,
    ) -> Option<usize> {
        let infeasibilities = simplex_tabular
            .basic_var_vals
            .iter()
            .zip(&simplex_tabular.basic_var_mins)
            .zip(&simplex_tabular.basic_var_maxs)
            .enumerate()
            .filter_map(|(r, ((&val, &min), &max))| {
                if val < min - EPS {
                    // dbg!(r, val, min, max, min - val);
                    Some((r, min - val))
                } else if val > max + EPS {
                    // dbg!(r, val, min, max, val - max);
                    Some((r, val - max))
                } else {
                    // dbg!(r, val, min, max);
                    None
                }
            })
            .rev();

        let most_infeasible =
            infeasibilities.max_by(|(_r1, v1), (_r2, v2)| v1.partial_cmp(v2).unwrap());

        most_infeasible.map(|(r, _max_score)| r)
    }
}
