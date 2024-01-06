use crate::problem::{OptimizationDirection, Variable};

#[derive(Debug)]
pub struct Solution {
    direction: OptimizationDirection,
    num_vars: usize,
    objective_val: f64,
    optimal_vars: Vec<f64>,
}

impl Solution {
    pub fn new(
        direction: OptimizationDirection,
        num_vars: usize,
        objective_val: f64,
        optimal_vars: Vec<f64>,
    ) -> Self {
        Solution {
            direction,
            num_vars,
            objective_val,
            optimal_vars,
        }
    }

    pub fn objective_value(&self) -> f64 {
        match self.direction {
            OptimizationDirection::Minimize => self.objective_val,
            OptimizationDirection::Maximize => -self.objective_val,
        }
    }

    pub fn var_value(&self, var: Variable) -> &f64 {
        assert!(var.0 < self.num_vars);
        &self.optimal_vars[var.0]
    }
}
