use super::SimpleSolver;

pub mod harris_devex;

pub trait PivotChooser {
    fn choose_pivot_col(&self, simplex_tabular: &SimpleSolver) -> Option<usize>;
    fn choose_pivot_row(
        &self,
        simplex_tabular: &SimpleSolver,
        entering_col: usize,
    ) -> Option<(usize, f64)>;
    fn choose_pivot_col_dual(&self, simplex_tabular: &SimpleSolver, row: usize) -> Option<usize>;
    fn choose_pivot_row_dual(&self, simplex_tabular: &SimpleSolver) -> Option<usize>;
}
