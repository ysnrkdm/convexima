use crate::sparse::SparseMat;

#[derive(Clone, Debug)]
pub struct EtaMatrices {
    pub leaving_rows: Vec<usize>,
    pub coeff_cols: SparseMat,
}

impl EtaMatrices {
    pub fn new(n_rows: usize) -> EtaMatrices {
        EtaMatrices {
            leaving_rows: vec![],
            coeff_cols: SparseMat::new(n_rows),
        }
    }

    pub fn len(&self) -> usize {
        self.leaving_rows.len()
    }

    pub fn clear_and_resize(&mut self, n_rows: usize) {
        self.leaving_rows.clear();
        self.coeff_cols.clear_and_resize(n_rows);
    }

    pub fn push(&mut self, leaving_row: usize, coeffs: impl Iterator<Item = (usize, f64)>) {
        self.leaving_rows.push(leaving_row);
        self.coeff_cols.append_col(coeffs);
    }

    pub fn nnz(&self) -> usize {
        self.coeff_cols.nnz()
    }
}
