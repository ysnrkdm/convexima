use crate::sparse::SparseMat;

#[derive(Clone, Debug)]
struct EtaMatrices {
    leaving_rows: Vec<usize>,
    coeff_cols: SparseMat,
}
