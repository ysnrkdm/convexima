use crate::sparse::ScatteredVec;

use super::{LUFactorizer, Permutation};

pub struct SupernodalLU {}

impl SupernodalLU {}

impl LUFactorizer for SupernodalLU {
    fn lu_factorize<'a>(
        self,
        col_size: usize,
        get_col: impl Fn(usize) -> (&'a [usize], &'a [f64]),
    ) -> Result<super::LUFactors, super::Error> {
        todo!()
    }
}

enum ColumnPermutationMethod {
    Natural,
    MultipleMinimumDegreeATA,
    MultipleMinimumDegreeATPlusA,
    ApproxMultipleMinimumDegree,
}

fn get_perm_c<'a>(
    ispec: ColumnPermutationMethod,
    size: usize,
    get_col: impl Fn(usize) -> (&'a [usize], &'a [f64]),
) -> Permutation {
    match ispec {
        ColumnPermutationMethod::Natural => Permutation::identity(size),
        ColumnPermutationMethod::MultipleMinimumDegreeATA => unimplemented!(),
        ColumnPermutationMethod::MultipleMinimumDegreeATPlusA => unimplemented!(),
        ColumnPermutationMethod::ApproxMultipleMinimumDegree => unimplemented!(),
    }
}

fn get_elim_tree<'a>(
    perm_c: Permutation,
    size: usize,
    get_col: impl Fn(usize) -> (&'a [usize], &'a [f64]),
) -> Vec<usize> {
    todo!();
}

/*
 *      Find the elimination tree for A'*A.
 *      This uses something similar to Liu's algorithm.
 *      It runs in time O(nz(A)*log n) and does not form A'*A.
 *
 *      Input:
 *        Sparse matrix A.  Numeric values are ignored, so any
 *        explicit zeros are treated as nonzero.
 *      Output:
 *        Integer array of parents representing the elimination
 *        tree of the symbolic product A'*A.  Each vertex is a
 *        column of A, and nc means a root of the elimination forest.
 *
 *      John R. Gilbert, Xerox, 10 Dec 1990
 *      Based on code by JRG dated 1987, 1988, and 1990.
 */

/*
 * Nonsymmetric elimination tree
 */
fn sp_col_elim_tree<'a>(
    perm_c: Permutation,
    size: usize,
    get_col: impl Fn(usize) -> (&'a [usize], &'a [f64]),
) -> Vec<usize> {
    let ncol = size;
    let nrow = size;

    let mut pp = DisjointSet::new(size);

    // Working column vector reused in this function
    let mut col_vec = ScatteredVec::empty(size);

    // Compute firstcol[row] = first nonzero column in row

    let mut firstcol = vec![nrow; ncol];

    (0..ncol).for_each(|i_orig_col| {
        let mat_col = get_col(perm_c.new_from_orig[i_orig_col]);
        col_vec.set(mat_col.0.iter().copied().zip(mat_col.1));
        for (i_row, &_val) in col_vec.iter() {
            if firstcol[i_row] > i_orig_col {
                firstcol[i_row] = i_orig_col;
            }
        }
    });

    // Compute etree by Liu's algorithm for symmetric matrices,
    // except use (firstcol[r],c) in place of an edge (r,c) of A.
    // Thus each row clique in A'*A is replaced by a star
    // centered at its first vertex, which has the same fill.

    let mut root = vec![0; ncol];
    let mut parent = vec![0; ncol];
    for col in 0..ncol {
        //
        let mut cset = pp.make_set(col);
        root[cset] = col;
        parent[col] = ncol;
        let mat_col = get_col(perm_c.new_from_orig[col]);
        col_vec.set(mat_col.0.iter().copied().zip(mat_col.1));

        for (p, &_val) in col_vec.iter() {
            let row = firstcol[p];
            if row >= col {
                continue;
            }
            let rset = pp.find(row);
            let rroot = root[rset];
            if (rroot != col) {
                parent[rroot] = col;
                cset = pp.link(cset, rset);
                root[cset] = col;
            }
        }
    }

    parent.clone()
}

struct DisjointSet {
    pp: Vec<usize>,
}

impl DisjointSet {
    fn new(size: usize) -> Self {
        DisjointSet { pp: vec![0; size] }
    }

    fn make_set(&mut self, i: usize) -> usize {
        self.pp[i] = i;
        i
    }

    fn link(&mut self, s: usize, t: usize) -> usize {
        self.pp[s] = t;
        t
    }

    fn find(&mut self, i: usize) -> usize {
        let mut ii = i;
        let mut p = self.pp[ii];
        let mut gp = self.pp[p];
        while gp != p {
            self.pp[ii] = gp;
            ii = gp;
            p = self.pp[ii];
            gp = self.pp[p];
        }

        p
    }
}

fn tree_post_order(n: usize, elim_tree: Vec<usize>) -> Vec<usize> {
    todo!();
}

#[cfg(test)]
mod tests {
    use crate::{
        helpers::helpers::mat_from_triplets, solvers::revised_dual_simplex::lu::pretty_print_csmat,
    };

    use super::*;

    #[test]
    fn test_sp_col_elim_tree_base1() {
        let nsize = 10;
        let test_mat = mat_from_triplets(
            nsize,
            nsize,
            &[
                (0, 0, 1.0),
                (0, 1, 1.0),
                (0, 2, 1.0),
                (0, 5, 1.0), //
                (1, 0, 1.0),
                (1, 1, 1.0),
                (1, 3, 1.0),
                (1, 7, 1.0), //
                (2, 2, 1.0),
                (2, 6, 1.0),
                (2, 7, 1.0), //
                (3, 3, 1.0),
                (3, 4, 1.0),
                (3, 5, 1.0),
                (3, 8, 1.0), //
                (4, 3, 1.0),
                (4, 6, 1.0), //
                (5, 0, 1.0),
                (5, 5, 1.0),
                (5, 8, 1.0), //
                (6, 6, 1.0),
                (6, 8, 1.0),
                (6, 9, 1.0), //
                (7, 0, 1.0),
                (7, 2, 1.0), //
                (8, 3, 1.0),
                (8, 5, 1.0),
                (8, 8, 1.0), //
                (9, 2, 1.0),
                (9, 6, 1.0),
                (9, 8, 1.0),
                (9, 9, 1.0), //
            ],
        );

        let actual = sp_col_elim_tree(Permutation::identity(nsize), nsize, |c| {
            test_mat.outer_view(c).unwrap().into_raw_storage()
        });

        pretty_print_csmat(&test_mat);

        assert_eq!(Vec::from_iter(1..=10), actual);
        println!("{:?}", actual);
    }

    #[test]
    fn test_sp_col_elim_tree_base2() {
        let nsize = 5;
        let test_mat = mat_from_triplets(
            nsize,
            nsize,
            &[
                (0, 0, 1.0),
                (0, 3, 1.0), //
                (1, 2, 1.0),
                (1, 3, 1.0),
                (1, 4, 1.0), //
                (2, 2, 1.0),
                (2, 4, 1.0), //
                (3, 0, 1.0),
                (3, 1, 1.0), //
                (4, 1, 1.0),
                (4, 4, 1.0),
            ],
        );

        let actual = sp_col_elim_tree(Permutation::identity(nsize), nsize, |c| {
            test_mat.outer_view(c).unwrap().into_raw_storage()
        });

        pretty_print_csmat(&test_mat);

        assert_eq!(vec![1, 3, 3, 4, 5], actual);
        println!("{:?}", actual);
    }
}
