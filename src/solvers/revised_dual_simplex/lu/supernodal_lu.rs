use std::cell::RefCell;

use crate::{consts::EMPTY, sparse::ScatteredVec};

use super::{LUFactorizer, Permutation};

pub struct SupernodalLU {}

impl SupernodalLU {}

impl LUFactorizer for SupernodalLU {
    fn lu_factorize<'a>(
        self,
        col_size: usize,
        get_col: impl Fn(usize) -> (&'a [usize], &'a [f64]),
    ) -> Result<super::LUFactors, super::Error> {
        let perm_c = get_perm_c(ColumnPermutationMethod::Natural, col_size, &get_col);
        let (etree, perm_c_reordered) = get_elim_tree(&perm_c, col_size, &get_col);

        let relaxed_snodes = relax_snode(col_size, &etree, 1);
        let panel_size = 20;

        /*
         *   m = number of rows in the matrix
         *   n = number of columns in the matrix
         *   W = panel size (defaulted to 20)
         *
         *   xprune[0:n-1]: xprune[*] points to locations in subscript
         *	vector lsub[*]. For column i, xprune[i] denotes the point where
         *	structural pruning begins. I.e. only xlsub[i],..,xprune[i]-1 need
         *	to be traversed for symbolic factorization.
         *
         *   marker[0:3*m-1]: marker[i] = j means that node i has been
         *	reached when working on column j.
         *	Storage: relative to original row subscripts
         *	NOTE: There are 3 of them: marker/marker1 are used for panel dfs,
         *	      see dpanel_dfs.c; marker2 is used for inner-factorization,
         *            see dcolumn_dfs.c.
         *
         *   parent[0:m-1]: parent vector used during dfs
         *      Storage: relative to new row subscripts
         *
         *   xplore[0:m-1]: xplore[i] gives the location of the next (dfs)
         *	unexplored neighbor of i in lsub[*]
         *
         *   segrep[0:nseg-1]: contains the list of supernodal representatives
         *	in topological order of the dfs. A supernode representative is the
         *	last column of a supernode.
         *      The maximum size of segrep[] is n.
         *
         *   repfnz[0:W*m-1]: for a nonzero segment U[*,j] that ends at a
         *	supernodal representative r, repfnz[r] is the location of the first
         *	nonzero in this segment.  It is also used during the dfs: repfnz[r]>0
         *	indicates the supernode r has been explored.
         *	NOTE: There are W of them, each used for one column of a panel.
         *
         *   panel_lsub[0:W*m-1]: temporary for the nonzeros row indices below
         *      the panel diagonal. These are filled in during dpanel_dfs(), and are
         *      used later in the inner LU factorization within the panel.
         *	panel_lsub[]/dense[] pair forms the SPA data structure.
         *	NOTE: There are W of them.
         *
         *   dense[0:W*m-1]: sparse accumulating (SPA) vector for intermediate values;
         *	    	   NOTE: there are W of them.
         *
         *   tempv[0:*]: real temporary used for dense numeric kernels;
         *	The size of this array is defined by NUM_TEMPV() in slu_ddefs.h.
         */

        let mut snodes = SuperNodes::new(col_size);
        // let mut xprune = vec![EMPTY; col_size + 1];
        let mut xplore = vec![EMPTY; col_size + 1];
        let mut marker = vec![EMPTY; col_size + 1];
        let mut marker1 = vec![EMPTY; col_size + 1];
        let mut marker2 = vec![EMPTY; col_size + 1];
        let mut dense = vec![0.0; col_size * panel_size + 1];
        let mut tempv = vec![0.0; col_size * panel_size + 1];

        // L related variables
        // let mut sup_to_col = vec![EMPTY; col_size + 1]; // xsup
        // let mut col_to_sup = vec![EMPTY; col_size + 1]; // supno
        // let mut lsub = vec![EMPTY; col_size + 1]; // compressed L subscripts (indexes)
        // let mut xlsub = vec![EMPTY; col_size + 1];
        // let mut lusup = vec![0.0; col_size + 1]; // L supernodes
        // let mut xlusup = vec![EMPTY; col_size + 1];

        // U related variables
        // let mut ucol = vec![0.0; col_size + 1]; // U columns
        // let mut usub = vec![EMPTY; col_size + 1]; // U subscripts
        // let mut xusub = vec![EMPTY; col_size + 1];

        let mut jcol = 0;

        // Work on one "panel" at a time. A panel is one of the following:
        //	   (a) a relaxed supernode at the bottom of the etree, or
        //	   (b) panel_size contiguous columns, defined by the user

        // jcol is the start of a relaxed snode
        // kcol is the end of a relaxed snode
        while jcol < col_size {
            jcol = jcol + 1;
            if relaxed_snodes[jcol] != EMPTY {
                // Work on relaxed (artificial) super nodes
                let kcol = relaxed_snodes[jcol];

                // Factorize the relaxed supernode(jcol:kcol)

                snodes.dsnode_dfs(jcol, kcol, &get_col);

                // let mut nextu = xusub[jcol];
                // let mut nextlu = xlusup[jcol];
                // let mut jsupno = col_to_sup[jcol];
                // let mut fsupc = sup_to_col[jsupno];

                // //
                // let mut max_icol = 0;
                // (jcol..=kcol).for_each(|icol| {
                //     max_icol = icol;
                //     xusub[icol + 1] = nextu;

                //     let mat_col = get_col(icol);
                //     mat_col
                //         .0
                //         .iter()
                //         .copied()
                //         .zip(mat_col.1)
                //         .for_each(|(krow, kval)| {
                //             dense[krow] = *kval;
                //         });

                // // dsnode_bmod
                // // Performs numeric block updates within the relaxed snode.
                // {
                //     // Process the supernodal portion of L\U[*,j]
                //     (xlsub[fsupc]..xlsub[fsupc + 1])
                //         .map(|isub| lsub[isub])
                //         .zip(nextlu..)
                //         .for_each(|(irow, nextlu)| {
                //             lusup[nextlu] = dense[irow];
                //             dense[irow] = 0.0;
                //         });

                //     xlusup[icol + 1] = nextlu + (xlsub[fsupc]..xlsub[fsupc + 1]).len();

                //     if (fsupc < jcol) {
                //         let luptr = xlusup[fsupc];
                //         let nsupr = xlsub[fsupc + 1] - xlsub[fsupc];
                //         let nsupc = jcol - fsupc; /* Excluding jcol */
                //         let ufirst = xlusup[jcol]; /* Points to the beginning of column
                //                                    jcol in supernode L\U(jsupno). */
                //         let nrow = nsupr - nsupc;
                //         // dlsolve
                //         {
                //             let ldm = nsupr;
                //             let ncol = nsupc;
                //             let mut firstcol = 0;

                //             let M0 = lusup[luptr];
                //             let rhs = lusup[ufirst];
                //             // Do 2 columns each (can do 8 cols, 4 cols then 2 cols, but for simplicity)
                //             (0..ncol - 1).step_by(2).for_each(|col| {
                //                 let M1 = luptr + ldm * col;
                //                 let M2 = luptr + ldm * (col+1);
                //                 let x0 = rhs[col]
                //             });
                //         }

                //         // dmatvec
                //     }
                // }
                // });
                // jcol = max_icol;
            } else {
                // Work on one panel of panel_size columns
                todo!("user defined panel size is not supported yet!");
            }
        }
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
    get_col: &impl Fn(usize) -> (&'a [usize], &'a [f64]),
) -> Permutation {
    match ispec {
        ColumnPermutationMethod::Natural => Permutation::identity(size),
        ColumnPermutationMethod::MultipleMinimumDegreeATA => unimplemented!(),
        ColumnPermutationMethod::MultipleMinimumDegreeATPlusA => unimplemented!(),
        ColumnPermutationMethod::ApproxMultipleMinimumDegree => unimplemented!(),
    }
}

// Returns (elim_tree, updated column permutation)
fn get_elim_tree<'a>(
    perm_c: &Permutation,
    size: usize,
    get_col: &impl Fn(usize) -> (&'a [usize], &'a [f64]),
) -> (Vec<usize>, Permutation) {
    let parent = sp_col_elim_tree(perm_c, size, get_col);
    let post = tree_post_order(size, &parent);

    let mut etree = vec![0; size];
    // reorder the parent by post
    for i in 0..size {
        etree[post[i]] = post[parent[i]];
    }

    // reorder the perm_c by post
    let mut perm_c_new = perm_c.clone();
    perm_c_new.reorder(post);

    (etree, perm_c_new)
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
    perm_c: &Permutation,
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
            if rroot != col {
                parent[rroot] = col;
                cset = pp.link(cset, rset);
                root[cset] = col;
            }
        }
    }

    parent.clone()
}

/*
 *  q = TreePostorder (n, p);
 *
 *	Postorder a tree.
 *	Input:
 *	  p is a vector of parent pointers for a forest whose
 *        vertices are the integers 0 to n-1; p[root]==n.
 *	Output:
 *	  q is a vector indexed by 0..n-1 such that q[i] is the
 *	  i-th vertex in a postorder numbering of the tree.
 *
 *        ( 2/7/95 modified by X.Li:
 *          q is a vector indexed by 0:n-1 such that vertex i is the
 *          q[i]-th vertex in a postorder numbering of the tree.
 *          That is, this is the inverse of the previous q. )
 *
 *	In the child structure, lower-numbered children are represented
 *	first, so that a tree which is already numbered in postorder
 *	will not have its order changed.
 *
 *  Written by John Gilbert, Xerox, 10 Dec 1990.
 *  Based on code written by John Gilbert at CMI in 1987.
 */
fn tree_post_order(n: usize, parent: &Vec<usize>) -> Vec<usize> {
    let mut first_kid = vec![EMPTY; n + 1];
    let mut next_kid = vec![0; n + 1];
    let mut post = vec![0; n + 1];

    let mut dad = 0;
    for v in (0..n).rev() {
        dad = parent[v];
        next_kid[v] = first_kid[dad];
        first_kid[dad] = v;
    }

    // nr_etdfs
    /*
     * Depth-first search from vertex n.  No recursion.
     * This routine was contributed by Cédric Doucet, CEDRAT Group, Meylan, France.
     */
    let mut postnum = 0;
    let mut current = n;
    let mut first = 0;
    let mut next = 0;
    while postnum != n {
        first = first_kid[current];
        if first == EMPTY {
            post[current] = postnum;
            postnum += 1;
            next = next_kid[current];

            while next == EMPTY {
                current = parent[current];
                post[current] = postnum;
                postnum += 1;
                next = next_kid[current];
            }

            if postnum == n + 1 {
                return post;
            }

            current = next;
        } else {
            current = first;
        }
    }

    post
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

#[derive(Clone, Debug)]
struct RelaxScratchSpace {
    descendants: Vec<usize>,
    relax_end: Vec<usize>,
}

impl RelaxScratchSpace {
    fn with_capacity(n: usize) -> Self {
        Self {
            descendants: Vec::with_capacity(n),
            relax_end: Vec::with_capacity(n),
        }
    }

    fn clear(&mut self) {
        self.descendants.clear();
        self.relax_end.clear();
    }

    fn clear_and_resize(&mut self, n: usize) {
        self.clear();
        self.descendants.resize(n, 0);
        self.relax_end.resize(n, EMPTY);
    }

    fn len(&self) -> usize {
        self.descendants.len()
    }
}

fn relax_snode(n: usize, etree: &Vec<usize>, relax_columns: usize) -> Vec<usize> {
    thread_local! {
        // Initialize with 1 to cause initialization always at the beginning
        static SCRACH: RefCell<RelaxScratchSpace> = RefCell::new(RelaxScratchSpace::with_capacity(1))
    };

    let result = SCRACH.with(|scratch_| {
        let mut scratch = scratch_.borrow_mut().to_owned();

        if scratch.len() != n {
            scratch.clear_and_resize(n);
        } else {
            scratch.clear();
        }

        let RelaxScratchSpace {
            mut descendants,
            mut relax_end,
        } = scratch;

        // Main Logic
        for j in 0..n {
            let parent = etree[j];
            if parent != n {
                descendants[parent] = descendants[parent] + descendants[j] + 1;
            }
        }
        println!("descendants, {:?}", descendants);

        let mut j = 0;
        while j < n {
            let mut parent = etree[j];
            let snode_start = j;
            while parent != n && descendants[parent] < relax_columns {
                j = parent;
                parent = etree[j];
            }
            relax_end[snode_start] = j;
            j += 1;
            while j < n && descendants[j] != 0 {
                j += 1;
            }
        }
        let ret = relax_end.clone();

        // Wrapping up
        scratch_.replace(RelaxScratchSpace {
            descendants,
            relax_end,
        });

        ret
    });

    result
}

#[derive(Clone, Debug, Default)]
pub(crate) struct CompressedColumnVec {
    col_indices: Vec<usize>,
    row_indices: Vec<usize>,
}

impl CompressedColumnVec {
    pub(crate) fn new() -> Self {
        Self {
            col_indices: vec![0],
            row_indices: vec![],
        }
    }

    pub(crate) fn clear(&mut self) {
        self.col_indices.clear();
        self.row_indices.clear();
    }

    pub(crate) fn push(&mut self, row: usize) {
        self.row_indices.push(row);
    }

    pub(crate) fn seal_column(&mut self) {
        self.col_indices.push(self.row_indices.len())
    }

    pub(crate) fn cols(&self) -> usize {
        self.col_indices.len() - 1
    }
}

#[derive(Debug)]
struct SuperNodes {
    col_size: usize,
    // L related variables
    sup_to_col: Vec<usize>, // xsup
    col_to_sup: Vec<usize>, // supno
    lsub: CompressedColumnVec,
    // lsub: Vec<usize>,   // rowind
    // xlsub: Vec<usize>,  // rowind_colptr
    lusup: Vec<f64>,    // nzval
    xlusup: Vec<usize>, // nzval_colptr: for ith col, lusup[i] to lusup[j] are values

    // U related variables
    ucol: Vec<f64>,
    usub: Vec<usize>,
    xusub: Vec<usize>,

    //
    // xprune: CompressedColumnVec,
    marker_dsnode_dfs: Vec<usize>, // marker[j] = i means that node i has been reached when working on  column j.
}

impl SuperNodes {
    pub fn new(col_size: usize) -> Self {
        Self {
            col_size,
            // L related variables
            sup_to_col: vec![0; col_size + 1], // xsup
            col_to_sup: vec![0; col_size + 1], // supno
            lsub: CompressedColumnVec::new(),  // compressed L subscripts
            lusup: vec![0.0; col_size + 1],    // L supernodes
            xlusup: vec![0; col_size + 1],

            // U related variables
            ucol: vec![0.0; col_size + 1], // U columns
            usub: vec![0; col_size + 1],   // U subscripts
            xusub: vec![0; col_size + 1],

            //
            // xprune: CompressedColumnVec::new(),
            marker_dsnode_dfs: vec![EMPTY; col_size + 1],
        }
    }

    /*
     *    dsnode_dfs() - Determine the union of the row structures of those
     *    columns within the relaxed snode.
     *    Note: The relaxed snodes are leaves of the supernodal etree, therefore,
     *    the portion outside the rectangular supernode must be zero.
     */
    pub fn dsnode_dfs<'a>(
        &mut self,
        jcol: usize,
        kcol: usize,
        get_col: impl Fn(usize) -> (&'a [usize], &'a [f64]),
    ) {
        // row index starting point in lsub
        let mut nextl = 0;
        let mut unioned_nonzero_row_indices = vec![0; self.col_size];

        // col_to_sup[jcol] contains the current max supernode number
        // so +1 to that is the next available supernode number to be applied to [jcol, kcol]
        self.col_to_sup[jcol] += 1;
        let nsuper = if jcol == 0 { 0 } else { self.col_to_sup[jcol] };

        for i in jcol..=kcol {
            // For each nonzero in A[*,i]
            let mat_col = get_col(i);
            mat_col.0.iter().copied().for_each(|krow| {
                // A[krow, i]
                let kmark = self.marker_dsnode_dfs[krow];
                if kmark != kcol {
                    // First time visit krow
                    self.marker_dsnode_dfs[krow] = kcol;
                    unioned_nonzero_row_indices[nextl] = krow;
                    nextl += 1;
                }
            });
        }

        // For columns in [jcol, kcol] -> belongs to the supernode of nsuper
        (jcol..=kcol).for_each(|col| {
            self.col_to_sup[col] = nsuper;
        });

        // If the width of the supernode is greater than 1, copy the subscripts for row indices
        // (for pruning done later in the process)
        for i in jcol..=kcol {
            assert!(self.lsub.cols() == i);
            for r in 0..nextl {
                self.lsub.push(unioned_nonzero_row_indices[r]);
            }
            self.lsub.seal_column();
        }

        self.col_to_sup[kcol + 1] = nsuper; // not nsuper + 1 here to increment in the next iteration
        self.sup_to_col[nsuper + 1] = kcol + 1;
    }

    pub fn dsnode_bmod<'a>(
        &mut self,
        jcol: usize,
        get_col: impl Fn(usize) -> (&'a [usize], &'a [f64]),
        dense_vec: &mut Vec<f64>,
    ) {
        // Process the supernodal portion of L\U[*,jcol]

        // dtrsv: x <- A^(-1)x
        // Solve one of the systems of the form Ax = b, where A is a lower triangular matrix

        // dgemv: y <- alpha*A*x + beta*y where alpha and beta = -1
        // Perform matrix-vector multiplication
    }
}

// dtrsv: x <- A^(-1)x
// Solve one of the systems of the form Ax = b, where A is a lower triangular matrix
fn dlsolve(n: usize, lda: usize, a: &[f64], x: &[f64]) -> Vec<f64> {
    // Initialize the solution vector with the same values as x
    let mut solution = x.to_vec();

    // Perform forward substitution
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += a[i * lda + j] * solution[j];
        }
        solution[i] = (solution[i] - sum) / a[i * lda + i];
    }

    solution
}

// dgemv: y <- A*x + y
// Perform matrix-vector multiplication
fn dgemv(lda: usize, rows: usize, cols: usize, a: &[f64], x: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; rows];
    for i in 0..rows {
        let mut sum = 0.0;
        for j in 0..cols {
            sum += a[i * lda + j] * x[j];
        }
        y[i] = 1.0 * sum + 1.0 * y[i];
    }

    y
}

#[cfg(test)]
mod tests {
    use crate::{
        consts::EMPTY, helpers::helpers::mat_from_triplets,
        solvers::revised_dual_simplex::lu::pretty_print_csmat,
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

        let actual = sp_col_elim_tree(&Permutation::identity(nsize), nsize, |c| {
            test_mat.outer_view(c).unwrap().into_raw_storage()
        });

        pretty_print_csmat(&test_mat);

        assert_eq!(Vec::from_iter(1..=10), actual);
        println!("{:?}", actual);

        let post_order = tree_post_order(nsize, &actual);
        assert_eq!(Vec::from_iter(0..=nsize), post_order);
        println!("{:?}", post_order);
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

        let actual = sp_col_elim_tree(&Permutation::identity(nsize), nsize, |c| {
            test_mat.outer_view(c).unwrap().into_raw_storage()
        });

        pretty_print_csmat(&test_mat);

        assert_eq!(vec![1, 3, 3, 4, 5], actual);
        println!("{:?}", actual);

        let post_order = tree_post_order(nsize, &actual);
        assert_eq!(Vec::from_iter(0..=nsize), post_order);
        println!("{:?}", post_order);
    }

    #[test]
    fn test_tree_post_order() {
        let nsize = 5;
        let post_order = tree_post_order(nsize, &vec![3, 5, 1, 1, 3]);
        assert_eq!(vec![1, 4, 0, 3, 2, 5], post_order);
        println!("{:?}", post_order);
    }

    #[test]
    fn test_relax_snode() {
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

        pretty_print_csmat(&test_mat);

        let (etree, _) = get_elim_tree(
            &Permutation::identity(nsize),
            nsize,
            &(|c| test_mat.outer_view(c).unwrap().into_raw_storage()),
        );

        println!("{:?}", etree);

        let relaxed = relax_snode(nsize, &etree, 3);
        println!("{:?}", relaxed);
        assert_eq!(vec![1, EMPTY, 2, EMPTY, EMPTY], relaxed);
    }

    #[test]
    fn test_dsnode_dfs() {
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

        pretty_print_csmat(&test_mat);

        let mut snodes = SuperNodes::new(nsize);
        snodes.dsnode_dfs(0, 1, |c| test_mat.outer_view(c).unwrap().into_raw_storage());

        println!("{:?}", snodes);

        // 0th to 1st nodes and 2nd nodes pointing to 0
        assert_eq!(vec![0, 0, 0, 0, 0, 0], snodes.col_to_sup);
        // node-0 are columns from 0th (0th in this array) to 2nd (1st in this array) columns
        assert_eq!(vec![0, 2, 0, 0, 0, 0], snodes.sup_to_col);
        // For 0 and 1st columns -> [0,3,4] are non-zero rows
        assert_eq!(vec![0, 3, 6], snodes.lsub.col_indices);
        assert_eq!(vec![0, 3, 4, 0, 3, 4], snodes.lsub.row_indices);

        snodes.dsnode_dfs(2, 4, |c| test_mat.outer_view(c).unwrap().into_raw_storage());

        println!("{:?}", snodes);

        assert_eq!(vec![0, 0, 1, 1, 1, 1], snodes.col_to_sup);
        assert_eq!(vec![0, 2, 5, 0, 0, 0], snodes.sup_to_col);
        assert_eq!(vec![0, 3, 6, 10, 14, 18], snodes.lsub.col_indices);
        assert_eq!(
            vec![0, 3, 4, 0, 3, 4, 1, 2, 0, 4, 1, 2, 0, 4, 1, 2, 0, 4],
            snodes.lsub.row_indices
        );
    }

    #[test]
    fn test_dlsolve() {
        let n = 3;
        let lda = 3;
        let a = vec![
            1.0, 0.0, 0.0, //
            2.0, 1.0, 0.0, //
            3.0, 4.0, 1.0,
        ];
        let x = vec![1.0, 2.0, 3.0];
        let result = dlsolve(n, lda, a.as_slice(), x.as_slice());
        assert_eq!(result, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dlsolve_2() {
        let n = 3;
        let lda = 3;
        let a = vec![
            1.0, 0.0, 0.0, //
            2.0, 1.0, 0.0, //
            3.0, 4.0, 1.0,
        ];
        let x = vec![1.0, 2.0, 2.0];
        let result = dlsolve(n, lda, a.as_slice(), x.as_slice());
        assert_eq!(result, vec![1.0, 0.0, -1.0]);
    }

    #[test]
    fn test_dlsolve_2x2() {
        let n = 2;
        let lda = 2;
        let a = vec![
            1.0, 0.0, //
            2.0, 1.0,
        ];
        let x = vec![1.0, 2.0];
        let result = dlsolve(n, lda, a.as_slice(), x.as_slice());
        assert_eq!(result, vec![1.0, 0.0]);
    }

    #[test]
    fn test_dlsolve_4x4() {
        let n = 4;
        let lda = 4;
        let a = vec![
            1.0, 0.0, 0.0, 0.0, //
            2.0, 1.0, 0.0, 0.0, //
            3.0, -4.0, 1.0, 0.0, //
            4.0, 5.0, 6.0, 1.0,
        ];
        let x = vec![1.0, 0.0, 13.0, 9.0];
        let result = dlsolve(n, lda, a.as_slice(), x.as_slice());
        assert_eq!(result, vec![1.0, -2.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dlsolve_non_zero_off_diagonal() {
        let n = 3;
        let lda = 3;
        let a = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];
        let x = vec![1.0, 2.0, 3.0];
        let result = dlsolve(n, lda, a.as_slice(), x.as_slice());
        assert_eq!(result, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_dgemv_2x2() {
        let m = 2;
        let n = 2;
        let lda = 2;
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 1.0];
        let y = dgemv(lda, m, n, a.as_slice(), x.as_slice());
        assert_eq!(y, vec![3.0, 7.0]);
    }

    #[test]
    fn test_dgemv_3x3() {
        let m = 3;
        let n = 3;
        let lda = 3;
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let x = vec![1.0, 1.0, 1.0];
        let y = dgemv(lda, m, n, a.as_slice(), x.as_slice());
        assert_eq!(y, vec![6.0, 15.0, 24.0]);
    }

    #[test]
    fn test_dgemv_4x4() {
        let m = 4;
        let n = 4;
        let lda = 4;
        let a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let x = vec![1.0, 1.0, 1.0, 1.0];
        let y = dgemv(lda, m, n, a.as_slice(), x.as_slice());
        assert_eq!(y, vec![10.0, 26.0, 42.0, 58.0]);
    }

    #[test]
    fn test_dgemv_non_square() {
        let m = 3;
        let n = 2;
        let lda = 2;
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = vec![1.0, 1.0];
        let y = dgemv(lda, m, n, a.as_slice(), x.as_slice());
        assert_eq!(y, vec![3.0, 7.0, 11.0]);
    }
}
