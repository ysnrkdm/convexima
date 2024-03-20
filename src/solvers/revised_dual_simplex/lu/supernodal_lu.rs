use std::cell::RefCell;

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
        let perm_c = get_perm_c(ColumnPermutationMethod::Natural, col_size, &get_col);
        let (etree, perm_c_reordered) = get_elim_tree(&perm_c, col_size, &get_col);

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

        let (etree, _) = get_elim_tree(
            &Permutation::identity(nsize),
            nsize,
            &(|c| test_mat.outer_view(c).unwrap().into_raw_storage()),
        );

        println!("{:?}", etree);

        let relaxed = relax_snode(nsize, &etree, 1);
        println!("{:?}", relaxed);
        assert_eq!(vec![0, EMPTY, 2, EMPTY, EMPTY], relaxed);
    }
}
