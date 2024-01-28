use std::{cell::RefCell, cmp::Ordering};

use crate::{
    solvers::revised_dual_simplex::lu::TriangleMat,
    sparse::{ScatteredVec, SparseMat},
};

use super::{Error, LUFactorizer, LUFactors, Permutation};

#[derive(Debug)]
struct ColsQueue {
    head_from_score: Vec<Option<usize>>,
    prev: Vec<usize>,
    next: Vec<usize>,
    min_score: usize,
    len: usize,
}

/// Simplest preordering: order columns based on their size
fn order_simple<'a>(col_size: usize, get_col: impl Fn(usize) -> &'a [usize]) -> Permutation {
    let mut cols_queue = ColsQueue::new(col_size);
    for c in 0..col_size {
        cols_queue.add(c, get_col(c).len() - 1);
    }

    let mut orig_from_new = Vec::with_capacity(col_size);
    while orig_from_new.len() < col_size {
        orig_from_new.push(cols_queue.pop_min().unwrap());
    }

    let mut new_from_orig = vec![0; col_size];
    for (new, &orig) in orig_from_new.iter().enumerate() {
        new_from_orig[orig] = new;
    }

    Permutation {
        new_from_orig,
        orig_from_new,
    }
}

impl ColsQueue {
    fn new(num_cols: usize) -> ColsQueue {
        ColsQueue {
            head_from_score: vec![None; num_cols],
            prev: vec![0; num_cols],
            next: vec![0; num_cols],
            min_score: num_cols,
            len: 0,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn pop_min(&mut self) -> Option<usize> {
        let col = loop {
            if self.min_score >= self.head_from_score.len() {
                return None;
            }
            if let Some(col) = self.head_from_score[self.min_score] {
                break col;
            }
            self.min_score += 1;
        };

        self.remove(col, self.min_score);
        Some(col)
    }

    fn add(&mut self, col: usize, score: usize) {
        self.min_score = std::cmp::min(self.min_score, score);
        self.len += 1;

        if let Some(head) = self.head_from_score[score] {
            self.prev[col] = self.prev[head];
            self.next[col] = head;
            self.next[self.prev[head]] = col;
            self.prev[head] = col;
        } else {
            self.prev[col] = col;
            self.next[col] = col;
            self.head_from_score[score] = Some(col);
        }
    }

    fn remove(&mut self, col: usize, score: usize) {
        self.len -= 1;
        if self.next[col] == col {
            self.head_from_score[score] = None;
        } else {
            self.next[self.prev[col]] = self.next[col];
            self.prev[self.next[col]] = self.prev[col];
            if self.head_from_score[score].unwrap() == col {
                self.head_from_score[score] = Some(self.next[col]);
            }
        }
    }
}

pub struct GPLUFactorizer {
    // Parameters
    stability_coeff: f64,
}

impl GPLUFactorizer {
    pub fn new(stability_coeff: f64) -> GPLUFactorizer {
        Self { stability_coeff }
    }
}

// NNZ = Number of Non-Zero elements
fn count_nnz<'a>(col_size: usize, get_col: impl Fn(usize) -> (&'a [usize], &'a [f64])) -> usize {
    (0..col_size).map(|c| get_col(c).0.len()).sum::<usize>()
}

impl LUFactorizer for GPLUFactorizer {
    fn lu_factorize<'a>(
        self,
        size: usize,
        get_col: impl Fn(usize) -> (&'a [usize], &'a [f64]),
    ) -> Result<LUFactors, Error> {
        let mut lower = SparseMat::new(size);
        let mut upper = SparseMat::new(size);
        let mut upper_diag: Vec<f64> = Vec::with_capacity(size);

        let mut orig_from_new_row = (0..size).collect::<Vec<_>>();
        let mut new_from_orig_row = orig_from_new_row.clone();

        let col_permutation = order_simple(size, |c| get_col(c).0);

        let mut elt_count_from_orig_row = vec![0; size];
        for col_rows in (0..size).map(|c| get_col(c).0) {
            for &orig_r in col_rows {
                elt_count_from_orig_row[orig_r] += 1;
            }
        }

        let mut u_j = ScatteredVec::empty(size);

        for i_orig_col in 0..size {
            let mat_col = get_col(col_permutation.orig_from_new[i_orig_col]);
            // You can assume u_j is empty at this point
            u_j.set(mat_col.0.iter().copied().zip(mat_col.1));
            let reachables = topo_sorted_reachables(
                size,
                &u_j.nonzero,
                |new_i| lower.col_rows(new_i),
                |new_i| new_i < i_orig_col,
                |orig_row| new_from_orig_row[orig_row],
            );

            // println!("{:?} -> {:?}", i_orig_col, reachables);

            for &visited in reachables.visited.iter() {
                if !u_j.is_nonzero[visited] {
                    u_j.is_nonzero[visited] = true;
                    u_j.nonzero.push(visited);
                }
            }

            // dbg!("visited", &reachables.visited);

            for &orig_i in reachables.visited.iter().rev() {
                //
                let new_i = new_from_orig_row[orig_i];
                if new_i < i_orig_col {
                    let x_val = u_j.values[orig_i];
                    for (orig_row, coeff) in lower.col_iter(new_i) {
                        u_j.values[orig_row] -= x_val * coeff;
                    }
                }
            }

            //
            let pivot_orig_row = {
                // dbg!(&u_j);
                // let max_abs = u_j
                //     .nonzero
                //     .par_iter()
                //     .filter(|&&orig_row| new_from_orig_row[orig_row] >= i_orig_col)
                //     .map(|&orig_row| f64::abs(u_j.values[orig_row]))
                //     .max_by(|x, y| x.partial_cmp(y).unwrap())
                //     .unwrap_or(0.0);

                let mut max_abs = 0.0;
                for &orig_row in &u_j.nonzero {
                    if new_from_orig_row[orig_row] < i_orig_col {
                        continue;
                    }

                    let abs = f64::abs(u_j.values[orig_row]);
                    if abs > max_abs {
                        max_abs = abs;
                    }
                }

                if max_abs < 1e-8 {
                    return Err(Error::SingularMatrix);
                }
                assert!(max_abs.is_normal());

                //
                let mut best_orig_row = None;
                let mut best_elt_count = None;
                for &orig_row in &u_j.nonzero {
                    if new_from_orig_row[orig_row] < i_orig_col {
                        continue;
                    }

                    if f64::abs(u_j.values[orig_row]) >= self.stability_coeff * max_abs {
                        let elt_count = elt_count_from_orig_row[orig_row];
                        if best_elt_count.is_none() || best_elt_count.unwrap() > elt_count {
                            best_orig_row = Some(orig_row);
                            best_elt_count = Some(elt_count);
                        }
                    }
                }

                best_orig_row.unwrap()
            };

            let pivot_val = u_j.values[pivot_orig_row];

            {
                let new_row = i_orig_col;
                let orig_row = orig_from_new_row[new_row];
                let pivor_row = new_from_orig_row[pivot_orig_row];
                orig_from_new_row.swap(new_row, pivor_row);
                new_from_orig_row.swap(orig_row, pivot_orig_row);
            }

            //

            for &orig_row in &u_j.nonzero {
                let val = u_j.values[orig_row];
                if val == 0.0 {
                    continue;
                }

                let new_row = new_from_orig_row[orig_row];
                match new_row.cmp(&i_orig_col) {
                    Ordering::Less => upper.push(new_row, val),
                    Ordering::Equal => upper_diag.push(pivot_val),
                    Ordering::Greater => lower.push(orig_row, val / pivot_val),
                }
            }

            upper.seal_column();
            lower.seal_column();
        }

        for i_orig_col in 0..lower.cols() {
            for row in lower.col_rows_mut(i_orig_col) {
                *row = new_from_orig_row[*row];
            }
        }

        Ok(LUFactors {
            lower: TriangleMat {
                nondiag: lower,
                // diagonal elements of the lower triangular matrix is always 1
                diag: None,
            },
            upper: TriangleMat {
                nondiag: upper,
                diag: Some(upper_diag),
            },
            row_permutation: Some(Permutation {
                new_from_orig: new_from_orig_row,
                orig_from_new: orig_from_new_row,
            }),
            col_permutation: Some(col_permutation),
        })
    }
}

#[derive(Clone, Debug)]
struct DfsStep {
    orig_i: usize,
    cur_child: usize,
}

#[derive(Clone, Debug)]
struct TopoScrachSpace {
    dfs_stack: Vec<DfsStep>,
    is_visited: Vec<bool>,
    visited: Vec<usize>,
}

impl TopoScrachSpace {
    fn with_capacity(n: usize) -> Self {
        Self {
            dfs_stack: Vec::with_capacity(n),
            is_visited: vec![false; n],
            visited: vec![],
        }
    }

    fn clear(&mut self) {
        assert!(self.dfs_stack.is_empty());
        for &i in &self.visited {
            self.is_visited[i] = false;
        }
        self.visited.clear();
    }

    fn clear_and_resize(&mut self, n: usize) {
        self.clear();
        self.dfs_stack.reserve(n);
        self.is_visited.resize(n, false);
    }

    fn len(&self) -> usize {
        self.is_visited.len()
    }
}

#[derive(Clone, Debug)]
pub struct Reachables {
    pub visited: Vec<usize>,
    // reachables: Vec<usize>,
}

pub fn topo_sorted_reachables<'a>(
    n: usize,
    initial_reachables: &Vec<usize>,
    get_children: impl Fn(usize) -> &'a [usize] + 'a,
    should_visit: impl Fn(usize) -> bool,
    new_from_orig_row: impl Fn(usize) -> usize,
) -> Reachables {
    thread_local! {
        // Initialize with 1 to cause initialization always at the beginning
        static SCRACH: RefCell<TopoScrachSpace> = RefCell::new(TopoScrachSpace::with_capacity(1))
    };

    let reachables = SCRACH.with(|scratch_| {
        let mut scratch = scratch_.borrow_mut().to_owned();

        if scratch.len() != n {
            dbg!("topo", scratch.len(), n, thread_id::get());
            scratch.clear_and_resize(n);
            dbg!("then", scratch.len(), n, thread_id::get());
        } else {
            scratch.clear();
        }

        let TopoScrachSpace {
            mut dfs_stack,
            mut is_visited,
            mut visited,
        } = scratch;

        // println!("init {:?}", initial_reachables);
        for &orig_row in initial_reachables {
            //
            let new_row = new_from_orig_row(orig_row);
            if !should_visit(new_row) {
                continue;
            }
            if is_visited[orig_row] {
                continue;
            }

            //
            dfs_stack.push(DfsStep {
                orig_i: orig_row,
                cur_child: 0,
            });
            while let Some(current_node) = dfs_stack.last_mut() {
                // println!("visiting: {:?}", current_node);
                //
                let new_i = new_from_orig_row(current_node.orig_i);
                let children = if should_visit(new_i) {
                    get_children(new_i)
                } else {
                    &[]
                };

                let cur_i = current_node.orig_i;

                if is_visited[cur_i] {
                    current_node.cur_child += 1;
                } else {
                    is_visited[cur_i] = true;
                }

                while current_node.cur_child < children.len() {
                    let child_orig_row = children[current_node.cur_child];
                    if !is_visited[child_orig_row] {
                        break;
                    }
                    current_node.cur_child += 1;
                }

                if current_node.cur_child < children.len() {
                    let i_child = current_node.cur_child;
                    dfs_stack.push(DfsStep {
                        orig_i: children[i_child],
                        cur_child: 0,
                    });
                } else {
                    visited.push(cur_i);
                    dfs_stack.pop();
                }
            }
        }

        let visited_ = visited.clone();

        scratch_.replace(TopoScrachSpace {
            dfs_stack,
            is_visited,
            visited,
        });

        Reachables {
            visited: visited_,
            // reachables,
        }
    });

    reachables
}

#[cfg(test)]
mod tests {
    use crate::{
        helpers::helpers::mat_from_triplets, solvers::revised_dual_simplex::lu::pretty_print_csmat,
    };

    use super::*;

    // #[test]
    // fn test_topo_sorted_reachables() {
    //     let f = |i| {
    //         let r: &[usize] = match i {
    //             1 => &[2, 3, 5],
    //             2 | 3 => &[4],
    //             4 => &[],
    //             5 => &[],
    //             _ => panic!("{} cannot be a valid node number!", i),
    //         };
    //         r
    //     };

    //     {
    //         let actual = topo_sorted_reachables(6, &vec![2, 3], f, |_i| true, |i| i);
    //         assert_eq!(vec![4], actual);
    //     }

    //     {
    //         let actual = topo_sorted_reachables(6, &vec![1], f, |_i| true, |i| i);
    //         assert_eq!(vec![4, 2, 3, 5], actual);
    //     }

    //     {
    //         let actual = topo_sorted_reachables(6, &vec![2], f, |_i| true, |i| i);
    //         assert_eq!(vec![4], actual);
    //     }

    //     {
    //         let actual = topo_sorted_reachables(6, &vec![3], f, |_i| true, |i| i);
    //         assert_eq!(vec![4], actual);
    //     }

    //     {
    //         let actual = topo_sorted_reachables(6, &vec![4], f, |_i| true, |i| i);
    //         let empty_vec: Vec<usize> = vec![];
    //         assert_eq!(empty_vec, actual);
    //     }
    // }

    #[test]
    fn test_gplu_lu_factorize() {
        //
        let gplu = GPLUFactorizer {
            stability_coeff: 0.9,
        };

        // [2.0, 2.0, (0.0,) 0.0]
        // [0.0, 0.0, (0.0,) 1.0]
        // [4.0, 3.0, (0.0,) 1.0]
        let test_mat = mat_from_triplets(
            3,
            4,
            &[
                (0, 0, 2.0),
                (0, 1, 2.0),
                (1, 3, 1.0),
                (2, 0, 4.0),
                (2, 1, 3.0),
                (2, 3, 1.0),
            ],
        );

        let factorized_lu = gplu.lu_factorize(test_mat.rows(), |c| {
            test_mat
                .outer_view([0, 1, 3][c])
                .unwrap()
                .into_raw_storage()
        });

        pretty_print_csmat(&test_mat);
        assert!(factorized_lu.is_ok());
        let mat = factorized_lu.unwrap();
        println!("{:?}", mat);
        let multiplied = &mat.lower.to_csmat() * &mat.upper.to_csmat();
        pretty_print_csmat(&multiplied);
    }

    #[test]
    fn test_gplu_lu_factorize_2() {
        //
        let gplu = GPLUFactorizer {
            stability_coeff: 0.9,
        };

        let test_mat = mat_from_triplets(
            6,
            6,
            &[
                (0, 0, 1.0),
                (0, 4, 5.0),
                (0, 5, 1.0),
                (1, 0, 10.0),
                (1, 1, -3.0),
                (1, 2, 6.0),
                (1, 3, 1.0),
                (1, 5, 2.0),
                (2, 0, 3.0),
                (2, 1, 5.0),
                (2, 3, 8.0),
                (2, 4, 4.0),
                (2, 5, 3.0),
                (3, 0, 2.0),
                (3, 1, 4.0),
                (3, 2, 5.0),
                (3, 4, 9.0),
                (3, 5, 4.0),
                (4, 0, 2.0),
                (4, 1, 4.0),
                (4, 2, 5.0),
                (4, 3, 6.0),
                (4, 5, 5.0),
                (5, 0, 9.0),
                (5, 1, 7.0),
                (5, 2, 6.0),
                (5, 3, 3.0),
                (5, 4, 1.0),
            ],
        );

        let factorized_lu = gplu.lu_factorize(test_mat.rows(), |c| {
            test_mat.outer_view(c).unwrap().into_raw_storage()
        });

        pretty_print_csmat(&test_mat);
        assert!(factorized_lu.is_ok());
        let mat = factorized_lu.unwrap();
        println!("{:?}", mat);
        let multiplied = &mat.lower.to_csmat() * &mat.upper.to_csmat();
        pretty_print_csmat(&multiplied);
    }
}
