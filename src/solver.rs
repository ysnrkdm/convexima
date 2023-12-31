use crate::{problem::Problem, solution::Solution};

pub enum Error {
    Infeasible,
    Unbounded,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let msg = match self {
            Error::Infeasible => "Problem is infeasible",
            Error::Unbounded => "Problem is unbounded",
        };
        msg.fmt(f)
    }
}

pub trait SolverTryNew<T> {
    fn try_new(problem: &Problem) -> Result<T, Error>;
}

pub trait Solver {
    fn solve(&mut self) -> Result<Solution, Error>;
}
