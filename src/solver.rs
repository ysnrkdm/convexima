use crate::{problem::Problem, solution::Solution};

#[derive(Debug)]
pub enum Error {
    Infeasible,
    Unbounded,
    Abort,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let msg = match self {
            Error::Infeasible => "Problem is infeasible",
            Error::Unbounded => "Problem is unbounded",
            Error::Abort => "Aborted by system",
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
