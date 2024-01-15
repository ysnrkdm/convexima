#[cfg(test)]
mod integration_test {
    use std::io;

    use convexima::{
        self,
        helpers::msp::MpsFile,
        problem::OptimizationDirection,
        solver::{Solver, SolverTryNew},
        solvers::revised_dual_simplex::SimpleSolver,
    };

    #[test]
    fn adlittle_test() {
        let _ = env_logger::builder().is_test(true).try_init();

        let direction = OptimizationDirection::Minimize;
        let file = {
            let file = std::fs::File::open("resources/lpopt-instances/adlittle.msp").unwrap();
            let input = io::BufReader::new(file);
            MpsFile::parse(input, direction).unwrap()
        };

        let mut solver = SimpleSolver::try_new(&file.problem).ok().unwrap();

        let solution = solver.solve().unwrap();
        assert_eq!(225494.96316237998, solution.objective_value());
        println!("objective value: {}", solution.objective_value());
    }
}
