macro_rules! test_for {
    ($test_name_: ident, $file_name_wo_suffix_: expr, $time_out_secs_:expr, $expected_objective_value_: expr) => {
        // #[ignore]
        #[test]
        fn $test_name_() {
            use std::time::Duration;
            let handle = procspawn::spawn((), |_| {
                let _ = env_logger::builder().is_test(true).try_init();

                let direction = OptimizationDirection::Minimize;
                let filepath = concat!(
                    // "tests/",
                    "resources/lpopt-instances/",
                    $file_name_wo_suffix_,
                    ".msp"
                );
                dbg!(filepath);
                let file = {
                    let file = std::fs::File::open(filepath).unwrap();
                    let input = io::BufReader::new(file);
                    MpsFile::parse(input, direction).unwrap()
                };

                let mut solver = SimpleSolver::try_new(&file.problem).ok().unwrap();

                let solution = solver.solve().unwrap();
                assert_eq!($expected_objective_value_, solution.objective_value());
                println!("objective value: {}", solution.objective_value());
            });

            match handle.join_timeout(Duration::from_secs($time_out_secs_)) {
                Ok(_result) => assert_eq!(1, 1),
                Err(e) => panic!("{}", e),
            }
        }
    };
}

#[cfg(test)]
mod lpopt_test {
    use std::io;

    use convexima::{
        self,
        helpers::msp::MpsFile,
        problem::OptimizationDirection,
        solver::{Solver, SolverTryNew},
        solvers::revised_dual_simplex::SimpleSolver,
    };

    procspawn::enable_test_support!();

    const TIMEOUT_SECS: u64 = 300;

    test_for!(adlittle, "adlittle", TIMEOUT_SECS, 225494.96316237998);
    // test_for!(a2864, "a2864", 3, 225494.1);
    test_for!(cont1, "cont1", TIMEOUT_SECS, 150.0);
    test_for!(bdry2, "bdry2", TIMEOUT_SECS, 150.0);

    // Probably solvable within
    // test_for!(qap15, "qap15", TIMEOUT_SECS, 150.0);
    test_for!(nug08_3rd, "nug08-3rd", TIMEOUT_SECS, 2.14000000e+02);
    test_for!(fome13, "fome13", TIMEOUT_SECS, 9.01311684e+07);
    test_for!(Linf_520c, "Linf_520c", TIMEOUT_SECS, 150.0);
}
