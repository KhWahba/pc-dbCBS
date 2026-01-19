import numpy as np
import sys
sys.path.append("/home/khaledwahba94/inria/pc-dbCBS/build_bindings")
import pcdbcbs

def main():
    opt = pcdbcbs.Options()
    # # === Match your CLI arguments ===
    opt.input_yaml = "deps/dynoplan/dynobench/envs/mujoco/mujocoquadspayload_empty2.yaml"
    opt.output_yaml = "stats_db/run_pcdbcbs_bindings/result_dbcbs.yaml"
    opt.optimization_yaml = "stats_db/run_pcdbcbs_bindings/result_dbcbs_opt.yaml"
    opt.pc_dbcbs_cfg_yaml = "configs/pc_dbcbs_bindings.yaml"
    opt.opt_cfg_yaml = "configs/opt.yaml"
    opt.time_limit = 350000.0

    # # Strongly recommended (avoids relative-path hell)
    opt.dynobench_base = "deps/dynoplan/dynobench/"
    opt.motion_primitives_base = "motion_primitives/"
    print("Running pc-dbCBS...")
    res = pcdbcbs.run(opt)

    print("=== Result ===")
    print("solved_db:", res.solved_db)
    print("solved_opt:", res.solved_opt)
    print("feasible:", res.feasible)
    print("cost:", res.cost)

    if res.X.size > 0:
        print("X shape:", res.X.shape)
        print("U shape:", res.U.shape)
        
        # Example: save trajectory
        np.save("stats_db/run_pcdbcbs_bindings/X.npy", res.X)
        np.save("stats_db/run_pcdbcbs_bindings/U.npy", res.U)

if __name__ == "__main__":
    main()
