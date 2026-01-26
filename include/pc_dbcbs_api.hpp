#pragma once

#include <Eigen/Dense>
#include <string>

namespace pcdbcbs {

/**
 * Matches your CLI:
 *   ./pc_dbcbs -i INPUT -o OUTPUT --optimization OPT --pc_dbcbs_cfg CFG --opt_cfg OPTCFG -t TIME
 */
struct Options {
  std::string input_yaml;          // -i
  std::string output_yaml;         // -o (discrete pc-dbCBS result yaml)
  std::string optimization_yaml;   // --optimization (optimized joint trajectory yaml)
  std::string pc_dbcbs_cfg_yaml;   // --pc_dbcbs_cfg
  std::string opt_cfg_yaml;        // --opt_cfg
  double time_limit = 0.0;         // -t (same unit as your code uses)

  // Optional convenience overrides (keep defaults to match current behavior)
  bool override_visualize_mujoco = false;
  bool visualize_mujoco = false;

  // If your code uses DYNOBENCH_BASE macro, expose it here to avoid cwd issues.
  // Leave empty to keep current behavior / macro.
  std::string dynobench_base = "";
  std::string motion_primitives_base = "";
  bool warmstart_optimization = true;
};

struct Result {
  // High-level solve flags
  bool solved_db = false;   // found conflict-free discrete plan
  bool solved_opt = false;  // optimization succeeded

  // Timing (seconds)
  double duration_discrete_sec = 0.0;
  double duration_opt_sec = 0.0;

  // Trajectory info (from dynobench::Trajectory sol)
  double cost = 1e8;
  bool feasible = false;
  std::string info;

  // NumPy-ready matrices
  // X: (T, nx) from sol.states
  // U: (T-1, nu) from sol.actions (often T-1, but we don't enforce)
  Eigen::MatrixXd X;
  Eigen::MatrixXd U;
};

/**
 * Core entrypoint: runs the same pipeline as the CLI version:
 *  - loads YAMLs
 *  - runs pc-dbCBS (discrete)
 *  - generates init guess
 *  - runs optimization (mujoco/payload/unicycle depending on robots[0]->name)
 *  - writes output_yaml and optimization_yaml (same as CLI)
 *  - returns Result with X/U from the optimized trajectory (sol)
 *
 * Throws std::runtime_error on fatal errors (bad yaml, missing files, etc).
 */
Result run(const Options& opt);

/**
 * Convenience overload matching your CLI parameters one-to-one.
 */
inline Result run(const std::string& input_yaml,
                  const std::string& output_yaml,
                  const std::string& optimization_yaml,
                  const std::string& pc_dbcbs_cfg_yaml,
                  const std::string& opt_cfg_yaml,
                  double time_limit) {
  Options o;
  o.input_yaml = input_yaml;
  o.output_yaml = output_yaml;
  o.optimization_yaml = optimization_yaml;
  o.pc_dbcbs_cfg_yaml = pc_dbcbs_cfg_yaml;
  o.opt_cfg_yaml = opt_cfg_yaml;
  o.time_limit = time_limit;
  return run(o);
}

}  // namespace pcdbcbs
