#include "pc_dbcbs_api.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <limits>
#include <memory>
#include <iostream>

#include <yaml-cpp/yaml.h>

// Boost (you use boost::heap)
#include <boost/heap/d_ary_heap.hpp>

// OMPL (for ob::RealVectorBounds)
#include <ompl/base/spaces/RealVectorBounds.h>
namespace ob = ompl::base;

// FCL
#include <fcl/fcl.h>
#include <fcl/broadphase/broadphase_collision_manager.h>

// Dynoplan / Dynobench + your project headers
#include "dynoplan/tdbastar/tdbastar.hpp"
#include "dynoplan/optimization/payloadTransport_optimization.hpp"
#include "dynoplan/optimization/unicyclesWithRods_optimization.hpp"
#include "dynoplan/optimization/opt_simulate_mujoco.hpp"

#include "dynobench/general_utils.hpp"
#include "dynobench/robot_models_base.hpp"

// IMPORTANT:
// pc_dbcbs_utils.hpp currently defines non-inline functions.
// If this header is included by multiple .cpp files, you'll get multiple-definition link errors.
// You MUST either move implementations to a .cpp OR mark them inline.
#include "pc_dbcbs_utils.hpp"

#include "init_guess_payload.hpp"
#include "init_guess_unicycles.hpp"
#include "init_guess_mujoco.hpp"

// In the CLI you had `using namespace dynoplan;`
// You need the same here because you use Options_tdbastar, Out_info_tdb, Motion, Constraint, etc.
using namespace dynoplan;

#ifndef DYNOBENCH_BASE
#define DYNOBENCH_BASE "../deps/dynoplan/dynobench/"
#endif

namespace fs = std::filesystem;

namespace {

static void write_joint_env_yaml_from_input(const std::string& input_env,
                                            const std::string& out_env_yaml) {
  YAML::Node env = YAML::LoadFile(input_env);

  if (!env["joint_robot"] || env["joint_robot"].size() == 0) {
    throw std::runtime_error("Input env missing 'joint_robot' (needed to generate env.yaml).");
  }

  YAML::Node robotsNode = env["joint_robot"];

  YAML::Node outputEnvYaml = YAML::Clone(env);
  outputEnvYaml.remove("joint_robot");
  outputEnvYaml["robots"] = robotsNode;

  std::ofstream envOut(out_env_yaml);
  if (!envOut) throw std::runtime_error("Could not write env.yaml: " + out_env_yaml);
  envOut << outputEnvYaml;
}

static void write_dummy_init_guess_from_joint_env(const std::string& joint_env_yaml,
                                                  const std::string& out_init_yaml, const size_t N_opt) {
  YAML::Node env = YAML::LoadFile(joint_env_yaml);

  if (!env["robots"] || env["robots"].size() == 0)
    throw std::runtime_error("env.yaml missing 'robots'.");

  auto start = env["robots"][0]["start"].as<std::vector<double>>();
  auto goal  = env["robots"][0]["goal"].as<std::vector<double>>();

  if (start.size() < 3 || goal.size() < 3)
    throw std::runtime_error("State dimension < 3, cannot extract payload position.");

  // action dim: mujoco payload case: 4 * number of quads
  int nu = 4;
  if (env["robots"][0]["quadsNum"]) {
    int n_quads = env["robots"][0]["quadsNum"].as<int>();
    nu = 4 * n_quads;
  }

  // compute N from payload distance and dt
  constexpr double dt = 0.02;
  Eigen::Vector3d p0(start[0], start[1], start[2]);
  Eigen::Vector3d p1(goal[0], goal[1], goal[2]);
  double dist = (p1 - p0).norm();

  // int N = std::max(50, static_cast<int>(std::ceil(dist / dt)));
  size_t N = N_opt;
  // states: N copies of start
  YAML::Node statesNode;
  for (int k = 0; k < N; ++k) {
    YAML::Node row;
    for (double v : start) row.push_back(v);
    statesNode.push_back(row);
  }

  // actions: (N-1) rows, all 1.0
  YAML::Node actionsNode;
  for (int k = 0; k < N - 1; ++k) {
    YAML::Node u;
    for (int i = 0; i < nu; ++i) u.push_back(1.0);
    actionsNode.push_back(u);
  }

  YAML::Node result;
  result["result"]["states"] = statesNode;
  result["result"]["actions"] = actionsNode;
  result["result"]["num_states"] = N;
  result["result"]["num_action"] = N - 1;

  std::ofstream out(out_init_yaml);
  if (!out) throw std::runtime_error("Could not write init_guess yaml: " + out_init_yaml);
  out << result;
}

static void traj_to_mats(const dynobench::Trajectory& tr,
                         Eigen::MatrixXd& X,
                         Eigen::MatrixXd& U) {
  const int T = static_cast<int>(tr.states.size());
  if (T == 0) {
    X.resize(0, 0);
    U.resize(0, 0);
    return;
  }

  const int nx = tr.states[0].size();
  X.resize(T, nx);
  for (int t = 0; t < T; ++t) {
    if (tr.states[t].size() != nx) {
      throw std::runtime_error("Trajectory.states has inconsistent state dimension");
    }
    X.row(t) = tr.states[t].transpose();
  }

  const int Tu = static_cast<int>(tr.actions.size());
  if (Tu == 0) {
    U.resize(0, 0);
    return;
  }

  const int nu = tr.actions[0].size();
  U.resize(Tu, nu);
  for (int t = 0; t < Tu; ++t) {
    if (tr.actions[t].size() != nu) {
      throw std::runtime_error("Trajectory.actions has inconsistent action dimension");
    }
    U.row(t) = tr.actions[t].transpose();
  }
}

static pcdbcbs::Result make_result(bool solved_db,
                                   bool solved_opt,
                                   const std::chrono::duration<double>& duration_discrete,
                                   const std::chrono::duration<double>& duration_opt,
                                   const dynobench::Trajectory& sol) {
  pcdbcbs::Result res;
  res.solved_db = solved_db;
  res.solved_opt = solved_opt;
  res.duration_discrete_sec = duration_discrete.count();
  res.duration_opt_sec = duration_opt.count();

  res.cost = sol.cost;
  res.feasible = sol.feasible;
  res.info = sol.info;

  traj_to_mats(sol, res.X, res.U);
  return res;
}

} // namespace

namespace pcdbcbs {

Result run(const Options& opt) {
  if (opt.input_yaml.empty()) throw std::runtime_error("Options.input_yaml is empty");
  if (opt.output_yaml.empty()) throw std::runtime_error("Options.output_yaml is empty");
  if (opt.optimization_yaml.empty()) throw std::runtime_error("Options.optimization_yaml is empty");
  if (opt.pc_dbcbs_cfg_yaml.empty()) throw std::runtime_error("Options.pc_dbcbs_cfg_yaml is empty");
  if (opt.opt_cfg_yaml.empty()) throw std::runtime_error("Options.opt_cfg_yaml is empty");
  if (opt.motion_primitives_base.empty()) throw std::runtime_error("Options.motion_primitives_base is empty");

  std::string inputFile = opt.input_yaml;
  std::string outputFile = opt.output_yaml;
  std::string optimizationFile = opt.optimization_yaml;
  std::string cfgFile = opt.pc_dbcbs_cfg_yaml;
  std::string optcfgFile = opt.opt_cfg_yaml;
  std::string motion_primitives_base = opt.motion_primitives_base;
  double timeLimit = opt.time_limit;

  // NOTE: we use this both to patch opt_cfg["warmstart"] and (when false) to trigger optimization-only mode.
  bool warmstart_optimization = opt.warmstart_optimization;
  const size_t N_opt = opt.N_opt;
  std::string dynobench_base =
      opt.dynobench_base.empty() ? std::string(DYNOBENCH_BASE) : opt.dynobench_base;

  YAML::Node cfg = YAML::LoadFile(cfgFile);
  if (cfg["pc-dbcbs"]) cfg = cfg["pc-dbcbs"]["default"];

  bool visualize_mujoco_cfg =
      cfg["visualize_mujoco"] ? cfg["visualize_mujoco"].as<bool>() : false;
  bool visualize_mujoco =
      opt.override_visualize_mujoco ? opt.visualize_mujoco : visualize_mujoco_cfg;

  fs::path output_path(outputFile);
  std::string output_folder = output_path.parent_path().string();

  // -------------------------
  // Declare these BEFORE lambdas (fixes your compile errors)
  // -------------------------
  bool solved_db = false;
  bool solved_opt = false;
  std::chrono::duration<double> duration_discrete{0.0};
  std::chrono::duration<double> duration_opt{0.0};

  dynobench::Trajectory sol;
  dynobench::Trajectory sol_broken;

  std::string optcfgFile_effective; // used in normal optimization path too
  // -------------------------

  auto write_patched_opt_cfg = [&](bool warmstart_flag) -> std::string {
    YAML::Node opt_cfg = YAML::LoadFile(optcfgFile);
    // opt_cfg["use_warmstart"] = warmstart_flag;

    fs::path patched = fs::path(output_folder) /
        (std::string("opt_cfg_patched_") + (warmstart_flag ? "1" : "0") + ".yaml");

    YAML::Emitter out;
    out << opt_cfg;

    std::ofstream f(patched.string());
    if (!f) throw std::runtime_error("Could not write patched opt cfg: " + patched.string());
    f << out.c_str();
    f.close();
    return patched.string();
  };

  // IMPORTANT: set optcfgFile_effective for the "normal" path (whatever warmstart_optimization currently is)
  optcfgFile_effective = write_patched_opt_cfg(warmstart_optimization);

  // Optimization-only fallback (your dummy init-guess hack)
  auto run_optimization_only = [&](bool warmstart_flag) -> pcdbcbs::Result {
    // local effective opt cfg for this mode
    std::string optcfgFile_effective_local = write_patched_opt_cfg(warmstart_flag);

    std::string resultPath = outputFile;
    const std::string needle = "result_dbcbs.yaml";
    const std::string repl   = "init_guess_mujoco.yaml";

    size_t pos_resultPath = resultPath.rfind(needle);
    if (pos_resultPath == std::string::npos) {
      throw std::runtime_error(
          "optimization-only mode requires output_yaml to contain 'result_dbcbs.yaml' "
          "so we can place init/env next to it.");
    }
    resultPath.replace(pos_resultPath, needle.size(), repl);

    fs::path folder = fs::path(resultPath).parent_path();
    std::string joint_robot_env_path = (folder / "env.yaml").string();

    // Create env.yaml (joint_robot -> robots)
    write_joint_env_yaml_from_input(inputFile, joint_robot_env_path);

    // Create minimal dummy init guess (guarantees horizon >= 25 and has actions)
    write_dummy_init_guess_from_joint_env(joint_robot_env_path, resultPath, N_opt);

    auto opt_start = std::chrono::steady_clock::now();
    bool sum_robot_cost = true;
    std::string optimizationFile_anytime = optimizationFile;

    bool feasible = execute_optILMujoco(
        joint_robot_env_path, resultPath,
        optimizationFile, optimizationFile_anytime,
        sol, dynobench_base.c_str(),
        sum_robot_cost, sol_broken,
        optcfgFile_effective_local, N_opt);

    duration_opt = std::chrono::steady_clock::now() - opt_start;
    sol.to_yaml_format(optimizationFile.c_str());

    pcdbcbs::Result res;
    res.solved_db = false;
    res.solved_opt = feasible;
    res.duration_discrete_sec = 0.0;
    res.duration_opt_sec = duration_opt.count();
    res.cost = sol.cost;
    res.feasible = sol.feasible;
    res.info = sol.info;
    traj_to_mats(sol, res.X, res.U);
    return res;
  };

  // -------------------------------------------------------------------------
  // If warmstart disabled at API level -> skip pc-dbCBS entirely
  // -------------------------------------------------------------------------
  if (!warmstart_optimization) {
    return run_optimization_only(false);
  }

  // -------------------------------------------------------------------------

  bool anytime_planning = false;
  bool solve_p0 = false;
  float tol = 0.3f;

  if (cfg["payload"]) {
    if (cfg["payload"]["solve_p0"]) {
      solve_p0 = cfg["payload"]["solve_p0"].as<bool>();
      anytime_planning = cfg["payload"]["anytime"].as<bool>();
    }
    if (cfg["payload"]["tol"]) {
      tol = cfg["payload"]["tol"].as<float>();
    }
  }

  payload_pcdbcbs_data pcdbcbs_data;
  pcdbcbs_data.load_from_yaml(cfg["payload_opt"]);

  float alpha = cfg["alpha"].as<float>();
  bool filter_duplicates = cfg["filter_duplicates"].as<bool>();
  (void)filter_duplicates;

  Options_tdbastar options_tdbastar;
  options_tdbastar.outFile = outputFile;
  options_tdbastar.search_timelimit = timeLimit;
  options_tdbastar.alpha = alpha;
  options_tdbastar.cost_delta_factor = 0;
  options_tdbastar.delta = cfg["delta_0"].as<float>();
  options_tdbastar.delta_factor_goal =
      cfg["delta_factor_goal"] ? cfg["delta_factor_goal"].as<float>() : 1.0f;
  options_tdbastar.fix_seed = 0;

  size_t init_prim_num = cfg["num_primitives_0"].as<size_t>();
  if (cfg["max_motions"]) options_tdbastar.max_motions = cfg["max_motions"].as<size_t>();
  else options_tdbastar.max_motions = 100000;

  options_tdbastar.shuffle = cfg["shuffle"] ? cfg["shuffle"].as<bool>() : false;
  options_tdbastar.rewire = true;

  // ------------------------------------------------------------------
  // BACKUP: store initial tdbastar options + primitive settings
  // ------------------------------------------------------------------
  const Options_tdbastar options_tdbastar_initial = options_tdbastar;
  const size_t init_prim_num_initial = init_prim_num;

  // choose a minimum delta threshold (use YAML if provided, else default)
  const float delta_min =
      (cfg["delta_min"] ? cfg["delta_min"].as<float>() : 0.6f);
  // ------------------------------------------------------------------

  bool save_forward_search_expansion = false;
  bool save_reverse_search_expansion = false;

  dynobench::Problem problem(inputFile);
  dynobench::Problem problem_original(inputFile);

  problem.models_base_path = dynobench_base + std::string("models/");

  Out_info_tdb out_tdb;
  std::cout << "*** options_tdbastar ***" << std::endl;
  options_tdbastar.print(std::cout);
  std::cout << "***" << std::endl;

  YAML::Node env = YAML::LoadFile(inputFile);
  std::vector<double> p0_init_guess(3);

  // start is in the ENV yaml under joint_robot[0].start
  p0_init_guess[0] = env["joint_robot"][0]["start"][0].as<double>();
  p0_init_guess[1] = env["joint_robot"][0]["start"][1].as<double>();
  p0_init_guess[2] = env["joint_robot"][0]["start"][2].as<double>();

  std::vector<fcl::CollisionObjectf*> obstacles;
  std::vector<std::shared_ptr<fcl::CollisionGeometryd>> collision_geometries;
  std::vector<Eigen::VectorXf> p0_sol;

  for (const auto& obs : env["environment"]["obstacles"]) {
    if (obs["type"].as<std::string>() == "box") {
      const auto& size = obs["size"];
      std::shared_ptr<fcl::CollisionGeometryf> geom;
      geom.reset(new fcl::Boxf(size[0].as<float>(), size[1].as<float>(), 1.0f));
      const auto& center = obs["center"];
      auto co = new fcl::CollisionObjectf(geom);
      co->setTranslation(fcl::Vector3f(center[0].as<float>(), center[1].as<float>(), 0));
      co->computeAABB();
      obstacles.push_back(co);
    } else {
      throw std::runtime_error("Unknown obstacle type!");
    }
  }

  const auto& env_min = env["environment"]["min"];
  const auto& env_max = env["environment"]["max"];

  ob::RealVectorBounds position_bounds(env_min.size());
  for (size_t k = 0; k < env_min.size(); ++k) {
    position_bounds.setLow(k, env_min[k].as<double>());
    position_bounds.setHigh(k, env_max[k].as<double>());
  }

  std::vector<std::shared_ptr<dynobench::Model_robot>> robots;
  std::string motionsFile;
  std::vector<std::string> all_motionsFile;

  for (const auto& robotType : problem.robotTypes) {
    std::shared_ptr<dynobench::Model_robot> robot = dynobench::robot_factory(
        (problem.models_base_path + robotType + ".yaml").c_str(),
        problem.p_lb, problem.p_ub);
    robots.push_back(robot);

    if (robotType == "unicycle1_v0" || robotType == "unicycle1_sphere_v0") {
      motionsFile = motion_primitives_base + "unicycle1_v0/unicycle1_v0.msgpack";
    } else if (robotType == "unicycle1_v0_no_right") {
      motionsFile = "../motion_primitives/unicycle1_v0/unicycle_no_right.bin.im.bin.sp.bin";
    } else if (robotType == "unicycle2_v0") {
      motionsFile = "../motion_primitives/unicycle2_v0/unicycle2_v0.msgpack";
    } else if (robotType == "car1_v0") {
      motionsFile = "../motion_primitives/car1_v0/car1_v0.msgpack";
    } else if (robotType == "integrator2_2d_v0") {
      motionsFile = "../motion_primitives/integrator2_2d_v0/integrator2_2d_v0.msgpack";
    } else if (robotType == "integrator2_3d_v0") {
      motionsFile = "../motion_primitives/integrator2_3d_v0/integrator2_3d_v0.bin.im.bin.sp.bin";
    } else if (robotType == "quad3d_v0" || startsWith(robots[0]->name, "mujocoquad")) {
      motionsFile = motion_primitives_base + "quad3d_max_tilt_angle_40/quad3d.bin.im.bin.sp.bin";
    } else {
      throw std::runtime_error("Unknown motion filename for this robottype!");
    }

    all_motionsFile.push_back(motionsFile);
  }

  std::map<std::string, std::vector<Motion>> robot_motions;
  std::map<std::string, std::vector<Motion>> sub_motions;

  std::vector<fcl::CollisionObjectd*> robot_objs;
  auto col_mng_robots = std::make_shared<fcl::DynamicAABBTreeCollisionManagerd>();
  col_mng_robots->setup();

  size_t col_geom_id = 0;
  for (size_t rr = 0; rr < robots.size(); ++rr) {
    auto& robot = robots[rr];

    collision_geometries.insert(collision_geometries.end(),
                                robot->collision_geometries.begin(),
                                robot->collision_geometries.end());

    auto robot_obj = new fcl::CollisionObject(collision_geometries[col_geom_id]);
    collision_geometries[col_geom_id]->setUserData((void*)rr);
    robot_objs.push_back(robot_obj);

    if (robot_motions.find(problem.robotTypes[rr]) == robot_motions.end()) {
      options_tdbastar.motionsFile = all_motionsFile[rr];
      load_motion_primitives_new(options_tdbastar.motionsFile,
                                 *robot,
                                 robot_motions[problem.robotTypes[rr]],
                                 options_tdbastar.max_motions,
                                 options_tdbastar.cut_actions,
                                 options_tdbastar.shuffle,
                                 options_tdbastar.check_cols);

      motion_to_motion(robot_motions[problem.robotTypes[rr]],
                       sub_motions[problem.robotTypes[rr]],
                       *robot,
                       init_prim_num);
    }

    col_geom_id++;
  }

  col_mng_robots->registerObjects(robot_objs);

  size_t num_robots = robots.size();
  std::vector<ompl::NearestNeighbors<std::shared_ptr<AStarNode>>*> heuristics(robots.size(), nullptr);
  std::vector<double> upper_bounds(num_robots, std::numeric_limits<double>::max());
  std::vector<double> hs(num_robots, -1.0);

  double lowest_cost = std::numeric_limits<double>::max();
  YAML::Node itr_cost_data;
  std::string itr_cost_file = output_folder + "/iteration_cost.yaml";
  std::string stats_file = output_folder + "/dbcbs_stats.yaml";
  std::vector<dynobench::Trajectory> expanded_trajs_tmp;

  // reverse-search heuristic
  if (cfg["heuristic1"].as<std::string>() == "reverse-search") {
    options_tdbastar.delta = cfg["heuristic1_delta"].as<float>();
    for (size_t robot_id = 0; robot_id < robots.size(); ++robot_id) {
      problem.starts[robot_id].head(robots[robot_id]->translation_invariance)
          .setConstant(std::sqrt(std::numeric_limits<double>::max()));

      Eigen::VectorXd tmp_state = problem.starts[robot_id];
      problem.starts[robot_id] = problem.goals[robot_id];
      problem.goals[robot_id] = tmp_state;

      LowLevelPlan<dynobench::Trajectory> tmp_solution;
      expanded_trajs_tmp.clear();

      options_tdbastar.motions_ptr = &robot_motions[problem.robotTypes[robot_id]];
      tdbastar(problem, options_tdbastar, tmp_solution.trajectory, {},
               out_tdb, robot_id, upper_bounds[robot_id], hs[robot_id],
               true, expanded_trajs_tmp, nullptr, &heuristics[robot_id]);

      if (save_reverse_search_expansion) {
        std::ofstream out2(output_folder + "/expanded_trajs_rev_" + gen_random(6) + ".yaml");
        export_node_expansion(expanded_trajs_tmp, &out2);
      }
    }
  }

  // main loop
  problem.starts = problem_original.starts;
  problem.goals = problem_original.goals;
  options_tdbastar.delta = cfg["delta_0"].as<float>();

  // -------------------------------------------------------------------------
  // TRIVIAL CASE GUARD:
  // If start is already within delta of goal for ALL robots, low-level returns
  // 1 state / 0 actions -> breaks optimization init guess pipeline.
  // In that case, skip pc-dbCBS and run optimization-only with dummy init guess.
  // -------------------------------------------------------------------------
  // {
    bool all_reach_goal = false;
    for (size_t i = 0; i < robots.size(); ++i) {      
      const double d = robots[i]->distance(problem.starts[i], problem.goals[i]);
      std::cout << "check 1 start from goal with delta ofor all robots (delta="
        << options_tdbastar.delta << "and d = " << d << ")" << std::endl;
      if (d <= static_cast<double>(options_tdbastar.delta)) {
        all_reach_goal = true;
        break;
      }
    }
    if (all_reach_goal) {
      std::cout << "[GUARD 1] start already within delta of goal for all robots (delta="
                << options_tdbastar.delta << "). Using optimization-only fallback.\n";
      return run_optimization_only(false);
    }

  // -------------------------------------------------------------------------

  int optimization_counter = 0;
  int optimization_counter_failed = 0;
  std::string optimizationFile_anytime = optimizationFile;

  auto discrete_start = std::chrono::steady_clock::now();

  for (size_t iteration = 0;; ++iteration) {
    // ------------------------------------------------------------------
    // BACKUP: if primitives explode or delta becomes too small, reset
    // ------------------------------------------------------------------
    if (options_tdbastar.delta < delta_min) {
      std::cout << "[BACKUP] Resetting options_tdbastar + primitives. "
                << "init_prim_num=" << init_prim_num
                << " delta=" << options_tdbastar.delta
                << " (threshold=" << delta_min << ")\n";

      options_tdbastar = options_tdbastar_initial;
      // init_prim_num = init_prim_num_initial;

      // // Rebuild sub_motions at the reset primitive count
      // for (size_t rr = 0; rr < problem.robotTypes.size(); ++rr) {
      //   motion_to_motion(robot_motions[problem.robotTypes[rr]],
      //                    sub_motions[problem.robotTypes[rr]],
      //                    *robots[rr],
      //                    init_prim_num);
      // }

      // Restart this HL iteration cleanly with reset settings
      discrete_start = std::chrono::steady_clock::now();
      continue;
    }
    bool all_reach_goal = false;
    double d = 0.0;
    
    for (size_t i = 0; i < robots.size(); ++i) {
      d = robots[i]->distance(problem.starts[i], problem.goals[i]);
       std::cout << "check 2 start from goal with delta ofor all robots (delta="
                << options_tdbastar.delta << "and d = " << d << std::endl;
      if (d <= static_cast<double>(options_tdbastar.delta)) {
        all_reach_goal = true;
        break;
      }
    }

    if (all_reach_goal) {
      std::cout << "[GUARD 2] start already within delta of goal for all robots (delta="
                << options_tdbastar.delta << "). Using optimization-only fallback." << "and d = " << d << std::endl;
      return run_optimization_only(false);
    }

    // ------------------------------------------------------------------

    if (iteration > 0) {
      init_prim_num = init_prim_num + init_prim_num * cfg["num_primitives_rate"].as<float>();
      init_prim_num = std::min<size_t>(init_prim_num, (size_t)1e6);

      for (size_t rr = 0; rr < problem.robotTypes.size(); ++rr) {
        motion_to_motion(robot_motions[problem.robotTypes[rr]],
                         sub_motions[problem.robotTypes[rr]],
                         *robots[rr],
                         init_prim_num);
      }
      discrete_start = std::chrono::steady_clock::now();
    }

    for (size_t rr = 0; rr < problem.robotTypes.size(); ++rr) {
      disable_motions(robots[rr], problem.robotTypes[rr],
                      options_tdbastar.delta, true, alpha,
                      init_prim_num, sub_motions[problem.robotTypes[rr]]);
    }

    solved_db = false;
    solved_opt = false;

    HighLevelNode start;
    start.solution.resize(env["robots"].size());
    start.constraints.resize(env["robots"].size());
    start.cost = 0;
    start.id = 0;

    bool start_node_valid = true;

    for (size_t robot_id = 0; robot_id < robots.size(); ++robot_id) {
      expanded_trajs_tmp.clear();
      options_tdbastar.motions_ptr = &sub_motions[problem.robotTypes[robot_id]];
      load_env(*robots[robot_id], problem);

      tdbastar(problem, options_tdbastar,
               start.solution[robot_id].trajectory,
               start.constraints[robot_id],
               out_tdb, robot_id,
               upper_bounds[robot_id], hs[robot_id],
               false, expanded_trajs_tmp,
               heuristics[robot_id], nullptr);

      if (!out_tdb.solved) {
        start_node_valid = false;
        break;
      }
      start.cost += start.solution[robot_id].trajectory.cost;
    }

    if (!start_node_valid) continue;

    using OpenT = boost::heap::d_ary_heap<HighLevelNode,
                                         boost::heap::arity<2>,
                                         boost::heap::mutable_<true>>;
    OpenT open;
    auto h = open.push(start);
    (*h).handle = h;

    int id = 1;
    double hs_total = std::accumulate(hs.begin(), hs.end(), 0.0);
    auto start_conflict = std::chrono::steady_clock::now();

    while (!open.empty()) {
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - start_conflict);
      if (elapsed.count() >= 40) break;

      HighLevelNode P = open.top();
      open.pop();

      Conflict inter_robot_conflict;
      if (!getEarliestConflict(P.solution, robots, col_mng_robots, robot_objs,
                               inter_robot_conflict, p0_init_guess, p0_sol,
                               solve_p0, tol, pcdbcbs_data)) {
        solved_db = true;
        duration_discrete = std::chrono::steady_clock::now() - discrete_start;

        create_dir_if_necessary(outputFile);
        std::ofstream out(outputFile);
        export_solutions(P.solution, robots.size(), &out, id);

        size_t pos = outputFile.rfind(".yaml");
        std::string joint_robot_env_path;
        std::string resultPath = outputFile;

        size_t pos_resultPath = resultPath.rfind("result_dbcbs.yaml");
        if (pos_resultPath == std::string::npos) {
          throw std::runtime_error("outputFile must contain 'result_dbcbs.yaml' for init_guess path replacement.");
        }

        if (solve_p0) {
          if (startsWith(robots[0]->name, "quad3d")) {
            std::string out_payload = outputFile.substr(0, pos) + "_payload.yaml";
            resultPath.replace(pos_resultPath, std::string("result_dbcbs.yaml").length(), "init_guess_payload.yaml");
            export_solution_p0(p0_sol, out_payload);
            generate_init_guess_payload(inputFile, out_payload, outputFile, resultPath, robots.size(), joint_robot_env_path);
            p0_sol.clear();
          } else if (startsWith(robots[0]->name, "unicycle")) {
            resultPath.replace(pos_resultPath, std::string("result_dbcbs.yaml").length(), "init_guess_unicycles.yaml");
            std::string out_dummy = outputFile.substr(0, pos) + "_unicycles_dummy.yaml";
            export_solution_p0(p0_sol, out_dummy);
            generate_init_guess_unicycles(inputFile, outputFile, resultPath, robots.size(), joint_robot_env_path);
          } else if (startsWith(robots[0]->name, "mujocoquad")) {
            std::string out_mj = outputFile.substr(0, pos) + "_mujoco.yaml";
            resultPath.replace(pos_resultPath, std::string("result_dbcbs.yaml").length(), "init_guess_mujoco.yaml");
            export_solution_p0(p0_sol, out_mj);
            generate_init_guess_mujoco(inputFile, out_mj, outputFile, resultPath, robots.size(), joint_robot_env_path, N_opt);
            p0_sol.clear();
          }
        } else if (startsWith(robots[0]->name, "mujocoquad")) {
          std::string out_mj = outputFile.substr(0, pos) + "_mujoco.yaml";
          resultPath.replace(pos_resultPath, std::string("result_dbcbs.yaml").length(), "init_guess_mujoco.yaml");
          export_solution_p0(p0_sol, out_mj);
          generate_init_guess_mujoco(inputFile, out_mj, outputFile, resultPath, robots.size(), joint_robot_env_path, N_opt);
          p0_sol.clear();
        }

        // Optimization
        auto opt_start = std::chrono::steady_clock::now();
        bool feasible = false;
        double sum_cost = 0.0;

        if (startsWith(robots[0]->name, "quad3d")) {
          bool sum_robot_cost = true;
          optimizationFile_anytime = optimizationFile.substr(0, pos) + "_" +
                                     std::to_string(optimization_counter) +
                                     optimizationFile.substr(pos);
          feasible = execute_payloadTransportOptimization(
              joint_robot_env_path, resultPath,
              optimizationFile, optimizationFile_anytime,
              sol, dynobench_base.c_str(),
              sum_robot_cost, sol_broken);

        } else if (startsWith(robots[0]->name, "unicycle")) {
          bool sum_robot_cost = true;
          optimizationFile_anytime = optimizationFile.substr(0, pos) + "_" +
                                     std::to_string(optimization_counter) +
                                     optimizationFile.substr(pos);
          feasible = execute_unicyclesWithRodsOptimization(
              joint_robot_env_path, resultPath,
              optimizationFile, optimizationFile_anytime,
              sol, dynobench_base.c_str(),
              sum_robot_cost, sol_broken);

        } else if (startsWith(robots[0]->name, "mujoco")) {
          bool sum_robot_cost = true;
          optimizationFile_anytime = optimizationFile.substr(0, pos) + "_" +
                                     std::to_string(optimization_counter) +
                                     optimizationFile.substr(pos);
          feasible = execute_optILMujoco(
              joint_robot_env_path, resultPath,
              optimizationFile, optimizationFile_anytime,
              sol, dynobench_base.c_str(),
              sum_robot_cost, sol_broken,
              optcfgFile_effective, N_opt);
        }

        if (!feasible) {
          add_motion_primitives(problem, sol_broken, sub_motions, robots, sum_cost);
          size_t pos2 = optimizationFile.rfind(".yaml");
          std::string broken =
              optimizationFile.substr(0, pos2) + "_broken_" +
              std::to_string(optimization_counter_failed) +
              optimizationFile.substr(pos2);
          sol_broken.to_yaml_format(broken.c_str());
          optimization_counter_failed++;
          break; // restart HL iteration
        }

        duration_opt = std::chrono::steady_clock::now() - opt_start;
        solved_opt = true;

        add_motion_primitives(problem, sol, sub_motions, robots, sum_cost);

        if (!anytime_planning) {
          sol.to_yaml_format(optimizationFile.c_str());

          if (startsWith(robots[0]->name, "mujoco") && visualize_mujoco) {
            std::string videoPath = resultPath;
            const std::string toReplace = "init_guess_mujoco.yaml";
            auto p2 = videoPath.find(toReplace);
            if (p2 != std::string::npos) videoPath.replace(p2, toReplace.size(), "output.mp4");
            execute_simMujoco(joint_robot_env_path, resultPath, sol,
                              dynobench_base.c_str(), videoPath, "auto", feasible = feasible);
          }

          return make_result(solved_db, solved_opt, duration_discrete, duration_opt, sol);
        }

        if (lowest_cost > sum_cost) {
          sol.to_yaml_format(optimizationFile.c_str());
          sol.to_yaml_format(optimizationFile_anytime.c_str());
          lowest_cost = sum_cost;
          optimization_counter++;
          for (size_t l = 0; l < num_robots; l++) {
            upper_bounds[l] = sum_cost - (hs_total - hs[l]);
          }
        }

        break;
      }

      std::map<size_t, std::vector<Constraint>> constraints;
      createConstraintsFromConflicts(inter_robot_conflict, constraints);

      for (const auto& c : constraints) {
        HighLevelNode newNode = P;
        size_t tmp_robot_id = c.first;
        newNode.id = id;

        newNode.constraints[tmp_robot_id].insert(newNode.constraints[tmp_robot_id].end(),
                                                 c.second.begin(), c.second.end());
        newNode.cost -= newNode.solution[tmp_robot_id].trajectory.cost;

        Out_info_tdb tmp_out_tdb;
        expanded_trajs_tmp.clear();

        options_tdbastar.motions_ptr = &sub_motions[problem.robotTypes[tmp_robot_id]];
        tdbastar(problem, options_tdbastar,
                 newNode.solution[tmp_robot_id].trajectory,
                 newNode.constraints[tmp_robot_id],
                 tmp_out_tdb, tmp_robot_id,
                 upper_bounds[tmp_robot_id], hs[tmp_robot_id],
                 false, expanded_trajs_tmp,
                 heuristics[tmp_robot_id]);

        if (tmp_out_tdb.solved) {
          newNode.cost += newNode.solution[tmp_robot_id].trajectory.cost;
          auto h2 = open.push(newNode);
          (*h2).handle = h2;
          id++;
        }
      }
    }
  }

  Result res;
  res.solved_db = solved_db;
  res.solved_opt = solved_opt;
  res.duration_discrete_sec = duration_discrete.count();
  res.duration_opt_sec = duration_opt.count();
  return res;
}

} // namespace pcdbcbs
