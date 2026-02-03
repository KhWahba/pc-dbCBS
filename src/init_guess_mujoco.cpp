#include "init_guess_mujoco.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <algorithm>
#include "dynobench/general_utils.hpp"

// Pad a matrix to match the maximum number of rows by repeating the last row
Eigen::MatrixXd pad_mat(const Eigen::MatrixXd &matrix, int maxRows) {
  Eigen::MatrixXd padded = Eigen::MatrixXd::Zero(maxRows, matrix.cols());
  int currentRows = (int)matrix.rows();

  // Copy existing rows
  if (currentRows > 0) {
    padded.topRows(currentRows) = matrix;

    // Pad remaining rows with the last row
    for (int i = currentRows; i < maxRows; ++i) {
      padded.row(i) = matrix.row(currentRows - 1);
    }
  }
  return padded;
}

// Clip actions to a threshold
Eigen::MatrixXd clip_act(const Eigen::MatrixXd &actions, double threshold) {
  Eigen::MatrixXd clipped = actions;
  for (int i = 0; i < clipped.rows(); ++i) {
    for (int j = 0; j < clipped.cols(); ++j) {
      clipped(i, j) = std::clamp(clipped(i, j), 0.0, threshold);
    }
  }
  return clipped;
}

void generate_init_guess_mujoco(std::string &envPath,
                                std::string &payloadPath,
                                std::string &dbcbsPath,
                                std::string &resultPath,
                                size_t numRobots,
                                std::string &joint_robot_env_path,
                                size_t N_opt) {
  YAML::Node env = YAML::LoadFile(envPath);
  YAML::Node dbcbs = YAML::LoadFile(dbcbsPath);

  if (!env["joint_robot"] || env["joint_robot"].size() == 0) {
    throw std::runtime_error("joint_robot key is missing or empty in the provided environment YAML.");
  }

  std::string robotname = env["joint_robot"][0]["type"].as<std::string>();
  std::cout << "generating init guess for: " << envPath
            << ", and robot name is: " << robotname << std::endl;

  int maxRowsStates = 0, maxRowsActions = 0;
  std::vector<Eigen::MatrixXd> robotStates(numRobots), robotActions(numRobots);

  bool special_case_single_state_no_action = false;

  // === Load robot states/actions ===
  for (size_t i = 0; i < numRobots; ++i) {
    auto statesNode = dbcbs["result"][i]["states"];
    auto actionsNode = dbcbs["result"][i]["actions"];

    if (!statesNode || !statesNode.IsSequence() || statesNode.size() == 0) {
      throw std::runtime_error("dbcbs result has no states for robot " + std::to_string(i));
    }

    // states matrix
    Eigen::MatrixXd stateMatrix((int)statesNode.size(), (int)statesNode[0].size());
    for (size_t r = 0; r < statesNode.size(); ++r) {
      for (size_t c = 0; c < statesNode[r].size(); ++c) {
        stateMatrix((int)r, (int)c) = statesNode[r][c].as<double>();
      }
    }

    // actions matrix (may be empty)
    Eigen::MatrixXd actionMatrix;
    bool actions_ok = (actionsNode && actionsNode.IsSequence() && actionsNode.size() > 0);

    if (actions_ok) {
      actionMatrix.resize((int)actionsNode.size(), (int)actionsNode[0].size());
      for (size_t j = 0; j < actionsNode.size(); ++j) {
        for (size_t k = 0; k < actionsNode[j].size(); ++k) {
          actionMatrix((int)j, (int)k) = actionsNode[j][k].as<double>();
        }
      }
    } else {
      actionMatrix.resize(0, 0);
    }

    std::cout << "robot " << i << " state size: "
              << stateMatrix.rows() << "x" << stateMatrix.cols() << std::endl;
    std::cout << "robot " << i << " action size: "
              << actionMatrix.rows() << "x" << actionMatrix.cols() << std::endl;

    // Special case: exactly 1 state and no actions
    if (stateMatrix.rows() == 1 && actionMatrix.rows() == 0) {
      special_case_single_state_no_action = true;
    }

    robotStates[i] = stateMatrix;
    robotActions[i] = (actionMatrix.rows() > 0) ? clip_act(actionMatrix, 1.4) : actionMatrix;

    maxRowsStates = std::max(maxRowsStates, (int)stateMatrix.rows());
    maxRowsActions = std::max(maxRowsActions, (int)actionMatrix.rows());
  }

  // === Build final outputs ===
  Eigen::MatrixXd finalStates;
  Eigen::MatrixXd concatenatedActions;

  // ------------------------------------------------------------
  // SPECIAL CASE: dbCBS returns 1 state and NO actions
  // -> Use env["joint_robot"][0]["start"] repeated for N_opt
  // -> Actions: fill with 1.0 (all entries), length N_opt
  // ------------------------------------------------------------
  if (special_case_single_state_no_action) {
    if (!env["joint_robot"][0]["start"] || !env["joint_robot"][0]["start"].IsSequence()) {
      throw std::runtime_error("joint_robot[0].start missing or not a sequence in env.yaml");
    }

    auto startNode = env["joint_robot"][0]["start"];
    const int nx = (int)startNode.size();
    if (nx <= 0) throw std::runtime_error("joint_robot[0].start is empty");

    // finalStates: (N_opt+1) x nx
    finalStates.resize((int)N_opt + 1, nx);
    for (int j = 0; j < nx; ++j) {
      finalStates(0, j) = startNode[j].as<double>();
    }
    for (int t = 1; t < (int)N_opt + 1; ++t) {
      finalStates.row(t) = finalStates.row(0);
    }

    // actions: assume 4 controls per quad -> joint nu = 4*numRobots
    const int nu_joint = (int)(4 * numRobots);
    concatenatedActions = Eigen::MatrixXd::Constant((int)N_opt, nu_joint, 1.0);

    std::cout << "[SPECIAL CASE] Using joint_robot.start repeated for "
              << (N_opt + 1) << " states and actions filled with 1.0 for "
              << N_opt << " steps. nu_joint=" << nu_joint << std::endl;
  } else {
    // ---------------- NORMAL CASE ----------------

    // Pad
    std::vector<Eigen::MatrixXd> paddedRobotActions(numRobots);
    for (size_t i = 0; i < numRobots; ++i) {
      robotStates[i] = pad_mat(robotStates[i], maxRowsStates);
      paddedRobotActions[i] = pad_mat(robotActions[i], maxRowsActions);
    }

    // Concatenate actions
    if (maxRowsActions > 0) {
      concatenatedActions = Eigen::MatrixXd::Zero(
          maxRowsActions, (int)(numRobots * paddedRobotActions[0].cols()));
      for (size_t i = 0; i < numRobots; ++i) {
        concatenatedActions.block(0,
                                  (int)(i * paddedRobotActions[i].cols()),
                                  maxRowsActions,
                                  (int)paddedRobotActions[i].cols()) = paddedRobotActions[i];
      }
    } else {
      // This shouldn't happen if not special_case, but keep it safe.
      concatenatedActions.resize(0, 0);
    }

    // === Build new state layout ===
    if (startsWith(robotname, "mujocoquadspayload")) {
      YAML::Node payloadYaml = YAML::LoadFile(payloadPath);
      auto payloadInit = payloadYaml["payload"].as<std::vector<std::vector<double>>>();

      Eigen::MatrixXd payloadPosMatrix((int)payloadInit.size(), (int)payloadInit[0].size());
      for (size_t i = 0; i < payloadInit.size(); ++i) {
        for (size_t j = 0; j < payloadInit[i].size(); ++j) {
          payloadPosMatrix((int)i, (int)j) = payloadInit[i][j];
        }
      }

      finalStates.resize(
          maxRowsStates,
          3 + 4 +                     // payload pos + quat
              (int)numRobots * (3 + 4) +   // quad pos + quat
              3 + 3 +                      // payload vel + ang vel
              (int)numRobots * 6           // quad linear+angular vel
      );

      for (int t = 0; t < maxRowsStates; ++t) {
        int idx = 0;

        // payload position
        Eigen::Vector3d payload_pos = payloadPosMatrix.row(t).head<3>();
        finalStates.row(t).segment(idx, 3) = payload_pos;
        idx += 3;

        // payload quaternion = identity
        finalStates.row(t).segment(idx, 4) << 0, 0, 0, 1;
        idx += 4;

        // quads positions + quaternions
        for (size_t r = 0; r < numRobots; ++r) {
          finalStates.row(t).segment(idx, 3) = robotStates[r].row(t).head<3>();
          idx += 3;
          finalStates.row(t).segment(idx, 4) << 0, 0, 0, 1;
          idx += 4;
        }

        // payload velocities = zero
        finalStates.row(t).segment(idx, 3) = Eigen::Vector3d::Zero();
        idx += 3;
        finalStates.row(t).segment(idx, 3) = Eigen::Vector3d::Zero();
        idx += 3;

        // quad velocities: lin vel (cols 7-9), ang vel (cols 10-12)
        for (size_t r = 0; r < numRobots; ++r) {
          finalStates.row(t).segment(idx, 3) = robotStates[r].row(t).segment<3>(7);
          idx += 3;
          finalStates.row(t).segment(idx, 3) = robotStates[r].row(t).segment<3>(10);
          idx += 3;
        }
      }

      std::cout << "finished writing the initial guess (normal payload case)" << std::endl;
    } else if (startsWith(robotname, "mujocoquad")) {
      finalStates = robotStates[0];
    } else {
      throw std::runtime_error("Unsupported robot type: " + robotname);
    }
  }

  // === env.yaml + env_traj_checker.yaml: keep same behavior ===
  YAML::Node robotsNode = env["joint_robot"];
  YAML::Node robotsNode_traj_checker = YAML::Clone(env["joint_robot"]);

  YAML::Node outputEnvYaml = YAML::Clone(env);
  outputEnvYaml.remove("joint_robot");
  outputEnvYaml["robots"] = robotsNode;

  YAML::Node outputEnvYaml_traj_checker = YAML::Clone(env);
  outputEnvYaml_traj_checker.remove("joint_robot");
  outputEnvYaml_traj_checker["robots"] = robotsNode_traj_checker;
  std::string typeField = outputEnvYaml_traj_checker["robots"][0]["type"].as<std::string>();
  outputEnvYaml_traj_checker["robots"][0]["type"] =
      typeField.substr(0, typeField.find_last_of('.')) + "_traj_checker";

  std::string envOutputPath = resultPath.substr(0, resultPath.find_last_of("/\\") + 1) + "env.yaml";
  joint_robot_env_path = envOutputPath;
  std::ofstream envOutFile(envOutputPath);
  envOutFile << outputEnvYaml;

  std::string envOutputPath_traj_checker =
      resultPath.substr(0, resultPath.find_last_of("/\\") + 1) + "env_traj_checker.yaml";
  std::ofstream envOutFile_traj_checker(envOutputPath_traj_checker);
  envOutFile_traj_checker << outputEnvYaml_traj_checker;

  // === Save result.yaml ===
  YAML::Node result;
  YAML::Node statesNode, actionsNode;

  // Save states
  for (int i = 0; i < finalStates.rows(); ++i) {
    YAML::Node stateRow;
    for (int j = 0; j < finalStates.cols(); ++j) {
      stateRow.push_back(finalStates(i, j));
    }
    statesNode.push_back(stateRow);
  }

  // Save actions
  for (int i = 0; i < concatenatedActions.rows(); ++i) {
    YAML::Node actionRow;
    for (int j = 0; j < concatenatedActions.cols(); ++j) {
      actionRow.push_back(concatenatedActions(i, j));
    }
    actionsNode.push_back(actionRow);
  }

  result["result"]["states"] = statesNode;
  result["result"]["actions"] = actionsNode;
  result["result"]["num_action"] = (int)concatenatedActions.rows();
  result["result"]["num_states"] = (int)finalStates.rows();

  std::ofstream resultOutFile(resultPath);
  resultOutFile << result;
  resultOutFile.close();

  std::cout << "Init guess generation complete: " << resultPath << "\n";
}
