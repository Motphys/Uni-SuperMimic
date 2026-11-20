#pragma once

// Uni-SuperMimic G1 deployment config + clip loading (contract unisupermimic_v1).
//
// Assets are plain files produced by tools/pack_assets.py:
//   uni_config.txt       flat key=value, vectors comma-separated
//   clips_bin/<name>.bin int32 T + goal (T*730) + joint_pos (T*29)
//                        + root_quat_wxyz (T*4) + root_pos (T*3), little-endian f32

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace uni {

constexpr int kNumJoints = 29;
constexpr int kObsDim = 99;
constexpr int kHistory = 10;
constexpr int kGoalFrameDim = 73;
constexpr int kFuture = 10;
constexpr float kControlDt = 0.02f;

using Vec = std::vector<float>;
using Mat3 = std::array<std::array<double, 3>, 3>;

struct DeployConfig {
    std::string contract;
    std::vector<std::string> joint_names;
    Vec default_joint_pos;
    Vec action_scale;
    Vec kp;
    Vec kd;
    Vec effort_limit;
    Vec velocity_limit;
    Vec joint_lower;
    Vec joint_upper;
    float soft_limit_factor = 0.9f;
    float control_dt = kControlDt;
};

struct Clip {
    std::string name;
    int32_t T = 0;
    Vec goal;          // T * 730
    Vec joint_pos;     // T * 29
    Vec root_quat;     // T * 4 (wxyz)
    Vec root_pos;      // T * 3

    const float* goal_row(int f) const { return goal.data() + size_t(f) * (kGoalFrameDim * kFuture); }
    const float* joint_row(int f) const { return joint_pos.data() + size_t(f) * kNumJoints; }
    const float* quat_row(int f) const { return root_quat.data() + size_t(f) * 4; }
};

DeployConfig load_config(const std::string& path);
Clip load_clip_bin(const std::string& path, const std::string& name);

// Frozen references for the self-balancing STANDBY states, mirroring
// tools/pack_assets.py / uni_deploy/contract.py semantics.
Clip make_static_clip(const DeployConfig& cfg, const std::array<double, 4>& root_quat_wxyz);
Clip make_frozen_clip(const DeployConfig& cfg, const Clip& clip, int frame);

Mat3 quat_to_mat(const double* q_wxyz);
Mat3 yaw_mat(const Mat3& R);
std::array<double, 6> mat_to_6d(const Mat3& R);

}  // namespace uni

namespace uni {

// Yaw helpers and trajectory rebasing (see uni_deploy/contract.py).
double yaw_of_quat(const double* q_wxyz);
std::array<double, 4> mat_to_quat(const Mat3& R);
// Rotate the clip root trajectory by delta_yaw about z at frame-0 position.
// The goal table is yaw-invariant and is carried over unchanged.
Clip yaw_rebase_clip(const Clip& clip, double delta_yaw);

}  // namespace uni
