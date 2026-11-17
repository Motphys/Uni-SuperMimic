#pragma once

// Uni-SuperMimic deployment core: obs/goal construction and safety.
// Mirrors deploy/uni_g1/uni_deploy/core.py (and thereby the training MDP):
//   obs frame (99) = [q - q0, dq, body gyro, -R[2, :], prev_action,
//                     mat_to_6d(R^T @ R_ref_current)]
//   obs history    = 10 x 99, oldest -> newest, frame-major flatten (990)
//   goal           = clip.goal[min(frame, T-1)] (730)
//   target         = q0 + action_scale * a, PD-torque pre-clamped to effort

#include <array>
#include <cmath>

#include "uni_config.hpp"

namespace uni {

struct SafetyCfg {
    double tilt_abort = 0.8;
    double vel_abort_factor = 0.8;
    double action_abort = 50.0;
    bool clamp_torque = true;
    double torque_clamp_factor = 1.0;
};

class DeployCore {
public:
    DeployCore(const DeployConfig& cfg, const Clip* clip, SafetyCfg safety = SafetyCfg())
        : cfg_(cfg), clip_(clip), safety_(safety) {
        reset(0);
    }

    void reset(int frame);
    void switch_clip(const Clip* clip, int frame, bool keep_history);

    int frame() const { return frame_; }
    bool clip_done() const { return frame_ >= clip_->T - 1; }
    const Clip& clip() const { return *clip_; }

    // Feed one state sample; returns the 990-D frame-major history.
    const std::array<double, kHistory * kObsDim>& observation(const double* q, const double* dq,
                                                              const double* gyro_body,
                                                              const double* quat_wxyz);
    const float* goal() const { return clip_->goal_row(std::min(frame_, clip_->T - 1)); }

    // Finish one 50 Hz tick after the policy produced `action`.
    // Returns "" or an abort reason; fills q_target (PD-torque pre-clamped).
    std::string step_post(const double* action, const double* q, const double* dq,
                          const double* quat_wxyz, double* q_target);

    void targets(const double* action, const double* q, const double* dq, double* q_target) const;

private:
    std::array<double, kObsDim> obs_frame(const double* q, const double* dq,
                                          const double* gyro_body, const double* quat_wxyz) const;

    const DeployConfig& cfg_;
    const Clip* clip_;
    SafetyCfg safety_;
    int frame_ = 0;
    bool warmed_ = false;
    std::array<double, kHistory * kObsDim> hist_{};
    std::array<double, kNumJoints> last_action_{};
};

inline std::array<double, 4> yaw_quat(const double* q_wxyz) {
    Mat3 R = quat_to_mat(q_wxyz);
    double yaw = std::atan2(R[1][0], R[0][0]);
    return {std::cos(yaw / 2.0), 0.0, 0.0, std::sin(yaw / 2.0)};
}

}  // namespace uni
