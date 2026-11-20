#include "uni_deploy_core.hpp"

#include <cmath>
#include <cstring>

namespace uni {

void DeployCore::reset(int frame) {
    frame_ = std::min(std::max(frame, 0), clip_->T - 1);
    hist_.fill(0.0);
    last_action_.fill(0.0);
    warmed_ = false;
}

void DeployCore::switch_clip(const Clip* clip, int frame, bool keep_history) {
    clip_ = clip;
    frame_ = std::min(std::max(frame, 0), clip->T - 1);
    if (!keep_history) {
        hist_.fill(0.0);
        last_action_.fill(0.0);
        warmed_ = false;
    }
}

std::array<double, kObsDim> DeployCore::obs_frame(const double* q, const double* dq,
                                                  const double* gyro_body,
                                                  const double* quat_wxyz) const {
    Mat3 R = quat_to_mat(quat_wxyz);
    int f = std::min(frame_, clip_->T - 1);
    double quat_ref[4];
    for (int i = 0; i < 4; ++i) quat_ref[i] = clip_->quat_row(f)[i];
    Mat3 R_ref = quat_to_mat(quat_ref);
    // rel6 = mat_to_6d(R^T @ R_ref)
    Mat3 rel{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) rel[i][j] += R[k][i] * R_ref[k][j];
    std::array<double, 6> rel6 = mat_to_6d(rel);

    std::array<double, kObsDim> o{};
    int idx = 0;
    for (int j = 0; j < kNumJoints; ++j) o[idx++] = q[j] - cfg_.default_joint_pos[j];
    for (int j = 0; j < kNumJoints; ++j) o[idx++] = dq[j];
    for (int j = 0; j < 3; ++j) o[idx++] = gyro_body[j];
    for (int j = 0; j < 3; ++j) o[idx++] = -R[2][j];  // training: -rotation[..., 2, :]
    for (int j = 0; j < kNumJoints; ++j) o[idx++] = last_action_[j];
    for (int j = 0; j < 6; ++j) o[idx++] = rel6[j];
    return o;
}

const std::array<double, kHistory * kObsDim>& DeployCore::observation(
    const double* q, const double* dq, const double* gyro_body, const double* quat_wxyz) {
    std::array<double, kObsDim> row = obs_frame(q, dq, gyro_body, quat_wxyz);
    if (!warmed_) {
        for (int h = 0; h < kHistory; ++h) std::memcpy(hist_.data() + h * kObsDim, row.data(),
                                                       kObsDim * sizeof(double));
        warmed_ = true;
    } else {
        std::memmove(hist_.data(), hist_.data() + kObsDim, (kHistory - 1) * kObsDim * sizeof(double));
        std::memcpy(hist_.data() + (kHistory - 1) * kObsDim, row.data(), kObsDim * sizeof(double));
    }
    return hist_;
}

void DeployCore::targets(const double* action, const double* q, const double* dq,
                         double* q_target) const {
    for (int j = 0; j < kNumJoints; ++j)
        q_target[j] = cfg_.default_joint_pos[j] + cfg_.action_scale[j] * action[j];
    if (safety_.clamp_torque) {
        for (int j = 0; j < kNumJoints; ++j) {
            double tau = safety_.torque_clamp_factor * cfg_.effort_limit[j];
            double lo = q[j] + (-tau + cfg_.kd[j] * dq[j]) / cfg_.kp[j];
            double hi = q[j] + (tau + cfg_.kd[j] * dq[j]) / cfg_.kp[j];
            q_target[j] = std::min(std::max(q_target[j], lo), hi);
        }
    }
}

std::string DeployCore::step_post(const double* action, const double* q, const double* dq,
                                  const double* quat_wxyz, double* q_target) {
    std::string abort;
    bool finite = true;
    for (int j = 0; j < kNumJoints; ++j)
        finite = finite && std::isfinite(q[j]) && std::isfinite(dq[j]) && std::isfinite(action[j]);
    if (!finite) abort = "non_finite";
    if (abort.empty()) {
        int f = std::min(frame_, clip_->T - 1);
        Mat3 R = quat_to_mat(quat_wxyz);
        double quat_ref[4];
        for (int i = 0; i < 4; ++i) quat_ref[i] = clip_->quat_row(f)[i];
        Mat3 R_ref = quat_to_mat(quat_ref);
        if (std::fabs(R_ref[2][2] - R[2][2]) > safety_.tilt_abort) abort = "tilt";
    }
    if (abort.empty()) {
        for (int j = 0; j < kNumJoints; ++j)
            if (std::fabs(dq[j]) > safety_.vel_abort_factor * cfg_.velocity_limit[j]) {
                abort = "joint_velocity";
                break;
            }
    }
    if (abort.empty()) {
        double amax = 0.0;
        for (int j = 0; j < kNumJoints; ++j) amax = std::max(amax, std::fabs(action[j]));
        if (amax > safety_.action_abort) abort = "action_magnitude";
    }
    targets(action, q, dq, q_target);
    for (int j = 0; j < kNumJoints; ++j) last_action_[j] = action[j];
    frame_ += 1;
    return abort;
}

}  // namespace uni
