#pragma once

// Deployment state machine with always-available damping (mirrors
// deploy/uni_g1/uni_deploy/fsm.py).
//
//   PASSIVE       zero torque, startup
//   MOVE_TO_START cosine interpolation to the motion clip frame-0 pose
//   STANDBY       self-balancing: policy tracks a frozen reference
//                 (pre-play: clip frame 0; post-play: yaw-anchored stand)
//   POLICY        self-balancing: policy tracks the motion clip
//   DAMP          kp=0, kd=DAMP_KD; terminal until process restart
//
//   PASSIVE/STANDBY --to_start--> MOVE_TO_START --auto--> STANDBY
//   STANDBY --play--> POLICY --clip end--auto--> STANDBY
//   STANDBY --to_passive--> PASSIVE ;  any state --damp_anytime--> DAMP

#include <string>

#include "uni_deploy_core.hpp"

namespace uni {

constexpr double kDampKd = 3.0;
constexpr double kInterpS = 4.0;

enum class State { PASSIVE, MOVE_TO_START, STANDBY, POLICY, DAMP };

const char* state_name(State s);

class FSM {
public:
    FSM(const DeployConfig& cfg, DeployCore& core, std::vector<const Clip*> clips)
        : cfg_(cfg), core_(core), clips_(std::move(clips)),
          motion_clip_(clips_.front()),
          standby_pre_(make_static_clip(cfg, {1.0, 0.0, 0.0, 0.0})),
          standby_post_(make_static_clip(cfg, {1.0, 0.0, 0.0, 0.0})) {}

    // Switch the armed motion clip (allowed in PASSIVE/STANDBY).
    const std::string& cycle_clip(int delta) {
        if (state_ == State::PASSIVE || state_ == State::STANDBY) {
            int n = int(clips_.size());
            clip_index_ = ((clip_index_ + delta) % n + n) % n;
            motion_clip_ = clips_[size_t(clip_index_)];
        }
        return motion_clip_->name;
    }

    const std::string& motion_clip_name() const { return motion_clip_->name; }

    bool select_clip(const std::string& name) {
        for (size_t i = 0; i < clips_.size(); ++i)
            if (clips_[i]->name == name) {
                clip_index_ = int(i);
                motion_clip_ = clips_[i];
                return true;
            }
        return false;
    }

    State state() const { return state_; }
    const std::string& abort_reason() const { return abort_reason_; }
    bool ever_played() const { return ever_played_; }

    bool request(const std::string& event, const double* q_now);
    void abort(const std::string& reason, const double* q_now);
    void reanchor_post_standby(const double* quat_wxyz);
    void reanchor_motion_clip(const double* quat_wxyz);
    // Anchor the pre-play stand reference to the robot's current yaw
    // (call right after a successful to_start request).
    void set_start_heading(const double* quat_wxyz) {
        standby_pre_ = make_static_clip(cfg_, yaw_quat(quat_wxyz));
        if (state_ == State::MOVE_TO_START) core_.switch_clip(&standby_pre_, 0, true);
    }

    // policy_result: (q_target, abort) from DeployCore.step_post, required in
    // STANDBY/POLICY. Returns via out params; state transitions applied.
    void update(const double* q_now, const double* policy_q_target, const std::string& policy_abort,
                double* q_target, double* kp, double* kd);

private:
    void enter(State s, const double* q_now);

    const DeployConfig& cfg_;
    DeployCore& core_;
    std::vector<const Clip*> clips_;
    int clip_index_ = 0;
    const Clip* motion_clip_;
    Clip motion_rebased_;
    Clip standby_pre_;
    Clip standby_post_;
    State state_ = State::PASSIVE;
    double t_ = 0.0;
    double state_t_ = 0.0;
    std::array<double, kNumJoints> q_from_{};
    std::string abort_reason_;
    bool ever_played_ = false;
};

}  // namespace uni
