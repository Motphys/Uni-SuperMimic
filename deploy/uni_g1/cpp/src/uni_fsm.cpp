#include "uni_fsm.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace uni {

const char* state_name(State s) {
    switch (s) {
        case State::PASSIVE: return "PASSIVE";
        case State::MOVE_TO_START: return "MOVE_TO_START";
        case State::STANDBY: return "STANDBY";
        case State::POLICY: return "POLICY";
        case State::DAMP: return "DAMP";
    }
    return "?";
}

static void cosine_blend(const double* from, const double* to, double t, double duration,
                         double* out) {
    double s = 0.5 * (1.0 - std::cos(M_PI * std::min(std::max(t / duration, 0.0), 1.0)));
    for (int j = 0; j < kNumJoints; ++j) out[j] = (1.0 - s) * from[j] + s * to[j];
}

void FSM::enter(State s, const double* q_now) {
    state_ = s;
    state_t_ = 0.0;
    for (int j = 0; j < kNumJoints; ++j) q_from_[j] = q_now[j];
    if (s == State::MOVE_TO_START) {
        // policy-driven settling on the stand reference (loop stays closed)
        core_.switch_clip(&standby_pre_, 0, false);
    } else if (s == State::STANDBY) {
        const Clip* c = ever_played_ ? &standby_post_ : &standby_pre_;
        core_.switch_clip(c, 0, ever_played_);
    } else if (s == State::POLICY) {
        core_.switch_clip(motion_clip_, 0, true);
        ever_played_ = true;
    }
}

bool FSM::request(const std::string& event, const double* q_now) {
    if (event == "damp_anytime") {
        enter(State::DAMP, q_now);
        return true;
    }
    if (event == "reset") {
        // recover from any state (incl. DAMP) to the just-started PASSIVE
        abort_reason_.clear();
        ever_played_ = false;
        core_.reset(0);
        enter(State::PASSIVE, q_now);
        return true;
    }
    if (state_ == State::DAMP) return false;  // DAMP only leaves via reset
    if (event == "to_start" && (state_ == State::PASSIVE || state_ == State::STANDBY)) {
        enter(State::MOVE_TO_START, q_now);
        return true;
    }
    if (event == "play" && state_ == State::STANDBY) {
        enter(State::POLICY, q_now);
        return true;
    }
    if (event == "to_passive" && state_ == State::STANDBY) {
        enter(State::PASSIVE, q_now);
        return true;
    }
    return false;
}

void FSM::abort(const std::string& reason, const double* q_now) {
    abort_reason_ = reason;
    enter(State::DAMP, q_now);
}

void FSM::reanchor_motion_clip(const double* quat_wxyz) {
    const Clip* original = clips_[size_t(clip_index_)];
    double q0[4];
    for (int i = 0; i < 4; ++i) q0[i] = original->quat_row(0)[i];
    double delta = yaw_of_quat(quat_wxyz) - yaw_of_quat(q0);
    while (delta > M_PI) delta -= 2.0 * M_PI;
    while (delta < -M_PI) delta += 2.0 * M_PI;
    motion_rebased_ = yaw_rebase_clip(*original, delta);
    motion_clip_ = &motion_rebased_;
    core_.switch_clip(motion_clip_, 0, true);
}

void FSM::reanchor_post_standby(const double* quat_wxyz) {
    standby_post_ = make_static_clip(cfg_, yaw_quat(quat_wxyz));
    if (state_ == State::STANDBY && ever_played_) core_.switch_clip(&standby_post_, 0, true);
}

void FSM::update(const double* q_now, const double* policy_q_target,
                 const std::string& policy_abort, double* q_target, double* kp, double* kd) {
    t_ += kControlDt;
    state_t_ += kControlDt;
    switch (state_) {
        case State::PASSIVE:
            for (int j = 0; j < kNumJoints; ++j) { q_target[j] = q_now[j]; kp[j] = 0.0; kd[j] = 0.0; }
            return;
        case State::DAMP:
            for (int j = 0; j < kNumJoints; ++j) { q_target[j] = q_now[j]; kp[j] = 0.0; kd[j] = kDampKd; }
            return;
        case State::MOVE_TO_START: {
            if (policy_q_target == nullptr) throw std::logic_error("MOVE_TO_START needs policy result");
            for (int j = 0; j < kNumJoints; ++j) {
                q_target[j] = policy_q_target[j];
                kp[j] = cfg_.kp[j];
                kd[j] = cfg_.kd[j];
            }
            if (!policy_abort.empty()) {
                abort(policy_abort, q_now);
            } else if (state_t_ >= kInterpS) {
                enter(State::STANDBY, q_now);
            }
            return;
        }
        case State::STANDBY:
        case State::POLICY: {
            if (policy_q_target == nullptr) throw std::logic_error("policy states need policy result");
            for (int j = 0; j < kNumJoints; ++j) {
                q_target[j] = policy_q_target[j];
                kp[j] = cfg_.kp[j];
                kd[j] = cfg_.kd[j];
            }
            if (!policy_abort.empty()) {
                abort(policy_abort, q_now);
            } else if (state_ == State::POLICY && core_.clip_done()) {
                enter(State::STANDBY, q_now);
            }
            return;
        }
    }
}

}  // namespace uni
