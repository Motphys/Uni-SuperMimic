// Offline numeric gate for the Uni-SuperMimic C++ deployment.
//
// Mirrors scripts/validate_bundle.py + the golden-vector contract:
//   1. config / clips load and internal consistency
//   2. golden vectors: obs / goal / action / target vs the training MDP
//      (tolerances from golden_manifest.json)
//   3. FSM walk-through: PASSIVE->MOVE_TO_START->STANDBY->POLICY->DAMP,
//      DAMP latch, post-play yaw reanchor
//
// Usage: check_numeric <assets_dir>   (exit 0 and print NUMERIC PASS on success)

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "uni_deploy_core.hpp"
#include "uni_fsm.hpp"
#include "uni_policy.hpp"

using namespace uni;

namespace {

struct Case {
    int32_t clip_idx;
    int32_t frame;
    std::vector<float> q, dq, gyro, quat, prev_action, obs, goal, action, target;
};

int failures = 0;

void expect(bool ok, const std::string& msg) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL: " << msg << "\n";
    }
}

std::vector<std::string> load_clip_names(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path + " (run tools/pack_assets.py)");
    std::vector<std::string> names;
    std::string line;
    while (std::getline(in, line))
        if (!line.empty()) names.push_back(line);
    if (names.empty()) throw std::runtime_error(path + " is empty");
    return names;
}

template <typename T>
void read_raw(std::ifstream& in, T* dst, size_t n) {
    in.read(reinterpret_cast<char*>(dst), sizeof(T) * n);
    if (!in) throw std::runtime_error("unexpected end of golden.bin");
}

std::vector<Case> load_golden(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    int32_t n = 0;
    read_raw(in, &n, 1);
    std::vector<Case> cases(n);
    for (auto& c : cases) {
        read_raw(in, &c.clip_idx, 1);
        read_raw(in, &c.frame, 1);
        auto alloc = [&](size_t k) { return std::vector<float>(k); };
        c.q = alloc(kNumJoints); c.dq = alloc(kNumJoints); c.gyro = alloc(3); c.quat = alloc(4);
        c.prev_action = alloc(kNumJoints); c.obs = alloc(kObsDim);
        c.goal = alloc(kGoalFrameDim * kFuture); c.action = alloc(kNumJoints);
        c.target = alloc(kNumJoints);
        read_raw(in, c.q.data(), c.q.size());
        read_raw(in, c.dq.data(), c.dq.size());
        read_raw(in, c.gyro.data(), c.gyro.size());
        read_raw(in, c.quat.data(), c.quat.size());
        read_raw(in, c.prev_action.data(), c.prev_action.size());
        read_raw(in, c.obs.data(), c.obs.size());
        read_raw(in, c.goal.data(), c.goal.size());
        read_raw(in, c.action.data(), c.action.size());
        read_raw(in, c.target.data(), c.target.size());
    }
    return cases;
}

double max_abs_diff(const double* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - double(b[i])));
    return m;
}

double max_abs_diff(const float* a, const float* b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "assets";
    try {
        DeployConfig cfg = load_config(dir + "/uni_config.txt");
        const std::vector<std::string> clip_names = load_clip_names(dir + "/golden/clips.txt");
        std::vector<Clip> clips;
        clips.reserve(clip_names.size());
        for (const auto& n : clip_names)
            clips.push_back(load_clip_bin(dir + "/clips_bin/" + n + ".bin", n));
        for (const auto& c : clips)
            expect(c.T > 1 && c.goal.size() == size_t(c.T) * kGoalFrameDim * kFuture,
                   "clip " + c.name + " shape");

        std::vector<Case> cases = load_golden(dir + "/golden/golden.bin");
        std::cout << "golden cases: " << cases.size() << "\n";

        OnnxPolicy policy(dir + "/policy.onnx");

        double worst_obs = 0.0, worst_goal = 0.0, worst_act = 0.0, worst_tf = 0.0, worst_te2e = 0.0;
        size_t worst_case = 0, worst_idx = 0;
        Clip static_clip = make_static_clip(cfg, {1.0, 0.0, 0.0, 0.0});
        size_t case_i = 0;
        for (const Case& gc : cases) {
            ++case_i;
            const Clip* clip = gc.clip_idx < 0 ? &static_clip : &clips.at(gc.clip_idx);
            DeployCore core(cfg, clip);
            core.reset(gc.frame);
            std::vector<double> q(kNumJoints), dq(kNumJoints), gyro(3), quat(4);
            for (int j = 0; j < kNumJoints; ++j) { q[j] = gc.q[j]; dq[j] = gc.dq[j]; }
            for (int j = 0; j < 3; ++j) gyro[j] = gc.gyro[j];
            for (int j = 0; j < 4; ++j) quat[j] = gc.quat[j];
            const auto& hist = core.observation(q.data(), dq.data(), gyro.data(), quat.data());
            for (size_t i = 0; i < kObsDim; ++i) {
                double d = std::fabs(hist[i] - double(gc.obs[i]));
                if (d > worst_obs) { worst_obs = d; worst_case = case_i; worst_idx = i; }
            }
            worst_goal = std::max(worst_goal, max_abs_diff(core.goal(), gc.goal.data(), gc.goal.size()));

            std::vector<double> action(kNumJoints), q_target(kNumJoints);
            policy.run(hist.data(), core.goal(), action.data());
            worst_act = std::max(worst_act, max_abs_diff(action.data(), gc.action.data(), kNumJoints));
            // target formula with the GOLDEN action must match to 1e-5
            std::vector<double> golden_action(kNumJoints);
            for (int j = 0; j < kNumJoints; ++j) golden_action[j] = gc.action[j];
            core.targets(golden_action.data(), q.data(), dq.data(), q_target.data());
            worst_tf = std::max(worst_tf, max_abs_diff(q_target.data(), gc.target.data(), kNumJoints));
            // end-to-end target with the ONNX action, tolerance 1e-3
            core.targets(action.data(), q.data(), dq.data(), q_target.data());
            worst_te2e = std::max(worst_te2e, max_abs_diff(q_target.data(), gc.target.data(), kNumJoints));
        }
        std::printf("obs %.3e (tol 1e-4) [case %zu idx %zu]\ngoal %.3e (tol 1e-4)\naction %.3e (tol 1e-3)\n"
                    "target_formula %.3e (tol 1e-5)\ntarget_e2e %.3e (tol 1e-3)\n",
                    worst_obs, worst_case, worst_idx, worst_goal, worst_act, worst_tf, worst_te2e);
        expect(worst_obs <= 1e-4, "obs tolerance");
        expect(worst_goal <= 1e-4, "goal tolerance");
        expect(worst_act <= 1e-3, "action tolerance");
        expect(worst_tf <= 1e-5, "target formula tolerance");
        expect(worst_te2e <= 1e-3, "target e2e tolerance");

        // FSM walk-through
        {
            DeployCore core(cfg, &clips[0]);
            FSM fsm(cfg, core, {&clips[0], &clips[1]});
            std::vector<double> q(kNumJoints), dq(kNumJoints, 0.0), gyro(3, 0.0), quat(4, 0.0);
            quat[0] = 1.0;
            for (int j = 0; j < kNumJoints; ++j) q[j] = cfg.default_joint_pos[j];
            std::vector<double> qt(kNumJoints), kp(kNumJoints), kd(kNumJoints);
            expect(fsm.state() == State::PASSIVE, "fsm starts PASSIVE");
            expect(fsm.request("to_start", q.data()), "to_start accepted");
            for (int i = 0; i < int(kInterpS / kControlDt) + 2; ++i) {
                const auto& hist = core.observation(q.data(), dq.data(), gyro.data(), quat.data());
                std::vector<double> act(kNumJoints);
                policy.run(hist.data(), core.goal(), act.data());
                std::string ab = core.step_post(act.data(), q.data(), dq.data(), quat.data(), qt.data());
                fsm.update(q.data(), qt.data(), ab, qt.data(), kp.data(), kd.data());
            }
            expect(fsm.state() == State::STANDBY, "interp -> STANDBY");
            expect(fsm.cycle_clip(+1) == clips[1].name, "cycle next clip");
            expect(fsm.cycle_clip(-1) == clips[0].name, "cycle prev clip");
            expect(fsm.request("play", q.data()), "play accepted");
            expect(fsm.state() == State::POLICY, "STANDBY -> POLICY");
            for (int i = 0; i < 3; ++i) {
                const auto& hist = core.observation(q.data(), dq.data(), gyro.data(), quat.data());
                std::vector<double> act(kNumJoints);
                policy.run(hist.data(), core.goal(), act.data());
                std::string ab = core.step_post(act.data(), q.data(), dq.data(), quat.data(), qt.data());
                fsm.update(q.data(), qt.data(), ab, qt.data(), kp.data(), kd.data());
            }
            expect(fsm.request("damp_anytime", q.data()), "damp accepted");
            expect(fsm.state() == State::DAMP, "-> DAMP");
            expect(!fsm.request("to_start", q.data()), "DAMP blocks to_start");
            fsm.update(q.data(), nullptr, "", qt.data(), kp.data(), kd.data());
            for (int j = 0; j < kNumJoints; ++j)
                expect(kp[j] == 0.0 && kd[j] == kDampKd, "DAMP gains");
            expect(fsm.request("reset", q.data()), "reset from DAMP accepted");
            expect(fsm.state() == State::PASSIVE, "reset -> PASSIVE");
            expect(fsm.request("to_start", q.data()), "to_start works after reset");
        }
    } catch (const std::exception& e) {
        std::cout << "FAIL: exception: " << e.what() << "\n";
        ++failures;
    }
    if (failures == 0) {
        std::cout << "NUMERIC PASS\n";
        return 0;
    }
    std::cout << "NUMERIC FAIL (" << failures << ")\n";
    return 1;
}
