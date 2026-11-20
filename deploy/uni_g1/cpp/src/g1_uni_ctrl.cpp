// Uni-SuperMimic G1 29-DoF real-robot controller.
//
// *** Started manually by the operator. ***
// Flow: PASSIVE --L2+Up--> MOVE_TO_START --auto--> STANDBY --R1+X--> POLICY
//       POLICY --clip end--auto--> STANDBY (yaw-reanchored stand)
//       any state --L2+B--> DAMP (kp=0, kd=3, terminal)
// Loop: 500 Hz lowcmd writer; 50 Hz FSM/policy, same DeployCore/FSM code that
// passed check_numeric locally.

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/dds_wrapper/common/crc.h>

#include "uni_deploy_core.hpp"
#include "uni_fsm.hpp"
#include "uni_policy.hpp"

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

namespace {

constexpr int kNumMotors = 29;
constexpr double kCmdDt = 0.002;  // 500 Hz lowcmd

union KeySwitch {
    struct {
        uint8_t R1 : 1, L1 : 1, start : 1, select : 1, R2 : 1, L2 : 1, F1 : 1, F2 : 1;
        uint8_t A : 1, B : 1, X : 1, Y : 1, up : 1, right : 1, down : 1, left : 1;
    } components;
    uint16_t value = 0;
};

std::atomic<bool> g_running{true};
void on_sigint(int) { g_running = false; }

struct SharedCmd {
    std::array<double, kNumMotors> q{};
    std::array<double, kNumMotors> kp{};
    std::array<double, kNumMotors> kd{};
    std::mutex mtx;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "usage: g1_uni_ctrl <network_if> <assets_dir> [clip_name]\n"
                  << "  e.g. g1_uni_ctrl eth0 ../assets cellphone_record_stand_idle_R_001__A423\n"
                  << "Buttons: L2+Up start | R1+X play | L2+Left/Right switch clip | L2+B DAMP\n";
        return 2;
    }
    const std::string network_if = argv[1];
    const std::string assets = argv[2];
    const std::string clip_name = argc > 3 ? argv[3] : "";

    uni::DeployConfig cfg = uni::load_config(assets + "/uni_config.txt");
    // Arm every clip found in clips_bin/ (sorted); no hardcoded clip list.
    std::vector<uni::Clip> clips;
    std::vector<const uni::Clip*> clip_ptrs;
    try {
        namespace fs = std::filesystem;
        std::vector<fs::path> bins;
        for (const auto& e : fs::directory_iterator(assets + "/clips_bin"))
            if (e.path().extension() == ".bin") bins.push_back(e.path());
        std::sort(bins.begin(), bins.end());
        if (bins.empty()) throw std::runtime_error("no clips under " + assets + "/clips_bin");
        clips.reserve(bins.size());
        clip_ptrs.reserve(bins.size());
        for (const auto& b : bins) clips.push_back(uni::load_clip_bin(b.string(), b.stem().string()));
        for (const auto& c : clips) clip_ptrs.push_back(&c);
    } catch (const std::exception& e) {
        std::cerr << "[g1_uni_ctrl] asset load failed: " << e.what() << std::endl;
        return 1;
    }
    uni::OnnxPolicy policy(assets + "/policy.onnx");
    uni::DeployCore core(cfg, clip_ptrs.front());
    uni::FSM fsm(cfg, core, clip_ptrs);
    if (clip_name.empty()) {
        // safe default: the standing clip when present, otherwise the first clip
        fsm.select_clip("cellphone_record_stand_idle_R_001__A423");
    } else if (!fsm.select_clip(clip_name)) {
        std::cerr << "[g1_uni_ctrl] unknown clip " << clip_name << std::endl;
        return 1;
    }

    ChannelFactory::Instance()->Init(0, network_if);
    LowState_ low_state{};
    std::mutex state_mtx;
    bool have_state = false;

    ChannelSubscriber<LowState_> state_sub("rt/lowstate");
    state_sub.InitChannel([&](const void* msg) {
        const LowState_& s = *static_cast<const LowState_*>(msg);
        LowState_ s_copy = s;  // crc32_core takes a non-const pointer (read-only in practice)
        if (s.crc() != crc32_core(reinterpret_cast<uint32_t*>(&s_copy),
                                  (sizeof(LowState_) >> 2) - 1)) {
            return;  // drop corrupted frame
        }
        std::lock_guard<std::mutex> lk(state_mtx);
        low_state = s;
        have_state = true;
    });
    ChannelPublisher<LowCmd_> cmd_pub("rt/lowcmd");
    cmd_pub.InitChannel();

    std::cout << "[g1_uni_ctrl] waiting for rt/lowstate ..." << std::endl;
    {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!have_state) {
            if (std::chrono::steady_clock::now() > deadline) {
                std::cerr << "[g1_uni_ctrl] no rt/lowstate within 5 s; abort" << std::endl;
                return 1;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    if (low_state.mode_machine() != 5) {
        std::cerr << "[g1_uni_ctrl] mode_machine=" << int(low_state.mode_machine())
                  << " != 5 (development mode); abort" << std::endl;
        return 1;
    }
    std::cout << "[g1_uni_ctrl] connected, mode_machine=5. clip=" << fsm.motion_clip_name()
              << ". PASSIVE." << std::endl;

    SharedCmd shared;
    {
        std::lock_guard<std::mutex> lk(shared.mtx);
        for (int i = 0; i < kNumMotors; ++i) {
            shared.q[i] = low_state.motor_state()[i].q();
            shared.kp[i] = 0.0;
            shared.kd[i] = 0.0;
        }
    }

    // 500 Hz lowcmd writer
    std::thread writer([&] {
        LowCmd_ cmd{};
        while (g_running) {
            {
                std::lock_guard<std::mutex> lk(shared.mtx);
                cmd.mode_pr() = 0;  // Mode::PR
                cmd.mode_machine() = low_state.mode_machine();
                for (int i = 0; i < kNumMotors; ++i) {
                    auto& mc = cmd.motor_cmd().at(i);
                    mc.mode() = 1;
                    mc.q() = float(shared.q[i]);
                    mc.dq() = 0.0f;
                    mc.tau() = 0.0f;
                    mc.kp() = float(shared.kp[i]);
                    mc.kd() = float(shared.kd[i]);
                }
                cmd.crc() = crc32_core(reinterpret_cast<uint32_t*>(&cmd),
                                      (sizeof(LowCmd_) >> 2) - 1);
            }
            cmd_pub.Write(cmd);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    std::signal(SIGINT, on_sigint);

    // 50 Hz FSM/policy loop
    std::array<double, kNumMotors> q{}, dq{}, q_target{}, kp{}, kd{};
    std::array<double, 3> gyro{};
    std::array<double, 4> quat{};
    int damp_votes = 0, up_votes = 0, x_votes = 0, down_votes = 0, next_votes = 0, prev_votes = 0, reset_votes = 0;
    bool running = true;
    while (g_running && running) {
        auto tick = std::chrono::steady_clock::now();
        uint16_t keys;
        {
            std::lock_guard<std::mutex> lk(state_mtx);
            for (int i = 0; i < kNumMotors; ++i) {
                q[i] = low_state.motor_state()[i].q();
                dq[i] = low_state.motor_state()[i].dq();
            }
            const auto& imu = low_state.imu_state();
            for (int i = 0; i < 3; ++i) gyro[i] = imu.gyroscope()[i];
            for (int i = 0; i < 4; ++i) quat[i] = imu.quaternion()[i];
            KeySwitch ks;
            std::memcpy(&ks.value, low_state.wireless_remote().data() + 2, 2);
            keys = ks.value;
        }
        KeySwitch ks;
        ks.value = keys;
        auto vote = [](bool pressed, int& counter) {
            counter = pressed ? counter + 1 : 0;
            if (pressed && counter == 3) return true;
            return false;
        };
        const uni::State state_before_events = fsm.state();
        if (vote(ks.components.L2 && ks.components.B, damp_votes)) fsm.request("damp_anytime", q.data());
        if (vote(ks.components.R1 && ks.components.A, reset_votes)) {
            fsm.request("reset", q.data());
            std::cout << "[g1_uni_ctrl] reset -> PASSIVE (initial state)" << std::endl;
        }
        if (vote(ks.components.L2 && ks.components.up, up_votes)) {
            if (fsm.request("to_start", q.data())) fsm.set_start_heading(quat.data());
        }
        if (vote(ks.components.R1 && ks.components.X, x_votes)) fsm.request("play", q.data());
        if (vote(ks.components.L2 && ks.components.down, down_votes)) fsm.request("to_passive", q.data());
        if (vote(ks.components.L2 && ks.components.right, next_votes))
            std::cout << "[g1_uni_ctrl] clip -> " << fsm.cycle_clip(+1) << std::endl;
        if (vote(ks.components.L2 && ks.components.left, prev_votes))
            std::cout << "[g1_uni_ctrl] clip -> " << fsm.cycle_clip(-1) << std::endl;
        if (state_before_events == uni::State::STANDBY && fsm.state() == uni::State::POLICY)
            fsm.reanchor_motion_clip(quat.data());

        const uni::State before = fsm.state();
        const double* policy_qt = nullptr;
        std::string policy_abort;
        std::array<double, kNumMotors> policy_target{};
        if (before == uni::State::MOVE_TO_START || before == uni::State::STANDBY ||
            before == uni::State::POLICY) {
            const auto& hist = core.observation(q.data(), dq.data(), gyro.data(), quat.data());
            std::array<double, kNumMotors> action{};
            policy.run(hist.data(), core.goal(), action.data());
            policy_abort = core.step_post(action.data(), q.data(), dq.data(), quat.data(),
                                          policy_target.data());
            policy_qt = policy_target.data();
        }
        fsm.update(q.data(), policy_qt, policy_abort, q_target.data(), kp.data(), kd.data());
        if (before == uni::State::POLICY && fsm.state() == uni::State::STANDBY)
            fsm.reanchor_post_standby(quat.data());
        {
            std::lock_guard<std::mutex> lk(shared.mtx);
            shared.q = q_target;
            shared.kp = kp;
            shared.kd = kd;
        }
        static int log_decim = 0;
        if (++log_decim >= 50) {
            log_decim = 0;
            std::cout << "[g1_uni_ctrl] state=" << uni::state_name(fsm.state())
                      << " frame=" << core.frame()
                      << (fsm.abort_reason().empty() ? "" : (" abort=" + fsm.abort_reason()))
                      << std::endl;
        }
        std::this_thread::sleep_until(tick + std::chrono::milliseconds(20));
    }

    // exit path: hold DAMP briefly so the robot relaxes before we stop publishing
    std::cerr << "[g1_uni_ctrl] exiting: DAMP for 1 s" << std::endl;
    {
        std::lock_guard<std::mutex> lk(shared.mtx);
        for (int i = 0; i < kNumMotors; ++i) { shared.kp[i] = 0.0; shared.kd[i] = uni::kDampKd; }
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    g_running = false;
    if (writer.joinable()) writer.join();
    return 0;
}
