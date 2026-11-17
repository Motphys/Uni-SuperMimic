#include "uni_config.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace uni {

static Vec parse_vec(const std::string& s) {
    Vec out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::stof(item));
    return out;
}

DeployConfig load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    DeployConfig cfg;
    std::string line;
    auto need = [&](const Vec& v, const char* key) {
        if (v.size() != size_t(kNumJoints))
            throw std::runtime_error(std::string("uni_config ") + key + " must have 29 values");
    };
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string key;
        ls >> key;
        std::string rest;
        std::getline(ls >> std::ws, rest);
        if (key == "contract") {
            cfg.contract = rest;
            if (rest != "unisupermimic_v1") throw std::runtime_error("bad contract " + rest);
        } else if (key == "joint_names") {
            std::stringstream ss(rest);
            std::string item;
            while (std::getline(ss, item, ',')) cfg.joint_names.push_back(item);
        } else if (key == "default_joint_pos") cfg.default_joint_pos = parse_vec(rest);
        else if (key == "action_scale") cfg.action_scale = parse_vec(rest);
        else if (key == "kp") cfg.kp = parse_vec(rest);
        else if (key == "kd") cfg.kd = parse_vec(rest);
        else if (key == "effort_limit") cfg.effort_limit = parse_vec(rest);
        else if (key == "velocity_limit") cfg.velocity_limit = parse_vec(rest);
        else if (key == "joint_lower") cfg.joint_lower = parse_vec(rest);
        else if (key == "joint_upper") cfg.joint_upper = parse_vec(rest);
        else if (key == "soft_limit_factor") cfg.soft_limit_factor = std::stof(rest);
        else if (key == "control_dt") cfg.control_dt = std::stof(rest);
    }
    need(cfg.default_joint_pos, "default_joint_pos");
    need(cfg.action_scale, "action_scale");
    need(cfg.kp, "kp");
    need(cfg.kd, "kd");
    need(cfg.effort_limit, "effort_limit");
    need(cfg.velocity_limit, "velocity_limit");
    need(cfg.joint_lower, "joint_lower");
    need(cfg.joint_upper, "joint_upper");
    if (cfg.joint_names.size() != size_t(kNumJoints)) throw std::runtime_error("need 29 joint names");
    if (std::fabs(cfg.control_dt - kControlDt) > 1e-9f) throw std::runtime_error("control_dt must be 0.02");
    return cfg;
}

template <typename T>
static void read_raw(std::ifstream& in, T* dst, size_t n) {
    in.read(reinterpret_cast<char*>(dst), sizeof(T) * n);
    if (!in) throw std::runtime_error("unexpected end of clip bin");
}

Clip load_clip_bin(const std::string& path, const std::string& name) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    Clip c;
    c.name = name;
    read_raw(in, &c.T, 1);
    if (c.T <= 0 || c.T > 100000) throw std::runtime_error("bad clip length");
    c.goal.resize(size_t(c.T) * kGoalFrameDim * kFuture);
    c.joint_pos.resize(size_t(c.T) * kNumJoints);
    c.root_quat.resize(size_t(c.T) * 4);
    c.root_pos.resize(size_t(c.T) * 3);
    read_raw(in, c.goal.data(), c.goal.size());
    read_raw(in, c.joint_pos.data(), c.joint_pos.size());
    read_raw(in, c.root_quat.data(), c.root_quat.size());
    read_raw(in, c.root_pos.data(), c.root_pos.size());
    return c;
}

Mat3 quat_to_mat(const double* q) {
    double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (!std::isfinite(n) || n < 0.5) throw std::runtime_error("invalid quaternion");
    double w = q[0] / n, x = q[1] / n, y = q[2] / n, z = q[3] / n;
    Mat3 R;
    R[0][0] = 1 - 2 * (y * y + z * z); R[0][1] = 2 * (x * y - w * z); R[0][2] = 2 * (x * z + w * y);
    R[1][0] = 2 * (x * y + w * z); R[1][1] = 1 - 2 * (x * x + z * z); R[1][2] = 2 * (y * z - w * x);
    R[2][0] = 2 * (x * z - w * y); R[2][1] = 2 * (y * z + w * x); R[2][2] = 1 - 2 * (x * x + y * y);
    return R;
}

Mat3 yaw_mat(const Mat3& R) {
    double fx = R[0][0], fy = R[1][0];
    double norm = std::max(std::hypot(fx, fy), 0.02);
    double c = fx / norm, s = fy / norm;
    Mat3 Y = {{{c, -s, 0.0}, {s, c, 0.0}, {0.0, 0.0, 1.0}}};
    return Y;
}

static Mat3 mat_mul_TA(const Mat3& A, const Mat3& B) {  // A^T * B
    Mat3 C{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k) C[i][j] += A[k][i] * B[k][j];
    return C;
}

std::array<double, 6> mat_to_6d(const Mat3& R) {
    // training geometry.mat_to_6d: rotation[..., :, :2].reshape(-1), row-major
    return {R[0][0], R[0][1], R[1][0], R[1][1], R[2][0], R[2][1]};
}

static Clip make_window_clip(const DeployConfig& cfg, const std::array<double, 4>& quat,
                             const float* joint29, const std::string& name) {
    Mat3 R = quat_to_mat(quat.data());
    std::array<double, 6> rot6 = mat_to_6d(mat_mul_TA(yaw_mat(R), R));
    std::array<float, kGoalFrameDim> row{};
    row[3] = float(rot6[0]); row[4] = float(rot6[1]); row[5] = float(rot6[2]);
    row[6] = float(rot6[3]); row[7] = float(rot6[4]); row[8] = float(rot6[5]);
    for (int j = 0; j < kNumJoints; ++j) row[15 + j] = joint29[j];
    Clip c;
    c.name = name;
    c.T = 2;
    c.goal.resize(2 * kGoalFrameDim * kFuture);
    for (int f = 0; f < 2; ++f)
        for (int w = 0; w < kFuture; ++w)
            std::memcpy(c.goal.data() + (size_t(f) * kFuture + w) * kGoalFrameDim, row.data(),
                        sizeof(row));
    c.joint_pos.resize(2 * kNumJoints);
    c.root_quat.resize(2 * 4);
    c.root_pos.resize(2 * 3, 0.0f);
    for (int f = 0; f < 2; ++f) {
        std::memcpy(c.joint_pos.data() + size_t(f) * kNumJoints, joint29, kNumJoints * sizeof(float));
        for (int i = 0; i < 4; ++i) c.root_quat[size_t(f) * 4 + i] = float(quat[i]);
    }
    c.root_pos[2] = c.root_pos[5] = 0.76f;
    return c;
}

Clip make_static_clip(const DeployConfig& cfg, const std::array<double, 4>& root_quat_wxyz) {
    return make_window_clip(cfg, root_quat_wxyz, cfg.default_joint_pos.data(), "static_stand");
}

Clip make_frozen_clip(const DeployConfig& cfg, const Clip& clip, int frame) {
    if (frame < 0 || frame >= clip.T) throw std::runtime_error("frozen frame out of range");
    std::array<double, 4> quat;
    for (int i = 0; i < 4; ++i) quat[i] = clip.quat_row(frame)[i];
    return make_window_clip(cfg, quat, clip.joint_row(frame), clip.name + "@frozen");
}

}  // namespace uni

namespace uni {

double yaw_of_quat(const double* q_wxyz) {
    Mat3 R = quat_to_mat(q_wxyz);
    return std::atan2(R[1][0], R[0][0]);
}

std::array<double, 4> mat_to_quat(const Mat3& R) {
    double t = R[0][0] + R[1][1] + R[2][2];
    std::array<double, 4> q{};
    if (t > 0.0) {
        double s = std::sqrt(t + 1.0) * 2.0;
        q = {0.25 * s, (R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s};
    } else {
        int i = 0;
        if (R[1][1] > R[0][0]) i = 1;
        if (R[2][2] > R[i][i]) i = 2;
        int j = (i + 1) % 3, k = (i + 2) % 3;
        double s = std::sqrt(R[i][i] - R[j][j] - R[k][k] + 1.0) * 2.0;
        q[0] = (R[k][j] - R[j][k]) / s;
        q[i + 1] = 0.25 * s;
        q[j + 1] = (R[j][i] + R[i][j]) / s;
        q[k + 1] = (R[k][i] + R[i][k]) / s;
    }
    double n = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    for (auto& v : q) v /= n;
    if (q[0] < 0.0) for (auto& v : q) v = -v;
    return q;
}

Clip yaw_rebase_clip(const Clip& clip, double delta_yaw) {
    double c = std::cos(delta_yaw), s = std::sin(delta_yaw);
    double R[3][3] = {{c, -s, 0.0}, {s, c, 0.0}, {0.0, 0.0, 1.0}};
    auto rot3 = [&](float x, float y, float z, double* out) {
        out[0] = R[0][0]*x + R[0][1]*y + R[0][2]*z;
        out[1] = R[1][0]*x + R[1][1]*y + R[1][2]*z;
        out[2] = R[2][0]*x + R[2][1]*y + R[2][2]*z;
    };
    Clip out = clip;  // goal/joint tables carried over unchanged (yaw-invariant)
    out.name = clip.name + "@rebased";
    double p0[3] = {clip.root_pos[0], clip.root_pos[1], clip.root_pos[2]};
    for (int f = 0; f < clip.T; ++f) {
        double r[3];
        rot3(clip.root_pos[f*3] - p0[0], clip.root_pos[f*3+1] - p0[1], clip.root_pos[f*3+2] - p0[2], r);
        for (int i = 0; i < 3; ++i) out.root_pos[f*3+i] = float(p0[i] + r[i]);
        double qd[4];
        for (int i = 0; i < 4; ++i) qd[i] = clip.root_quat[f*4+i];
        Mat3 Rq = quat_to_mat(qd), Rq2{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                Rq2[i][j] = R[i][0]*Rq[0][j] + R[i][1]*Rq[1][j] + R[i][2]*Rq[2][j];
        auto q2 = mat_to_quat(Rq2);
        for (int i = 0; i < 4; ++i) out.root_quat[f*4+i] = float(q2[i]);
    }
    return out;
}

}  // namespace uni
