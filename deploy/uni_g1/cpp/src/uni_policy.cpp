#include "uni_policy.hpp"

#include <cstring>
#include <stdexcept>

namespace uni {

OnnxPolicy::OnnxPolicy(const std::string& path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "uni_deploy"),
      session_(env_, path.c_str(), Ort::SessionOptions{}) {
    Ort::AllocatorWithDefaultOptions alloc;
    auto check_dim = [&](bool input, const char* name, int64_t expected) {
        size_t n = input ? session_.GetInputCount() : session_.GetOutputCount();
        for (size_t i = 0; i < n; ++i) {
            auto nm = input ? session_.GetInputNameAllocated(i, alloc)
                            : session_.GetOutputNameAllocated(i, alloc);
            if (std::strcmp(nm.get(), name) != 0) continue;
            auto info = input ? session_.GetInputTypeInfo(i) : session_.GetOutputTypeInfo(i);
            auto shape = info.GetTensorTypeAndShapeInfo().GetShape();
            int64_t total = 1;
            for (int64_t d : shape) total *= d;
            if (total != expected)
                throw std::runtime_error(std::string("ONNX ") + name + " dim mismatch");
            return;
        }
        throw std::runtime_error(std::string("ONNX missing tensor ") + name);
    };
    check_dim(true, "obs", kHistory * kObsDim);
    check_dim(true, "goal", kFuture * kGoalFrameDim);
    check_dim(false, "action", kNumJoints);
}

void OnnxPolicy::run(const double* obs, const float* goal, double* action) {
    for (int i = 0; i < kHistory * kObsDim; ++i) obs_f32_[i] = float(obs[i]);
    std::array<int64_t, 2> obs_shape{1, kHistory * kObsDim};
    std::array<int64_t, 2> goal_shape{1, kFuture * kGoalFrameDim};
    std::array<int64_t, 2> out_shape{1, kNumJoints};
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<float, kNumJoints> out{};
    std::array<Ort::Value, 2> inputs{
        Ort::Value::CreateTensor<float>(mem, obs_f32_.data(), obs_f32_.size(), obs_shape.data(),
                                        obs_shape.size()),
        Ort::Value::CreateTensor<float>(mem, const_cast<float*>(goal), kFuture * kGoalFrameDim,
                                        goal_shape.data(), goal_shape.size())};
    std::array<Ort::Value, 1> outputs{
        Ort::Value::CreateTensor<float>(mem, out.data(), out.size(), out_shape.data(),
                                        out_shape.size())};
    const char* in_names[] = {"obs", "goal"};
    const char* out_names[] = {"action"};
    session_.Run(Ort::RunOptions{}, in_names, inputs.data(), inputs.size(), out_names,
                 outputs.data(), outputs.size());
    for (int j = 0; j < kNumJoints; ++j) action[j] = double(out[j]);
}

}  // namespace uni
