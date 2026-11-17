#pragma once

// ONNX Runtime wrapper for the Uni-SuperMimic policy:
//   obs [1,990], goal [1,730] -> action [1,29]

#include <onnxruntime_cxx_api.h>

#include <array>
#include <string>

#include "uni_config.hpp"

namespace uni {

class OnnxPolicy {
public:
    explicit OnnxPolicy(const std::string& path);

    // obs: 990 doubles, goal: 730 floats -> action: 29 doubles
    void run(const double* obs, const float* goal, double* action);

private:
    Ort::Env env_;
    Ort::Session session_;
    std::array<float, kHistory * kObsDim> obs_f32_{};
};

}  // namespace uni
