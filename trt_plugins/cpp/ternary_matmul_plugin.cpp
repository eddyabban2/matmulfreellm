// Reference TensorRT IPluginV2DynamicExt for mmfreelm::TernaryMatMul.
// Prefer the Python plugin (mmfreelm/trt_plugins/ternary_matmul.py) unless you
// need a standalone .so without Python at inference time.

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <vector>

namespace mmfreelm {
void launchTernaryMatmul(
    const void* x,
    const void* w,
    void* y,
    int m,
    int n,
    int k,
    float scale,
    cudaDataType_t dtype,
    cudaStream_t stream);
}  // namespace mmfreelm

using namespace nvinfer1;

namespace {

constexpr const char* kPluginName = "TernaryMatMul";
constexpr const char* kPluginVersion = "1";
constexpr const char* kPluginNamespace = "mmfreelm";

class TernaryMatMulPlugin : public IPluginV2DynamicExt {
 public:
  TernaryMatMulPlugin() = default;

  int getNbOutputs() const noexcept override { return 1; }

  DimsExprs getOutputDimensions(
      int,
      const DimsExprs* inputs,
      int,
      IExprBuilder& exprBuilder) noexcept override {
    DimsExprs out{};
    out.nbDims = 2;
    out.d[0] = inputs[0].d[0];
    out.d[1] = inputs[1].d[0];
    return out;
  }

  bool supportsFormatCombination(
      int pos,
      const PluginTensorDesc* inOut,
      int,
      int) noexcept override {
    if (pos == 0) {
      return (inOut[pos].type == DataType::kHALF || inOut[pos].type == DataType::kFLOAT) &&
             inOut[pos].format == TensorFormat::kLINEAR;
    }
    if (pos == 1) {
      return inOut[pos].type == DataType::kINT8 && inOut[pos].format == TensorFormat::kLINEAR;
    }
    if (pos == 2) {
      return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }
    return inOut[pos].type == inOut[0].type && inOut[pos].format == TensorFormat::kLINEAR;
  }

  void configurePlugin(
      const DynamicPluginTensorDesc*,
      int,
      const DynamicPluginTensorDesc*,
      int) noexcept override {}

  size_t getWorkspaceSize(
      const PluginTensorDesc*,
      int,
      const PluginTensorDesc*,
      int) const noexcept override {
    return 0;
  }

  int enqueue(
      const PluginTensorDesc* inputDesc,
      const PluginTensorDesc* outputDesc,
      const void* const* inputs,
      void* const* outputs,
      void*,
      cudaStream_t stream) noexcept override {
    const auto& xDesc = inputDesc[0].desc;
    int m = xDesc.dims.d[0];
    int k = xDesc.dims.d[1];
    int n = inputDesc[1].desc.dims.d[0];
    float scale = *static_cast<const float*>(inputs[2]);
    cudaDataType_t dtype =
        xDesc.type == DataType::kHALF ? CUDA_R_16F : CUDA_R_32F;
    mmfreelm::launchTernaryMatmul(
        inputs[0], inputs[1], outputs[0], m, n, k, scale, dtype, stream);
    return 0;
  }

  const char* getPluginType() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override { return kPluginVersion; }
  int initialize() noexcept override { return 0; }
  void terminate() noexcept override {}
  size_t getSerializationSize() const noexcept override { return 0; }
  void serialize(void*) const noexcept override {}
  void destroy() noexcept override { delete this; }
  void setPluginNamespace(const char* pluginNamespace) noexcept override {
    namespace_ = pluginNamespace;
  }
  const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }
  DataType getOutputDataType(
      int,
      const DataType* inputTypes,
      int) const noexcept override {
    return inputTypes[0];
  }

 private:
  std::string namespace_{kPluginNamespace};
};

class TernaryMatMulPluginCreator : public IPluginCreator {
 public:
  const char* getPluginName() const noexcept override { return kPluginName; }
  const char* getPluginVersion() const noexcept override { return kPluginVersion; }
  const PluginFieldCollection* getFieldNames() noexcept override {
    return &fields_;
  }

  IPluginV2* createPlugin(const char*, const PluginFieldCollection*) noexcept override {
    return new TernaryMatMulPlugin();
  }

  IPluginV2* deserializePlugin(
      const char*,
      const void*,
      size_t) noexcept override {
    return new TernaryMatMulPlugin();
  }

  void setPluginNamespace(const char* pluginNamespace) noexcept override {
    namespace_ = pluginNamespace;
  }
  const char* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

 private:
  PluginFieldCollection fields_{0, nullptr};
  std::string namespace_{kPluginNamespace};
};

REGISTER_TENSORRT_PLUGIN(TernaryMatMulPluginCreator);

}  // namespace
