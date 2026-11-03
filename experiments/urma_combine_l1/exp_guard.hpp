#pragma once
#include "acl/acl.h"
#include "aclnnop/aclnn_exp.h"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// Native equivalent of torch.exp(zero_input, out=output), on the SAME stream
// as the communication kernels. Planning/allocation occurs before the guard.
class ExpGuard {
public:
  void Init(uint32_t rows, uint32_t cols, uint32_t repeat) {
    repeat_ = repeat;
    if (repeat == 0)
      return;
    const uint64_t bytes = uint64_t(rows) * cols * sizeof(float);
    Check(aclrtMalloc(&inputData_, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
          "Exp input allocation");
    Check(aclrtMalloc(&outputData_, bytes, ACL_MEM_MALLOC_HUGE_FIRST),
          "Exp output allocation");
    Check(aclrtMemset(inputData_, bytes, 0, bytes), "Exp zero input");
    const int64_t shape[] = {rows, cols}, strides[] = {cols, 1};
    input_ = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND,
                             shape, 2, inputData_);
    output_ = aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0, ACL_FORMAT_ND,
                              shape, 2, outputData_);
    if (!input_ || !output_)
      throw std::runtime_error("Cannot create Exp tensors");
  }

  void Prepare() {
    // Executors are single-use: get a fresh one for every Exp invocation.
    // Sequential executions can share the maximum-sized workspace.
    plans_.clear();
    plans_.resize(repeat_);
    uint64_t required = 0;
    for (auto &plan : plans_) {
      Check(aclnnExpGetWorkspaceSize(input_, output_, &plan.bytes,
                                     &plan.executor),
            "aclnnExpGetWorkspaceSize");
      required = std::max(required, plan.bytes);
    }
    if (required > workspaceBytes_) {
      if (workspace_)
        Check(aclrtFree(workspace_), "free Exp workspace");
      workspace_ = nullptr;
      Check(aclrtMalloc(&workspace_, required, ACL_MEM_MALLOC_HUGE_FIRST),
            "Exp workspace allocation");
      workspaceBytes_ = required;
    }
  }

  void Enqueue(aclrtStream stream) {
    for (auto &plan : plans_) {
      Check(aclnnExp(workspace_, plan.bytes, plan.executor, stream),
            "aclnnExp");
      plan.executor = nullptr; // consumed by the second-stage API
    }
    // Deliberately no stream synchronization here or between samples.
  }

  void Close() {
    if (input_)
      Check(aclDestroyTensor(input_), "destroy Exp input tensor");
    if (output_)
      Check(aclDestroyTensor(output_), "destroy Exp output tensor");
    if (workspace_)
      Check(aclrtFree(workspace_), "free Exp workspace");
    if (inputData_)
      Check(aclrtFree(inputData_), "free Exp input");
    if (outputData_)
      Check(aclrtFree(outputData_), "free Exp output");
  }

private:
  static void Check(int status, const char *operation) {
    if (status)
      throw std::runtime_error(std::string(operation) +
                               " failed: " + std::to_string(status));
  }
  struct Plan {
    uint64_t bytes = 0;
    aclOpExecutor *executor = nullptr;
  };
  uint32_t repeat_ = 0;
  uint64_t workspaceBytes_ = 0;
  void *inputData_ = nullptr, *outputData_ = nullptr, *workspace_ = nullptr;
  aclTensor *input_ = nullptr, *output_ = nullptr;
  std::vector<Plan> plans_;
};
