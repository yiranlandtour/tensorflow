/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_CODEGEN_MATH_ERF_H_
#define XLA_CODEGEN_MATH_ERF_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "llvm/IR/Function.h"
#include "xla/codegen/math/intrinsic.h"
#include "xla/xla_data.pb.h"

namespace xla::codegen {

class Intrinsic::Erf {
 public:
  static std::string Name(PrimitiveType type);
  static std::string Name(PrimitiveType type, int64_t vector_width);

  static llvm::Function* GetOrInsertDeclaration(llvm::Module* module,
                                                PrimitiveType type);
};

namespace math {

// Return the XLA intrinsic name for the erf function:
//
// `xla.erf.v<num_elements><type>`
std::string ErfFunctionName(size_t num_elements, PrimitiveType type);

llvm::Function* CreateErf(llvm::Module* module, llvm::Type* type);

}  // namespace math
}  // namespace xla::codegen

#endif  // XLA_CODEGEN_MATH_ERF_H_
