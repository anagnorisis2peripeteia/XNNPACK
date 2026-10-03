// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "include/xnnpack.h"
#include "src/xnnpack/datatype.h"

using testing::Eq;

namespace {

struct DatatypeSize {
  xnn_datatype datatype;
  size_t log2_size_bits;
};

// Every valid datatype, with log2 of its size in bits.
const DatatypeSize kDatatypeSizes[] = {
    {xnn_datatype_qint2, 1},   {xnn_datatype_qcint2, 1},
    {xnn_datatype_qint4, 2},   {xnn_datatype_qcint4, 2},
    {xnn_datatype_qbint4, 2},  {xnn_datatype_qint8, 3},
    {xnn_datatype_pqint8, 3},  {xnn_datatype_quint8, 3},
    {xnn_datatype_qcint8, 3},  {xnn_datatype_qdint8, 3},
    {xnn_datatype_qduint8, 3}, {xnn_datatype_qpint8, 3},
    {xnn_datatype_fp16, 4},    {xnn_datatype_bf16, 4},
    {xnn_datatype_pfp16, 4},   {xnn_datatype_qint32, 5},
    {xnn_datatype_qcint32, 5}, {xnn_datatype_int32, 5},
    {xnn_datatype_fp32, 5},    {xnn_datatype_pfp32, 5},
};

TEST(Datatype, Log2SizeBits) {
  for (const DatatypeSize& entry : kDatatypeSizes) {
    EXPECT_THAT(xnn_datatype_log2_size_bits(entry.datatype),
                Eq(entry.log2_size_bits));
  }
}

TEST(Datatype, SizeBits) {
  for (const DatatypeSize& entry : kDatatypeSizes) {
    EXPECT_THAT(xnn_datatype_size_bits(entry.datatype),
                Eq(size_t{1} << entry.log2_size_bits));
  }
}

TEST(Datatype, SizeBytes) {
  // Sub-byte types have no whole-byte size, so they are not covered here.
  for (const DatatypeSize& entry : kDatatypeSizes) {
    if (entry.log2_size_bits < 3) {
      continue;
    }
    EXPECT_THAT(xnn_datatype_log2_size_bytes(entry.datatype),
                Eq(entry.log2_size_bits - 3));
    EXPECT_THAT(xnn_datatype_size_bytes(entry.datatype),
                Eq(size_t{1} << (entry.log2_size_bits - 3)));
  }
}

// xnn_datatype_log2_size_bits reports SIZE_MAX for xnn_datatype_invalid, so
// these must not shift by it: `1 << SIZE_MAX` is undefined behaviour, and the
// value it produced in practice was 2^31 for size_bits and 2^28 (256 MiB) for
// size_bytes. The latter is used directly as a memcpy length by
// xnn_subgraph's constant folding, so it has to be small.
TEST(Datatype, InvalidDatatypeHasNoSize) {
  EXPECT_THAT(xnn_datatype_size_bits(xnn_datatype_invalid), Eq(size_t{0}));
  EXPECT_THAT(xnn_datatype_size_bytes(xnn_datatype_invalid), Eq(size_t{0}));
}

}  // namespace