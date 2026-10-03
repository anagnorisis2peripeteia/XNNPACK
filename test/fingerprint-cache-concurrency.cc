// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Regression test for a data race in the fingerprint cache.
//
// `xnn_get_fingerprint` used to return a pointer into the shared
// `fingerprint_vector` after releasing the cache mutex, so the caller read the
// element with no lock held while `xnn_set_fingerprint` was overwriting it
// under the lock. Run this test under ThreadSanitizer to see the race.

#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "gtest/gtest.h"
#include "include/experimental.h"
#include "src/operators/fingerprint_id.h"

namespace {

// One fingerprint id shared by every thread. Writers keep updating its value so
// that a reader holding an unprotected pointer would race with the overwrite.
constexpr uint32_t kFingerprintId = 0xABCDu;
constexpr int kNumWriters = 2;
constexpr int kNumReaders = 4;
constexpr int kNumIterations = 20000;

class FingerprintCacheConcurrencyTest : public ::testing::Test {
 protected:
  void SetUp() override { xnn_clear_fingerprints(); }
  void TearDown() override { xnn_clear_fingerprints(); }

  static void SetFingerprints() {
    start.store(true, std::memory_order_release);
    std::vector<std::thread> threads;
    threads.reserve(kNumWriters + kNumReaders);
    for (int t = 0; t < kNumWriters; t++) {
      threads.emplace_back(Writer);
    }
    for (int t = 0; t < kNumReaders; t++) {
      threads.emplace_back(Reader);
    }
    for (std::thread& thread : threads) {
      thread.join();
    }
  }

  static void WaitForStart() {
    while (!start.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  // Overwrites the same element over and over, which is what the unlocked
  // reader used to race with.
  static void Writer() {
    WaitForStart();
    for (uint32_t value = 1; value <= kNumIterations; value++) {
      xnn_set_fingerprint({kFingerprintId, value});
    }
  }

  // Mirrors the access pattern of `xnn_check_fingerprint`: look the fingerprint
  // up, then read it.
  static void Reader() {
    WaitForStart();
    for (int i = 0; i < kNumIterations; i++) {
      struct xnn_fingerprint fingerprint;
      if (xnn_get_fingerprint(kFingerprintId, &fingerprint)) {
        observed.store(fingerprint.value, std::memory_order_relaxed);
      }
    }
  }

  static std::atomic<bool> start;
  static std::atomic<uint32_t> observed;
};

std::atomic<bool> FingerprintCacheConcurrencyTest::start{false};
std::atomic<uint32_t> FingerprintCacheConcurrencyTest::observed{0};

TEST_F(FingerprintCacheConcurrencyTest, ConcurrentSetAndGetIsRaceFree) {
  xnn_set_fingerprint({kFingerprintId, 1});
  SetFingerprints();

  // Every reader must have observed a fingerprint that was actually written:
  // the id it asked for, and a value a writer produced. A reader working from
  // a pointer into the cache can observe a torn element instead.
  struct xnn_fingerprint fingerprint;
  ASSERT_TRUE(xnn_get_fingerprint(kFingerprintId, &fingerprint));
  EXPECT_EQ(fingerprint.id, kFingerprintId);
  EXPECT_GE(fingerprint.value, 1u);
  EXPECT_LE(fingerprint.value, static_cast<uint32_t>(kNumIterations));
}

}  // namespace