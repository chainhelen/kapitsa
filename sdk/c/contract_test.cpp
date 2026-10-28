#include "kapitsa.h"
#include "options.h"
#include <iostream>
#include <sys/file.h>
#include <unistd.h>
#include <fstream>
#include <cstring>
#include <stdexcept>
#define CHECK(x) do { if (!(x)) throw std::runtime_error(#x); } while (0)
template<class F> void invalid(F f) {
  bool threw = false;
  try { f(); } catch (const std::exception&) { threw = true; }
  CHECK(threw);
}
int main() {
  CHECK(kapitsa_abi_version() == 1);
  for (auto bytes : {std::string("\xe4\xbd"), std::string("\xa0")})
    CHECK(ni::ojson::parse(kapitsa_sdk::dump(ni::ojson{{"content", bytes}}))["content"] == "\xef\xbf\xbd");
  kapitsa_engine* engine = nullptr;
  char* error = nullptr;
  CHECK(kapitsa_engine_create("{", &engine, &error) == KAPITSA_INVALID_ARGUMENT);
  CHECK(!engine && error); kapitsa_string_free(error);
  CHECK(kapitsa_engine_create("{}", &engine, &error) == KAPITSA_INVALID_ARGUMENT);
  CHECK(!engine && error); kapitsa_string_free(error);
  CHECK(kapitsa_engine_create("{\"model_dir\":\"/missing-kapitsa-sdk-model\"}", &engine, &error) == KAPITSA_INVALID_ARGUMENT);
  CHECK(!engine && error); kapitsa_string_free(error);
  char path[] = "/tmp/kapitsa-sdk-lock-XXXXXX";
  int fd = mkstemp(path);
  CHECK(fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) == 0);
  setenv("BENCH_GPU_LOCK", path, 1);
  unsetenv("NI_LOCK_HELD"); unsetenv("NACE_GPU_LOCK_HELD");
  std::string dir = std::string(path) + "-model";
  std::filesystem::create_directory(dir);
  std::ofstream(dir + "/config.json") << "{}";
  auto config = kapitsa_sdk::dump(ni::ojson{{"model_dir", dir}});
  alarm(5);  // A regression to blocking acquire must fail, not hang CI.
  for (int i = 0; i < 2; ++i) {
    CHECK(kapitsa_engine_create(config.c_str(), &engine, &error) == KAPITSA_BUSY);
    CHECK(!engine && error && std::strstr(error, "GPU lock")); kapitsa_string_free(error);
  }
  alarm(0);
  close(fd); unlink(path); unsetenv("BENCH_GPU_LOCK");
  std::filesystem::remove_all(dir);
  char* result = nullptr;
  CHECK(kapitsa_generate(nullptr, nullptr, nullptr, &result, &error) == KAPITSA_INVALID_ARGUMENT);
  CHECK(!result && error); kapitsa_string_free(error);
  // Exercise the full integer range without converting an unsigned value to signed first.
  CHECK(kapitsa_sdk::integer(ni::ojson{{"depth", -1}}, "depth", 0, -1, 32) == -1);
  CHECK(kapitsa_sdk::integer(ni::ojson{{"depth", 32}}, "depth", 0, -1, 32) == 32);
  for (const char* value : {"-2", "33", "9223372036854775808", "18446744073709551615"}) {
    const auto json = ni::ojson::parse(std::string("{\"depth\":") + value + "}");
    invalid([&] { kapitsa_sdk::integer(json, "depth", 0, -1, 32); });
  }
  ni::ChatRequest defaults;
  auto good = ni::ojson::parse(R"({"messages":[{"role":"user","content":"hello"}],"temperature":0,"seed":0})");
  auto r = kapitsa_sdk::request(good, defaults);
  CHECK(r.temperature == 0 && r.seed == 0 && r.max_tokens == defaults.max_tokens);
  good["seed"] = UINT32_MAX;
  CHECK(kapitsa_sdk::request(good, defaults).seed == UINT32_MAX);
  for (auto bad : {R"({"messages":[]})", R"({"messages":[{"role":"user"}],"max_tokens":-1})",
      R"({"messages":[{"role":"user"}],"seed":4294967296})",
      R"({"messages":[{"role":"user"}],"temperature":true})",
      R"({"messages":[{"role":"user"}],"typo":1})"})
    invalid([&] { kapitsa_sdk::request(ni::ojson::parse(bad), defaults); });
  kapitsa_request_cancel(nullptr); kapitsa_request_destroy(nullptr);
  kapitsa_engine_destroy(nullptr); kapitsa_string_free(nullptr);
  std::cout << "SDK CPU contract checks passed (no model or GPU)\n";
}
