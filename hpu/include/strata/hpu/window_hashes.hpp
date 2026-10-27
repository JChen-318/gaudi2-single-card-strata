#pragma once
// hpu/include/strata/hpu/window_hashes.hpp - upstream's per-verify-window determinism record.
//
// `--window-hashes PATH` (declared src/program/generate.cpp:586, parsed :2088, opened :13723-13727, written
// :13825-13834, closed :13912) emits ONE LINE PER VERIFY WINDOW:
//     "<window index> <position> <T> <16 hex FNV-1a> <argmax> ..."
// where T is the number of rows the window actually carried, the argmaxes are the window's emitted tokens, and the
// hex field is a 64-bit FNV-1a over the RAW BYTES of the window's T final residual rows (T * hc * n_embd floats,
// upstream's `ver.final_R_all()` copied down in row order - :13826-13831). Its stated purpose is the one that makes
// the exact byte layout a law rather than an implementation detail: "two builds that do the same arithmetic write the
// same file" (:861-863), so any change in the offset basis, the prime, the byte order of the rows, or the line format
// silently breaks every comparison against a captured baseline.
//
// Upstream reads the rows off the device; this backend assembles them from its own per-position residual producer
// (`SessionOperations::residual`, the hc x n_embd value upstream calls `ss.R`). A window whose rows the engine cannot
// hand over row by row is REFUSED rather than hashed over less data, because a shorter hash would still produce a
// well-formed file and then differ from the baseline for the wrong reason.

#include <cstdint>
#include <cstdio>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::hpu {

// Upstream's exact constants (:13829-13830): offset basis 1469598103934665603, prime 1099511628211, one byte at a
// time, multiply AFTER the xor.
inline uint64_t fnv1a64(const std::span<const std::uint8_t> bytes) {
    uint64_t hash = 1469598103934665603ull;
    for (const auto byte : bytes) hash = (hash ^ byte) * 1099511628211ull;
    return hash;
}

// The hash field of one window: the rows in window order, each `hc * n_embd` floats, hashed over their FP32 object
// representation exactly as upstream memcpy's them into a host vector.
inline uint64_t window_residual_hash(const std::span<const float> rows) {
    return fnv1a64(std::span(reinterpret_cast<const std::uint8_t*>(rows.data()), rows.size() * sizeof(float)));
}

class WindowHashLog {
public:
    explicit WindowHashLog(std::string path) : path_(std::move(path)) {
        if (path_.empty()) return;
        // Upstream opens this with mode "w" (:13723), which on Linux IS the byte stream. This port opens it in
        // binary so the record is the same bytes on every host: text mode would turn each '\n' into CRLF on
        // Windows, and a determinism file whose bytes depend on the platform defeats its only purpose (:861-863).
        file_ = std::fopen(path_.c_str(), "wb");
        if (file_ == nullptr) throw std::invalid_argument("strata generate: cannot write " + path_);
    }
    WindowHashLog(const WindowHashLog&) = delete;
    WindowHashLog& operator=(const WindowHashLog&) = delete;
    WindowHashLog(WindowHashLog&& other) noexcept : path_(std::move(other.path_)), file_(other.file_),
                                                    lines_(other.lines_), open_failed_(other.open_failed_) {
        other.file_ = nullptr;
        other.lines_ = 0;
    }
    WindowHashLog& operator=(WindowHashLog&& other) noexcept {
        if (this != &other) {
            close();
            path_ = std::move(other.path_);
            file_ = other.file_;
            lines_ = other.lines_;
            open_failed_ = other.open_failed_;
            other.file_ = nullptr;
            other.lines_ = 0;
        }
        return *this;
    }
    ~WindowHashLog() { close(); }

    bool armed() const { return file_ != nullptr; }
    uint64_t lines() const { return lines_; }
    const std::string& path() const { return path_; }
    std::string close_error() const { return close_error_; }

    // One window's line. `rows` must be exactly `row_count` residual rows of `row_floats` floats each; anything else
    // is refused, because a short hash would look like a determinism break instead of a wiring gap.
    void record(const int64_t window, const int64_t position, const uint32_t row_count,
                const std::span<const float> rows, const std::span<const int32_t> argmaxes, const uint32_t row_floats);

    void close();

private:
    std::string path_;
    std::FILE* file_ = nullptr;
    uint64_t lines_ = 0;
    bool open_failed_ = false;
    std::string close_error_;
};

}
