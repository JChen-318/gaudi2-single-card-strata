// hpu/src/window_hashes.cpp - the writer for upstream's per-verify-window determinism line (see the header for the
// law and the pinned upstream sites).
#include "strata/hpu/window_hashes.hpp"

#include <stdexcept>

namespace strata::hpu {

void WindowHashLog::record(const int64_t window,const int64_t position,const uint32_t row_count,
                          const std::span<const float> rows,const std::span<const int32_t> argmaxes,
                          const uint32_t row_floats) {
    if(file_ == nullptr) return;
    // Upstream hashes exactly the T rows it just copied (`n_floats = T * hc * n_embd`, :13826) and prints exactly the
    // T argmaxes (:13833). A window handed over with fewer rows would hash short and read as a numerics break, and a
    // window whose argmax list is a different length would print a line that no baseline has.
    if(row_count == 0 || row_floats == 0) throw std::invalid_argument("window hash needs rows and a row width");
    if(uint64_t(row_count) * uint64_t(row_floats) != uint64_t(rows.size()))
        throw std::invalid_argument("window hash residual rows do not match the advertised shape");
    if(uint64_t(argmaxes.size()) != uint64_t(row_count))
        throw std::invalid_argument("window hash argmax count differs from the window rows");
    std::fprintf(file_, "%lld %lld %d %016llx", static_cast<long long>(window), static_cast<long long>(position),
                 static_cast<int>(row_count), static_cast<unsigned long long>(window_residual_hash(rows)));
    for (const auto token : argmaxes) std::fprintf(file_, " %d", static_cast<int>(token));
    std::fprintf(file_, "\n");
    ++lines_;
    if (std::fflush(file_) != 0) throw std::invalid_argument("cannot finish window hash record");
}

void WindowHashLog::close() {
    if (file_ == nullptr) return;
    if (std::fflush(file_) != 0 && close_error_.empty()) close_error_ = "cannot flush window hashes";
    std::FILE* finished = file_;
    file_ = nullptr;
    if (std::fclose(finished) != 0 && close_error_.empty()) close_error_ = "cannot close window hash file";
    (void)open_failed_;
}

}
