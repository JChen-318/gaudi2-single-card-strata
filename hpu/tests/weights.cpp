#include "strata/hpu/weights.hpp"
#include <ggml.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

using namespace strata::hpu;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F&& call) {
    bool rejected = false;
    try { call(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid weight access was accepted");
}
struct Entry {
    uint32_t type;
    uint64_t columns, rows, experts;
    std::vector<uint8_t> data;
    std::vector<uint16_t> expected;
    bool vector = false;
    std::vector<float> expected_f32{};
};
template<class T> void append(std::vector<uint8_t>& bytes, T value) {
    auto* data = reinterpret_cast<const uint8_t*>(&value);
    bytes.insert(bytes.end(), data, data + sizeof(value));
}
void string(std::vector<uint8_t>& bytes, const std::string& value) {
    append<uint64_t>(bytes, value.size());
    bytes.insert(bytes.end(), value.begin(), value.end());
}
struct TempFile {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("strata-hpu-weights-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    ~TempFile() { std::error_code error; std::filesystem::remove(path, error); }
};
void write_fixture(const std::filesystem::path& path, const std::vector<Entry>& entries) {
    std::vector<uint8_t> header;
    append<uint32_t>(header, 0x46554747);
    append<uint32_t>(header, 3);
    append<uint64_t>(header, entries.size());
    append<uint64_t>(header, 0); // No model metadata: this is an operator fixture.
    uint64_t offset = 0;
    for (const auto& entry : entries) {
        string(header, "weight_" + std::to_string(entry.type));
        append<uint32_t>(header, entry.vector ? 1 : entry.experts ? 3 : 2);
        append<uint64_t>(header, entry.columns);
        if (!entry.vector) append<uint64_t>(header, entry.rows);
        if (entry.experts) append<uint64_t>(header, entry.experts);
        append<uint32_t>(header, entry.type);
        append<uint64_t>(header, offset);
        offset = (offset + entry.data.size() + 31) / 32 * 32;
    }
    header.resize((header.size() + 31) / 32 * 32);
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(header.data()), header.size());
    for (const auto& entry : entries) {
        out.write(reinterpret_cast<const char*>(entry.data.data()), entry.data.size());
        const std::array<char, 32> zeros{};
        out.write(zeros.data(), (32 - entry.data.size() % 32) % 32);
    }
    if (!out) throw std::runtime_error("cannot write GGUF fixture");
}
Entry entry(uint32_t type) {
    const auto t = static_cast<ggml_type>(type);
    const auto block = uint64_t(ggml_blck_size(t));
    const auto bytes = ggml_type_size(t);
    Entry result{type, block * 3, 4, type == 0 || type == 1 || type == 30 ? 0U : 3U, {}, {}};
    const auto elements = result.columns * result.rows * std::max(uint64_t(1), result.experts);
    result.data.resize(elements / block * bytes);
    for (size_t i = 0; i < result.data.size(); ++i) result.data[i] = uint8_t(1 + (i * 17 + i / 19) % 47);
    if (type == 0) {
        for (uint64_t i = 0; i < elements; ++i) {
            const float value = float(int(i) - 5) / 8;
            std::memcpy(result.data.data() + i * 4, &value, 4);
        }
    } else if (type == 30) {
        const uint16_t samples[] = {0x0000, 0x8000, 0x3f80, 0xc000, 0x7f81, 0x7fc3, 0x7f80, 0xff80};
        for (uint64_t i = 0; i < elements; ++i) {
            const auto value = samples[i % std::size(samples)];
            std::memcpy(result.data.data() + i * 2, &value, 2);
        }
    }
    std::vector<float> reference(elements);
    const auto* traits = ggml_get_type_traits(t);
    if (type == 0) std::memcpy(reference.data(), result.data.data(), result.data.size());
    else {
        require(traits && traits->to_float, "fixture type lacks reference decoder");
        traits->to_float(result.data.data(), reference.data(), int64_t(elements));
    }
    result.expected.resize(elements);
    result.expected_f32=reference;
    for (size_t i = 0; i < reference.size(); ++i) result.expected[i] = to_bf16(reference[i]);
    if (type == 30) std::memcpy(result.expected.data(), result.data.data(), result.data.size());
    return result;
}
}
int main() {
    try {
        // Actual bytes are placed into one GGUF. The oracle decodes whole source
        // tensors; the implementation maps/slices and decodes bounded row chunks.
        std::vector<Entry> entries;
        for (const uint32_t type : {0U, 1U, 2U, 3U, 6U, 7U, 8U, 10U, 11U, 12U, 13U, 14U,
                                   16U, 17U, 18U, 19U, 20U, 21U, 22U, 23U, 29U, 30U, 34U, 35U, 39U, 40U, 41U, 42U})
            entries.push_back(entry(type));
        TempFile file;
        write_fixture(file.path, entries);
        {
            WeightSource source(file.path.string());
            require(source.inventory().size() == entries.size(), "GGUF inventory mismatch");
            for (const auto& e : entries) {
                const auto name = "weight_" + std::to_string(e.type);
                for (uint64_t expert = 0; expert < std::max(uint64_t(1), e.experts); ++expert) {
                    auto slice = e.experts ? source.expert(name, expert) : source.matrix(name);
                    require(slice.columns() == e.columns && slice.rows() == e.rows, "matrix axes reversed");
                    std::vector<uint16_t> actual(2 * e.columns);
                    slice.convert_rows(1, actual);
                    const auto offset = (expert * e.rows + 1) * e.columns;
                    require(std::equal(actual.begin(), actual.end(), e.expected.begin() + offset), "slice conversion mismatch");
                    std::vector<float> f32(2*e.columns);
                    slice.convert_rows_f32(1,f32);
                    require(std::memcmp(f32.data(),e.expected_f32.data()+offset,f32.size()*sizeof(float))==0,"FP32 slice conversion mismatch");
                    slice.convert_rows(e.rows, {}); // legal empty tail
                    rejects([&] { slice.convert_rows(e.rows, actual); });
                    rejects([&] { slice.convert_rows(UINT64_MAX, actual); });
                    rejects([&] { slice.convert_rows(0, std::span<uint16_t>(actual).first(1)); });
                }
                if (e.experts) rejects([&] { (void)source.expert(name, e.experts); });
                std::cout << "PASS GGUF type=" << e.type << " expert_stride rows[1:3] BF16_bits=exact\n";
            }
            rejects([&] { (void)source.matrix("missing"); });
            rejects([&] { (void)source.expert("weight_0", 0); });
        }
        // A borrowed expert retains its mapping after the source is destroyed.
        const auto retained = [&] { WeightSource source(file.path.string()); return source.expert("weight_17", 2); }();
        std::vector<uint16_t> output(retained.columns());
        retained.convert_rows(3, output);
        const auto& iq = *std::find_if(entries.begin(), entries.end(), [](const Entry& e) { return e.type == 17; });
        require(std::equal(output.begin(), output.end(), iq.expected.begin() + (2 * iq.rows + 3) * iq.columns),
                "mmap lifetime mismatch");
        // Truncating an otherwise valid directory is caught before tensor access.
        TempFile truncated;
        write_fixture(truncated.path, entries);
        std::filesystem::resize_file(truncated.path, std::filesystem::file_size(truncated.path) - 64);
        rejects([&] { WeightSource source(truncated.path.string()); (void)source.inventory(); });
        TempFile overflow;
        auto malformed = entries;
        malformed[0].columns = UINT64_MAX;
        write_fixture(overflow.path, malformed);
        rejects([&] { WeightSource source(overflow.path.string()); (void)source.inventory(); });
        // Raw I8 is used for FP8 tables by upstream. It must not be interpreted
        // as an ordinary signed integer or silently converted as BF16 weights.
        TempFile raw;
        write_fixture(raw.path, {{24, 32, 2, 0, std::vector<uint8_t>(64), {}}});
        WeightSource raw_source(raw.path.string());
        require(!raw_source.inventory()[0].bf16_conversion_supported, "raw I8/FP8 incorrectly supported");
        rejects([&] { (void)raw_source.matrix("weight_24"); });
        {
            WeightSource source(file.path.string());
            const auto matrix=source.matrix("weight_0");
            std::vector<float> transposed(matrix.rows()*matrix.columns());
            transpose_f32(matrix,transposed,matrix.columns()*sizeof(float));
            for(uint64_t row=0;row<matrix.rows();++row) for(uint64_t col=0;col<matrix.columns();++col) {
                float expected=0;
                std::memcpy(&expected,entries[0].data.data()+(row*matrix.columns()+col)*4,4);
                require(transposed[col*matrix.rows()+row]==expected,"FP32 transpose changes value or axes");
            }
            rejects([&] { transpose_f32(matrix,transposed,1); });
            rejects([&] { transpose_f32(matrix,std::span<float>(transposed).first(1)); });
        }
        TempFile vectors;
        auto scalar_vector=entry(0);
        scalar_vector.rows=1; scalar_vector.columns=scalar_vector.data.size()/4; scalar_vector.vector=true;
        write_fixture(vectors.path,{scalar_vector});
        WeightSource vector_source(vectors.path.string());
        auto mapped_vector=vector_source.vector("weight_0");
        std::vector<float> decoded_vector(mapped_vector.columns());
        mapped_vector.convert_rows_f32(0,decoded_vector);
        require(std::memcmp(decoded_vector.data(),scalar_vector.data.data(),scalar_vector.data.size())==0,"FP32 vector bits changed");
        rejects([&] { (void)vector_source.matrix("weight_0"); });
        std::cout << "PASS FP32 constants, rank-1 views, bounded GGUF-to-HPU conv transpose\n";
        std::cout << "PASS mmap lifetime, truncated payload, overflow and expert bounds; no_device_acquired\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
