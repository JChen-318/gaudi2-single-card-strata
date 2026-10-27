// hpu/tests/vision_prompt.cpp - host evidence for --vision's request law
// (src/program/generate.cpp:9682-9686, :10635-10845; include/strata/kernels/mrope.hpp).
//
// Claims, all without a device:
//  (a) the SVE1 reader accepts exactly the records upstream accepts and refuses each malformed file
//      with upstream's sentence (missing file, bad magic/geometry/width, short payload, partial tail);
//  (b) the prompt plan reproduces mtmd's positions: t = p, h = p + y, w = p + x, text after a picture
//      at p + max(nx, ny), the table's pair sector = pair % 3, and image rows only on image-pad cells;
//  (c) the picture identity is the grid then the server's img_keys[k] when one was given per picture,
//      else the embedding bytes, hashed with upstream's fnv1a;
//  (d) every planning refusal carries upstream's exact sentence.
#include "strata/hpu/vision_prompt.hpp"
#include "strata/hpu/protocol.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace strata::hpu;

int failures = 0;
void require(const bool ok, const char* const what) {
    if (!ok) { std::printf("FAIL %s\n", what); ++failures; }
}
template <class Call>
void must_refuse(Call&& call, const char* const needle, const char* const what) {
    std::string message;
    bool refused = false;
    try { call(); } catch (const std::exception& error) { refused = true; message = error.what(); }
    if (!refused) { std::printf("FAIL no refusal: %s\n", what); ++failures; }
    else if (message.find(needle) == std::string::npos) { std::printf("FAIL wrong refusal for %s: %s\n", what, message.c_str()); ++failures; }
}

// The test's own record writer: little-endian int32 header, then the rows. Never uses the module.
void append_i32(std::string& out, const int32_t v) { out.append(reinterpret_cast<const char*>(&v), 4); }
std::string sve1(const int32_t n, const int32_t nx, const int32_t ny, const int32_t width, const float first) {
    std::string out;
    append_i32(out, 0x31455653); append_i32(out, n); append_i32(out, nx); append_i32(out, ny); append_i32(out, width);
    for (int i = 0; i < n * width; ++i) {
        const float value = first + float(i);
        out.append(reinterpret_cast<const char*>(&value), 4);
    }
    return out;
}
std::filesystem::path write(const std::string& name, const std::string& bytes) {
    const auto path = std::filesystem::temp_directory_path() / ("strata_vision_" + name);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), std::streamsize(bytes.size()));
    out.close();
    if (!out) throw std::runtime_error("cannot write fixture");
    return path;
}

// Upstream's hash, transcribed independently for the test (src/program/generate.cpp:1502-1506).
uint64_t test_fnv(const void* data, const std::size_t n, uint64_t h = 1469598103934665603ull) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (std::size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}
uint64_t grid_hash(const uint32_t n, const uint32_t nx, const uint32_t ny) {
    const int64_t grid[3] = {int64_t(n), int64_t(nx), int64_t(ny)};
    return test_fnv(grid, sizeof grid);
}

void reader_cases() {
    const std::string both = sve1(4, 2, 2, 3, 1.0f) + sve1(6, 3, 2, 3, 101.0f);
    const auto good = write("both.sve1", both);
    const auto embeddings = VisionEmbeddings::load(good.string(), 3);
    require(embeddings.rows() == 10 && embeddings.hidden() == 3 && embeddings.records().size() == 2, "two records load");
    require(embeddings.records()[1].row == 4 && embeddings.records()[1].nx == 3 && embeddings.records()[1].ny == 2, "record offsets");
    const auto row = embeddings.row(4);
    require(row.size() == 3 && row[0] == 101.0f && row[2] == 103.0f, "row 4 is record 2's first row");
    require(embeddings.row(9)[2] == 118.0f, "last row's last value");

    must_refuse([&] { (void)VisionEmbeddings::load((std::filesystem::temp_directory_path() / "strata_vision_absent.sve1").string(), 3); },
                "cannot open", "missing file");
    std::string magic = both;
    magic[0] = 'X';
    must_refuse([&] { (void)VisionEmbeddings::load(write("magic.sve1", magic).string(), 3); },
                "bad embeddings file (expected strata-vision records of width 3)", "bad magic");
    std::string geometry = sve1(4, 3, 2, 3, 0.0f);  // nx*ny = 6 != n = 4
    must_refuse([&] { (void)VisionEmbeddings::load(write("geometry.sve1", geometry).string(), 3); },
                "bad embeddings file (expected strata-vision records of width 3)", "grid does not match n");
    std::string width = sve1(4, 2, 2, 4, 0.0f);
    must_refuse([&] { (void)VisionEmbeddings::load(write("width.sve1", width).string(), 3); },
                "bad embeddings file (expected strata-vision records of width 3)", "record width");
    std::string short_payload = both.substr(0, both.size() - 4);
    must_refuse([&] { (void)VisionEmbeddings::load(write("short.sve1", short_payload).string(), 3); },
                "short embeddings file", "truncated payload");
    std::string partial = both + std::string("\x01\x02", 2);
    must_refuse([&] { (void)VisionEmbeddings::load(write("partial.sve1", partial).string(), 3); },
                "bad embeddings file (expected strata-vision records of width 3)", "partial trailing record");
}

void plan_cases() {
    const auto path = write("plan.sve1", sve1(4, 2, 2, 3, 1.0f) + sve1(6, 3, 2, 3, 101.0f));
    const auto embeddings = VisionEmbeddings::load(path.string(), 3);
    const std::vector<int32_t> prompt{7, kVisionImagePad, kVisionImagePad, kVisionImagePad, kVisionImagePad, 8,
                                      kVisionImagePad, kVisionImagePad, kVisionImagePad, kVisionImagePad, kVisionImagePad,
                                      kVisionImagePad, 9};
    const auto plan = plan_vision_prompt(prompt, embeddings, 20);
    require(plan.cells == 20 && plan.rows.size() == prompt.size() && plan.images.size() == 2, "plan shapes");
    // Text before the first picture, then mtmd's grid, then text at p + max(nx, ny).
    const int32_t text_expected[][3] = {{0, 0, 0}, {3, 3, 3}, {7, 7, 7}, {8, 8, 8}, {14, 14, 14}};
    const uint64_t text_cells[] = {0, 5, 12, 13, 19};
    for (int i = 0; i < 5; ++i)
        for (int stream = 0; stream < 3; ++stream)
            require(plan.positions[text_cells[i] * 3 + stream] == text_expected[i][stream], "text position");
    // Image A: cells 1..4, t = 1, h = 1 + y, w = 1 + x.
    const int32_t a_expected[][3] = {{1, 1, 1}, {1, 1, 2}, {1, 2, 1}, {1, 2, 2}};
    for (int j = 0; j < 4; ++j)
        for (int stream = 0; stream < 3; ++stream)
            require(plan.positions[(1 + j) * 3 + stream] == a_expected[j][stream], "image A position");
    // Image B: cells 6..11, t = 4, h = 4 + y, w = 4 + x.
    for (int j = 0; j < 6; ++j) {
        const int y = j / 3, x = j % 3;
        require(plan.positions[(6 + j) * 3] == 4 && plan.positions[(6 + j) * 3 + 1] == 4 + y &&
                plan.positions[(6 + j) * 3 + 2] == 4 + x, "image B position");
    }
    require(plan.position(6 + 5, 1) == 5 && plan.position(6 + 5, 2) == 6 && plan.position(6 + 5, 3) == 4, "pair sector law");
    require(plan.image_row(0) == -1 && plan.image_row(1) == 0 && plan.image_row(4) == 3 && plan.image_row(5) == -1 &&
            plan.image_row(6) == 4 && plan.image_row(11) == 9 && plan.image_row(12) == -1, "image rows land on pads");
    require(plan.images[0].start == 1 && plan.images[0].rows == 4 && plan.images[0].nx == 2 && plan.images[0].ny == 2, "image A run");
    require(plan.images[1].start == 6 && plan.images[1].rows == 6 && plan.images[1].nx == 3, "image B run");

    // Identity: grid, then the embedding bytes.
    const uint64_t key_a = test_fnv(embeddings.row(0).data(), 4 * 3 * sizeof(float), grid_hash(4, 2, 2));
    const uint64_t key_b = test_fnv(embeddings.row(4).data(), 6 * 3 * sizeof(float), grid_hash(6, 3, 2));
    require(plan.images[0].key == key_a && plan.images[1].key == key_b, "embedding identity");
    const std::vector<uint64_t> keys{0x0123456789abcdefull, 0xfedcba9876543210ull};
    const auto keyed = plan_vision_prompt(prompt, embeddings, 20, keys);
    const uint64_t keyed_a = test_fnv(&keys[0], sizeof(uint64_t), grid_hash(4, 2, 2));
    const uint64_t keyed_b = test_fnv(&keys[1], sizeof(uint64_t), grid_hash(6, 3, 2));
    require(keyed.images[0].key == keyed_a && keyed.images[1].key == keyed_b, "img_keys identity");
    // A key count that is not the picture count is ignored, exactly as upstream reads it.
    const std::vector<uint64_t> wrong_keys{1};
    require(plan_vision_prompt(prompt, embeddings, 20, wrong_keys).images[0].key == key_a, "mismatched img_keys ignored");
}

void refusal_cases() {
    const auto path = write("refuse.sve1", sve1(4, 2, 2, 3, 1.0f));
    const auto embeddings = VisionEmbeddings::load(path.string(), 3);
    std::vector<int32_t> prompt{1, 2, kVisionImagePad, 3};
    must_refuse([&] { (void)plan_vision_prompt(prompt, embeddings, 8); },
                "image 1 has 4 rows but fewer <|image_pad|> tokens", "short image run");
    must_refuse([&] { (void)plan_vision_prompt(std::vector<int32_t>{1, 2, kVisionImagePad, kVisionImagePad, kVisionImagePad, kVisionImagePad}, embeddings, 8); },
                "the prompt cannot end in an image", "prompt ends in image");
    const auto two = write("refuse2.sve1", sve1(4, 2, 2, 3, 1.0f) + sve1(4, 2, 2, 3, 21.0f));
    const auto two_embeddings = VisionEmbeddings::load(two.string(), 3);
    must_refuse([&] { (void)plan_vision_prompt(std::vector<int32_t>{1, 2}, two_embeddings, 8); },
                "the embeddings file has more images than the prompt", "file has more pictures");
    must_refuse([&] { (void)plan_vision_prompt(prompt, VisionEmbeddings{}, 8); },
                "the prompt has more images than the embeddings file", "prompt has more pictures");
    must_refuse([&] { (void)plan_vision_prompt(std::vector<int32_t>{1, 2}, embeddings, 1); },
                "prompt exceeds the image position table", "table shorter than prompt");
}

void text_and_rope_cases() {
    const auto text = plan_text_prompt(6, 3);
    require(text.images.empty() && text.rows.size() == 3 && text.image_row(1) == -1, "text plan has no pictures");
    for (uint64_t c = 0; c < 6; ++c)
        require(text.positions[c * 3] == int32_t(c) && text.positions[c * 3 + 1] == int32_t(c) &&
                text.positions[c * 3 + 2] == int32_t(c), "identity table");

    const strata::kernels::RopeScaling scaling;  // the identity: freq_base 1e7, no scaling
    const auto sectors = rope_row_sectors(6, 10, 20, 30, scaling);
    require(sectors.cosine.size() == 3 && sectors.sine.size() == 3, "sector rope pairs");
    const uint64_t positions[3] = {10, 20, 30};
    for (uint32_t pair = 0; pair < 3; ++pair) {
        const double angle = double(positions[pair % 3]) * std::pow(scaling.freq_base, -2.0 * double(pair) / 6.0);
        require(sectors.cosine[pair] == float(std::cos(angle)) && sectors.sine[pair] == float(std::sin(angle)), "sector angle");
    }
    const auto flat = rope_row(6, 10, scaling);
    const auto same = rope_row_sectors(6, 10, 10, 10, scaling);
    require(flat.cosine == same.cosine && flat.sine == same.sine, "text is the identity of the sector law");
    // Pair 0 is t, pair 1 h, pair 2 w: three different positions must give three different angles.
    require(!(sectors.cosine[0] == sectors.cosine[1] && sectors.cosine[1] == sectors.cosine[2]), "sectors differ");
    must_refuse([&] { (void)rope_row_sectors(6, uint64_t(INT32_MAX) + 1, 0, 0, scaling); }, "invalid RoPE geometry/position", "sector overflow");
}

// The installed table must be the single rotation seam: identity cells reproduce rope_row exactly, and a
// picture request reads its (t,h,w) sectors through rope_row_at, including pair % 3.
void installed_rope_cases() {
    const strata::kernels::RopeScaling scaling;
    const auto flat7 = rope_row(6, 7, scaling);
    const auto via_null = rope_row_at(nullptr, 6, 7, scaling);
    require(via_null.cosine == flat7.cosine && via_null.sine == flat7.sine, "null table is the text rope");
    RopePositions empty;
    require(!empty.installed() && empty.cells() == 0, "no table before --vision");
    RopePositions table(9);
    require(table.installed() && table.cells() == 9, "installed table cells");
    for (uint64_t cell = 0; cell < 9; ++cell)
        for (uint32_t pair = 0; pair < 5; ++pair)
            require(table.position(cell, pair) == int32_t(cell), "identity cell");
    const auto flat4 = rope_row(6, 4, scaling);
    const auto via_identity = rope_row_at(&table, 6, 4, scaling);
    require(via_identity.cosine == flat4.cosine && via_identity.sine == flat4.sine, "identity table matches text rope");
    std::vector<int32_t> cells(27);
    for (uint64_t cell = 0; cell < 9; ++cell) {
        cells[cell * 3] = int32_t(cell + 1);
        cells[cell * 3 + 1] = int32_t(2 * cell + 3);
        cells[cell * 3 + 2] = int32_t(cell + 5);
    }
    table.install(cells);
    require(table.position(2, 0) == 3 && table.position(2, 1) == 7 && table.position(2, 2) == 7 &&
            table.position(2, 3) == 3, "pair sector selects t/h/w");
    const auto via_table = rope_row_at(&table, 6, 2, scaling);
    const auto sectors = rope_row_sectors(6, 3, 7, 7, scaling);
    require(via_table.cosine == sectors.cosine && via_table.sine == sectors.sine, "installed table drives sectors");
    const auto flat9 = rope_row(6, 9, scaling);
    const auto past = rope_row_at(&table, 6, 9, scaling);
    require(past.cosine == flat9.cosine && past.sine == flat9.sine, "past the table is text");
    table.reset_identity();
    require(table.position(5, 2) == 5, "identity restored");
    must_refuse([&] { table.install(std::span<const int32_t>(cells.data(), 26)); },
                "image position table shape mismatch", "table shape");
    std::vector<int32_t> negative(cells);
    negative[0] = -1;
    must_refuse([&] { table.install(negative); }, "image position table contains a negative position", "negative cell");
    must_refuse([&] { (void)RopePositions(uint64_t(INT32_MAX) + 1); },
                "image position table exceeds the int32 cell range", "table cell overflow");
}

// GENI's request grammar and its wiring into GenerationRequest (upstream :10635-10828).
void request_cases() {
    const auto path = write("request.sve1", sve1(4, 2, 2, 3, 5.0f));
    const VisionLoad load{true, 3};
    must_refuse([&] { (void)parse_generation_request("GENI 3 " + path.string() + " 1,248056,248056,248056,248056,2", 300000, 32); },
                "this engine was started without --vision", "GENI on a text engine");
    const auto request = parse_generation_request("GENI 3 " + path.string() + " 1,248056,248056,248056,248056,2", 300000, 32, load);
    require(request.prompt.size() == 6 && request.embeddings && request.image_rows.size() == 6, "GENI request shapes");
    require(request.image_rows[0] == -1 && request.image_rows[5] == -1 && request.image_rows[1] == 0 &&
            request.image_rows[4] == 3, "GENI image rows");
    require(request.positions.size() == (uint64_t(32) + 64) * 3, "GENI position table is max_context + 64 cells");
    const int32_t expected[][3] = {{1, 1, 1}, {1, 2, 1}, {1, 2, 2}, {3, 3, 3}};
    const uint64_t at[] = {1, 3, 4, 5};
    for (int i = 0; i < 4; ++i)
        for (int stream = 0; stream < 3; ++stream)
            require(request.positions[at[i] * 3 + stream] == expected[i][stream], "GENI planned position");
    require(request.images.size() == 1 && request.images[0].start == 1, "GENI picture list");
    // The picture identity is the grid then the embedding bytes; the fixture's floats are 5..16.
    float rows[12];
    for (int i = 0; i < 12; ++i) rows[i] = 5.0f + float(i);
    require(request.images[0].key == test_fnv(rows, sizeof rows, grid_hash(4, 2, 2)), "GENI embedding identity");
    const auto keyed = parse_generation_request("GENI 3 img_keys=0123456789abcdef " + path.string() + " 1,248056,248056,248056,248056,2", 300000, 32, load);
    const uint64_t server_key = 0x0123456789abcdefull;
    require(keyed.images.size() == 1 && keyed.images[0].key == test_fnv(&server_key, sizeof server_key, grid_hash(4, 2, 2)),
            "GENI server img_keys identity");
    const auto malformed = parse_generation_request("GENI 3 img_keys=xyz " + path.string() + " 1,248056,248056,248056,248056,2", 300000, 32, load);
    require(malformed.images.size() == 1 && malformed.images[0].key == request.images[0].key, "malformed img_keys ignored");
    must_refuse([&] { (void)parse_generation_request("GENI 3 " + path.string() + " 1,248056,248056,248056,248056", 300000, 32, load); },
                "the prompt cannot end in an image", "GENI ends in a picture");
    must_refuse([&] { (void)parse_generation_request("GENI 3 " + path.string() + " 1,2", 300000, 32, VisionLoad{true, 4}); },
                "bad embeddings file (expected strata-vision records of width 4)", "GENI record width");
    const auto text = parse_generation_request("GEN 3 1,2,3", 300000, 32, load);
    require(!text.embeddings && text.image_rows.empty() && text.positions.empty() && text.images.empty(),
            "text request carries no picture state");
}

}  // namespace

int main() {
    reader_cases();
    plan_cases();
    refusal_cases();
    text_and_rope_cases();
    installed_rope_cases();
    request_cases();
    if (failures) { std::printf("vision_prompt FAIL %d\n", failures); return 1; }
    std::printf("vision_prompt PASS\n");
    return 0;
}
