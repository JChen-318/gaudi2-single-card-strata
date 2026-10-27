#include "strata/hpu/runtime.hpp"
#include "strata/hpu/kernel_path.hpp"
#include "strata/hpu/performance_modes.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <mutex>
#include <unistd.h>
#include <utility>

namespace strata::hpu {
namespace {
// Installed layouts move the backend out of its build tree. The compiled-in kernel path names the build's own
// libstrata_hpu_tpc.so, which an installed prefix cannot reach; the fallback rule itself is the pure, host-tested
// function in strata/hpu/kernel_path.hpp, and this only supplies the directory the executable lives in.
std::string executable_directory() {
    std::error_code error;
    const auto self = std::filesystem::read_symlink("/proc/self/exe", error);
    if (error || self.empty()) return {};
    return self.parent_path().string();
}
}  // namespace
void configure_kernel_libraries() {
    const char* existing = std::getenv("GC_KERNEL_PATH");
    std::string path = existing ? existing : "";
    const auto directory = executable_directory();
    std::istringstream required(STRATA_HPU_DEFAULT_KERNEL_PATH);
    std::string item;
    while (std::getline(required, item, ':')) {
        if (item.empty()) continue;
        item = resolve_kernel_entry(item, directory, STRATA_HPU_KERNEL_NAME);
        std::istringstream configured(path);
        std::string candidate;
        bool found = false;
        while (std::getline(configured, candidate, ':')) if (candidate == item) found = true;
        if (!found) { if (!path.empty()) path += ':'; path += item; }
    }
    if (setenv("GC_KERNEL_PATH", path.c_str(), 1) != 0)
        throw std::runtime_error("cannot configure GC_KERNEL_PATH for the installed SDK");
}
void check(synStatus status, const char* operation) {
    if (status == synSuccess) return;
    char description[256] = {};
    (void)synStatusGetBriefDescription(status, description, sizeof(description));
    throw std::runtime_error(std::string(operation) + ": " + description +
                             " (Synapse status " + std::to_string(status) + ")");
}
namespace {
std::mutex lifecycle_mutex;
bool acquired_session = false;
void cleanup(synStatus status, const char* operation) noexcept {
    if (status != synSuccess)
        std::fprintf(stderr, "strata-hpu: cleanup %s failed (status %d)\n", operation, status);
}
}
namespace detail {
struct DeviceState {
    synDeviceId id = 0;
    AllocationBudget hbm,pinned;
    explicit DeviceState(std::optional<uint32_t> module_id) {
        std::lock_guard lock(lifecycle_mutex);
        if (acquired_session)
            throw std::logic_error("one acquired Device per process; copy the existing Device to share it");
        // Frameworks normally set this. A standalone recipe also needs the NOP/TPC
        // kernels, even when its visible compute node is an MME GEMM.
        configure_kernel_libraries();
        // This standalone backend owns Synapse. It does not share a process with torch.
        check(synInitialize(), "synInitialize (one acquired Device per process)");
        bool acquired = false;
        try {
            if (module_id)
                check(synDeviceAcquireByModuleId(&id, *module_id), "synDeviceAcquireByModuleId");
            else
                check(synDeviceAcquireByDeviceType(&id, synDeviceGaudi2), "synDeviceAcquireByDeviceType Gaudi2");
            acquired = true;
            synDeviceInfo info{};
            check(synDeviceGetInfo(id, &info), "synDeviceGetInfo");
            if (info.deviceType != synDeviceGaudi2)
                throw std::runtime_error("this backend currently supports Gaudi2 only");
            acquired_session = true;
        } catch (...) {
            if (acquired) cleanup(synDeviceRelease(id), "device release during initialization");
            cleanup(synDestroy(), "Synapse shutdown during initialization");
            throw;
        }
    }
    ~DeviceState() {
        std::lock_guard lock(lifecycle_mutex);
        cleanup(synDeviceSynchronize(id), "device synchronize");
        cleanup(synDeviceRelease(id), "device release");
        cleanup(synDestroy(), "Synapse shutdown");
        acquired_session = false;
    }
};
}
namespace {
void same_device(const std::shared_ptr<detail::DeviceState>& a,
                 const std::shared_ptr<detail::DeviceState>& b) {
    if (!a || a != b) throw std::invalid_argument("resources belong to different devices");
}
bool drain(const std::shared_ptr<detail::DeviceState>& state) noexcept {
    const auto status = synDeviceSynchronize(state->id);
    cleanup(status, "drain before release");
    return status == synSuccess;
}
}

Device::Device(std::optional<uint32_t> module_id) : state_(std::make_shared<detail::DeviceState>(module_id)) {}
DeviceInfo Device::info() const {
    DeviceInfo value;
    value.id = state_->id;
    char name[256] = {}, driver[256] = {};
    check(synDeviceGetName(name, sizeof(name), state_->id), "synDeviceGetName");
    check(synDriverGetVersion(driver, sizeof(driver)), "synDriverGetVersion");
    check(synDeviceGetMemoryInfo(state_->id, &value.free_bytes, &value.total_bytes), "synDeviceGetMemoryInfo");
    value.name = name;
    value.driver = driver;
    return value;
}
void Device::synchronize() const { check(synDeviceSynchronize(state_->id), "synDeviceSynchronize"); }
void Device::allocation_limits(uint64_t hbm,uint64_t pinned) const {
    if(!hbm || !pinned || state_->hbm.stats().used || state_->pinned.stats().used) throw std::logic_error("allocation limits must precede all buffers");
    state_->hbm.configure(hbm);state_->pinned.configure(pinned);
}
AllocationStats Device::hbm_allocations() const {return state_->hbm.stats();}
AllocationStats Device::pinned_allocations() const {return state_->pinned.stats();}

DeviceBuffer::DeviceBuffer(const Device& device, uint64_t bytes) : state_(device.state()), bytes_(bytes) {
    state_->hbm.charge(bytes);
    try {if (bytes) check(synDeviceMalloc(state_->id, bytes, 0, 0, &address_), "synDeviceMalloc");}
    catch(...) {state_->hbm.release(bytes);throw;}
}
DeviceBuffer::~DeviceBuffer() { reset(); }
void DeviceBuffer::reset() noexcept {
    if (address_ && drain(state_)) {
        const auto status=synDeviceFree(state_->id,address_,0);cleanup(status,"device free");
        if(status==synSuccess) state_->hbm.release(bytes_);
    }
    address_ = 0;
    bytes_ = 0;
}
DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : state_(std::move(other.state_)), address_(std::exchange(other.address_, 0)),
      bytes_(std::exchange(other.bytes_, 0)) {}
DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        state_ = std::move(other.state_);
        address_ = std::exchange(other.address_, 0);
        bytes_ = std::exchange(other.bytes_, 0);
    }
    return *this;
}
HostBuffer::HostBuffer(const Device& device, uint64_t bytes) : state_(device.state()), bytes_(bytes) {
    state_->pinned.charge(bytes);
    try {if (bytes) check(synHostMalloc(state_->id, bytes, 0, &data_), "synHostMalloc");}
    catch(...) {state_->pinned.release(bytes);throw;}
}
HostBuffer::~HostBuffer() {
    if (data_ && drain(state_)) {
        const auto status=synHostFree(state_->id,data_,0);cleanup(status,"host free");
        if(status==synSuccess) state_->pinned.release(bytes_);
    }
}
Stream::Stream(const Device& device) : state_(device.state()) {
    check(synStreamCreateGeneric(&handle_, state_->id, 0), "synStreamCreateGeneric");
}
Stream::~Stream() {
    if (handle_) {
        cleanup(synStreamSynchronize(handle_), "stream synchronize");
        cleanup(synStreamDestroy(handle_), "stream destroy");
    }
}
void Stream::synchronize() const { check(synStreamSynchronize(handle_), "synStreamSynchronize"); }
void Stream::wait(const Event& event) const {
    same_device(state_, event.state());
    if (!event.recorded()) throw std::logic_error("cannot wait for an unrecorded event");
    check(synStreamWaitEvent(handle_, event.handle(), 0), "synStreamWaitEvent");
}
Event::Event(const Device& device) : state_(device.state()) {
    check(synEventCreate(&handle_, state_->id, 0), "synEventCreate");
}
Event::~Event() {
    if (handle_) {
        if (recorded_) cleanup(synEventSynchronize(handle_), "event synchronize");
        cleanup(synEventDestroy(handle_), "event destroy");
    }
}
void Event::record(const Stream& stream) {
    same_device(state_, stream.state());
    check(synEventRecord(handle_, stream.handle()), "synEventRecord");
    recorded_ = true;
}
void Event::synchronize() const {
    if (!recorded_) throw std::logic_error("cannot synchronize an unrecorded event");
    if(event_poll_enabled()) {
        // Short decode graphs often finish before the interrupt-based wait wakes the host.
        // Bound active polling so long kernels/copies still use the normal SDK wait.
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::microseconds(500);
        do {
            const auto status=synEventQuery(handle_);
            if(status!=synBusy) {check(status,"synEventQuery during event wait");return;}
        } while(std::chrono::steady_clock::now()<deadline);
    }
    check(synEventSynchronize(handle_), "synEventSynchronize");
}
bool Event::ready() const {
    if (!recorded_) throw std::logic_error("cannot query an unrecorded event");
    const auto status = synEventQuery(handle_);
    if (status == synBusy) return false;
    check(status, "synEventQuery");
    return true;
}
void upload(const Stream& stream, DeviceBuffer& dst, const HostBuffer& src, uint64_t bytes,
            uint64_t device_offset, uint64_t host_offset) {
    same_device(stream.state(), dst.state());
    same_device(stream.state(), src.state());
    check_range(dst.size(), device_offset, bytes);
    check_range(src.size(), host_offset, bytes);
    if (!bytes) return;
    const auto source = reinterpret_cast<uint64_t>(src.data()) + host_offset;
    check(synMemCopyAsync(stream.handle(), source, bytes, dst.address() + device_offset, HOST_TO_DRAM),
          "synMemCopyAsync H2D");
}
void zero(const Stream& stream, DeviceBuffer& dst, uint64_t bytes) {
    same_device(stream.state(), dst.state());
    check_range(dst.size(), 0, bytes);
    if (bytes % 4) throw std::invalid_argument("device zero fill needs whole 32-bit words");
    if (!bytes) return;
    check(synMemsetD32Async(dst.address(), 0, bytes / 4, stream.handle()), "synMemsetD32Async");
}
void download(const Stream& stream, HostBuffer& dst, const DeviceBuffer& src, uint64_t bytes,
              uint64_t host_offset, uint64_t device_offset) {
    same_device(stream.state(), dst.state());
    same_device(stream.state(), src.state());
    check_range(dst.size(), host_offset, bytes);
    check_range(src.size(), device_offset, bytes);
    if (!bytes) return;
    const auto target = reinterpret_cast<uint64_t>(dst.data()) + host_offset;
    check(synMemCopyAsync(stream.handle(), src.address() + device_offset, bytes, target, DRAM_TO_HOST),
          "synMemCopyAsync D2H");
}
} // namespace strata::hpu
