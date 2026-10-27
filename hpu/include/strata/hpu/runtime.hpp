#pragma once

#include "strata/hpu/tensor.hpp"
#include "strata/hpu/allocation_budget.hpp"
#include <synapse_api.h>
#include <memory>
#include <optional>
#include <string>

namespace strata::hpu {
namespace detail { struct DeviceState; }
void check(synStatus status, const char* operation);
// Process-local environment only; call before synInitialize, including offline compilation.
void configure_kernel_libraries();

struct DeviceInfo {
    std::string name, driver;
    uint32_t id = 0;
    uint64_t free_bytes = 0, total_bytes = 0;
};

// All child resources retain the acquired device and the process Synapse session.
// Acquire once per process; copy Device to share. No mixed torch/SDK ownership.
class Device {
public:
    // AIP index and physical module ID are different. Default to an available Gaudi2.
    explicit Device(std::optional<uint32_t> module_id = std::nullopt);
    DeviceInfo info() const;
    void synchronize() const;
    // Before any buffers are allocated. Caps requested native buffer bytes;
    // driver/compiler allocations require a separate free-memory margin.
    void allocation_limits(uint64_t hbm,uint64_t pinned) const;
    AllocationStats hbm_allocations() const;
    AllocationStats pinned_allocations() const;
    const std::shared_ptr<detail::DeviceState>& state() const { return state_; }
private:
    std::shared_ptr<detail::DeviceState> state_;
};

class DeviceBuffer {
public:
    DeviceBuffer(const Device& device, uint64_t bytes);
    ~DeviceBuffer();
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&&) noexcept;
    uint64_t address() const { return address_; }
    uint64_t size() const { return bytes_; }
    const auto& state() const { return state_; }
private:
    void reset() noexcept;
    std::shared_ptr<detail::DeviceState> state_;
    uint64_t address_ = 0, bytes_ = 0;
};

class HostBuffer {
public:
    HostBuffer(const Device& device, uint64_t bytes);
    ~HostBuffer();
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
    void* data() const { return data_; }
    uint64_t size() const { return bytes_; }
    const auto& state() const { return state_; }
private:
    std::shared_ptr<detail::DeviceState> state_;
    void* data_ = nullptr;
    uint64_t bytes_ = 0;
};

class Event;
class Stream {
public:
    explicit Stream(const Device& device);
    ~Stream();
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    void synchronize() const;
    void wait(const Event& event) const;
    synStreamHandle handle() const { return handle_; }
    const auto& state() const { return state_; }
private:
    std::shared_ptr<detail::DeviceState> state_;
    synStreamHandle handle_ = nullptr;
};

class Event {
public:
    explicit Event(const Device& device);
    ~Event();
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    void record(const Stream& stream);
    void synchronize() const;
    bool ready() const;
    bool recorded() const { return recorded_; }
    synEventHandle handle() const { return handle_; }
    const auto& state() const { return state_; }
private:
    std::shared_ptr<detail::DeviceState> state_;
    synEventHandle handle_ = nullptr;
    bool recorded_ = false;
};

// Enqueue only. Callers must not mutate a host source before its transfer event.
// Destruction conservatively drains the device before freeing memory (P0 safety).
void upload(const Stream&, DeviceBuffer&, const HostBuffer&, uint64_t bytes,
            uint64_t device_offset = 0, uint64_t host_offset = 0);
void download(const Stream&, HostBuffer&, const DeviceBuffer&, uint64_t bytes,
              uint64_t host_offset = 0, uint64_t device_offset = 0);
// Enqueue only: the first `bytes` (a multiple of 4) of a device buffer set to zero on the device.
void zero(const Stream&, DeviceBuffer&, uint64_t bytes);
} // namespace strata::hpu
