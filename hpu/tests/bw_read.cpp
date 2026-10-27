// TPC read-bandwidth and launch-overhead diagnostic (hpu/tpc/bw_read_probe.c). The resident IQ4 GEMV saturates near
// 1.2 TB/s of *logical* weight bytes while HBM peak is ~2.45 TB/s, and that estimate cannot say which limit we hit.
// This program reads exactly members*vectors*256 bytes per launch, so every GB/s below is a hardware rate.
//
// Sweeps and questions, in the order the numbers answer them:
//   default            chains x ops x members: requests in flight versus instruction issue (bw-chains-512miB.json).
//   --total-mib N      the same at another footprint; a small footprint turns the run into a launch-cost measurement.
//   --launch-attribution  is the fixed ~14 µs per launch un-warmed work, a device-side per-graph cost, or the host's
//                      own submit rate? Times single launches cold (each after a full device sync, no warm launch at
//                      all), then queue depth 1..1024 comparing device ms/launch with wall ms/launch. Enqueues
//                      Recipe::enqueue directly so the measurement path has no per-launch event and no HostRecipe.
//   --node-attribution  is that cost charged per GRAPH or per NODE? Same total work as 1 node versus 4 nodes inside
//                      one graph (the kernel's `offset` splits the buffer), timed back to back.
// Hardware only, not registered in CTest: needs a card.
//   strata-hpu-bw-test [--total-mib 512] [--iters 256] [--launch-attribution] [--node-attribution]
#include "strata/hpu/recipe.hpp"
#include "strata/hpu/host_recipe.hpp"
#include "strata/hpu/model_abi.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
using namespace strata::hpu;
namespace {
constexpr double kPeakGBs=2450.0; // printed only as an efficiency column
constexpr int kMaxMembers=4096;

struct TimedEvent {
    synEventHandle handle=nullptr;
    explicit TimedEvent(const Device& device) {check(synEventCreate(&handle,device.info().id,EVENT_COLLECT_TIME),"timing event create");}
    ~TimedEvent() {if(handle) (void)synEventDestroy(handle);}
};

// One graph, `nodes` copies of the probe node, the whole buffer split between them by `offset`. Each node gets its
// own sink tensor (a distinct slice of one device buffer) so the test measures per-node cost without depending on
// whether Synapse tolerates two writers on one tensor.
struct Graph {std::unique_ptr<Recipe> recipe;std::vector<Binding> bindings;};

Graph compile(const Device& device,const DeviceBuffer& source,const DeviceBuffer& sink,uint64_t total_vectors,
              int members,int nodes,int chains,int ops,const std::string& tag) {
    if(members%nodes) throw std::invalid_argument("members must divide by the node count");
    GraphBuilder graph(device);
    const auto source_id=graph.tensor({"source",DType::F32,{total_vectors*64},true});
    const int part=members/nodes;
    Graph out;out.bindings.push_back({"source",&source});
    for(int node=0;node<nodes;++node) {
        const std::string name="sink"+std::to_string(node);
        const auto sink_id=graph.tensor({name,DType::F32,{uint64_t(part)*64},true});
        out.bindings.push_back({name,&sink,uint64_t(node)*part*64*4});
        const abi::BwProbe a{part,int32_t(total_vectors/uint64_t(members)),chains,ops,node*part};
        const TensorId in[]={source_id},one[]={sink_id};
        graph.op("strata_bw_read_probe_f32",in,one,&a,sizeof(a));
    }
    out.recipe=graph.compile("strata_bw_"+tag);
    return out;
}

double device_ms(const Stream& stream,TimedEvent& start,TimedEvent& end,const std::function<void()>& body) {
    check(synEventRecord(start.handle,stream.handle()),"timing start record");
    body();
    check(synEventRecord(end.handle,stream.handle()),"timing end record");
    check(synEventSynchronize(end.handle),"timing end wait");
    uint64_t ns=0;check(synEventElapsedTime(&ns,start.handle,end.handle),"timing event elapsed");
    if(!ns) throw std::runtime_error("zero device interval");
    return double(ns)/1e6;
}

double median(std::vector<double> values) {std::sort(values.begin(),values.end());return values[values.size()/2];}

void sweep(const Device& device,const DeviceBuffer& source,const DeviceBuffer& sink,uint64_t total_vectors,
           int members,int chains,int ops,int iterations) {
    auto graph=compile(device,source,sink,total_vectors,members,1,chains,ops,"sweep");
    const std::span<const Binding> bindings(graph.bindings);
    Stream stream(device);graph.recipe->enqueue(stream,bindings);stream.synchronize(); // initialize once
    TimedEvent start(device),end(device);std::vector<double> samples;
    for(int pass=0;pass<5;++pass) {
        const double ms=device_ms(stream,start,end,[&]{for(int i=0;i<iterations;++i) graph.recipe->enqueue(stream,bindings);});
        samples.push_back(ms/iterations);
    }
    const double ms=median(samples),bytes=double(total_vectors)*256.0;
    std::cout<<chains<<' '<<ops<<' '<<members<<' '<<total_vectors/members<<"   "<<ms<<"  "
             <<bytes/1e6/ms<<"  "<<100.0*bytes/1e6/ms/kPeakGBs<<'\n';
}
}

int main(int argc,char** argv) {
    try {
        uint64_t total_mib=512;int iterations=256;bool launch_probe=false,node_probe=false;
        for(int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if(arg=="--total-mib" && i+1<argc) total_mib=std::stoull(argv[++i]);
            else if(arg=="--iters" && i+1<argc) iterations=std::stoi(argv[++i]);
            else if(arg=="--launch-attribution") launch_probe=true;
            else if(arg=="--node-attribution") node_probe=true;
            else {std::cerr<<"usage: bw-read-test [--total-mib N] [--iters N] [--launch-attribution] [--node-attribution]\n";return 2;}
        }
        Device device;
        const uint64_t bytes=total_mib<<20,total_vectors=bytes/256;
        if(total_vectors%kMaxMembers || (total_vectors/kMaxMembers)%8)
            throw std::invalid_argument("total size must divide by 4096 members and 8 chains");
        DeviceBuffer source(device,bytes),sink(device,uint64_t(kMaxMembers)*64*4);
        { // any non-zero pattern: the value only feeds the probe's own adds, never a host comparison
            HostBuffer host(device,1u<<20);auto* raw=static_cast<uint8_t*>(host.data());
            for(size_t i=0;i<host.size();++i) raw[i]=uint8_t((i*37)&0x7F);
            Stream copy(device);
            for(uint64_t at=0;at<bytes;at+=host.size()) upload(copy,source,host,host.size(),at);
            std::memset(host.data(),0,host.size());upload(copy,sink,host,sink.size());copy.synchronize();
        }
        std::cout<<"bw probe: "<<total_mib<<" MiB read per launch at 256-byte granularity; bytes are exactly what "
                 <<"the kernel touches; members 1024, chains 8 unless a row says otherwise\n";
        if(!launch_probe && !node_probe) {
            std::cout<<"chains ops members vectors   ms        GB/s     %peak\n";
            for(const int chains:{1,2,4,8}) for(const int ops:{0,7})
                sweep(device,source,sink,total_vectors,1024,chains,ops,iterations);
            for(const int members:{16,64,256,4096}) sweep(device,source,sink,total_vectors,members,8,0,iterations);
        }
        if(launch_probe) {
            // Small footprint so the fixed cost is not hidden by work: 8 MiB is 3.8 µs of reading at 2.2 TB/s.
            const uint64_t small=total_vectors/64; // 8 MiB at the default size
            std::cout<<"\nlaunch attribution, 8 MiB graph. Cold single launches, each after a full device sync, "
                     <<"NEVER warmed before launch 1:\n";
            {
                auto graph=compile(device,source,sink,small,1024,1,8,0,"cold");
                const std::span<const Binding> bindings(graph.bindings);
                Stream stream(device);TimedEvent start(device),end(device);
                for(int i=0;i<8;++i) {
                    device.synchronize();
                    const double ms=device_ms(stream,start,end,[&]{graph.recipe->enqueue(stream,bindings);});
                    std::cout<<"  launch "<<i+1<<": "<<ms<<" ms\n";
                }
            }
            std::cout<<"queue depth (steady state, median of 5): device ms/launch versus host wall ms/launch\n";
            auto graph=compile(device,source,sink,small,1024,1,8,0,"depth");
            const std::span<const Binding> bindings(graph.bindings);
            Stream stream(device);graph.recipe->enqueue(stream,bindings);stream.synchronize();
            TimedEvent start(device),end(device);
            for(const int depth:{1,2,4,8,16,64,256,1024}) {
                std::vector<double> device_samples,wall_samples;
                for(int pass=0;pass<5;++pass) {
                    device.synchronize();
                    const auto host_start=std::chrono::steady_clock::now();
                    const double ms=device_ms(stream,start,end,[&]{
                        for(int i=0;i<depth;++i) graph.recipe->enqueue(stream,bindings);});
                    wall_samples.push_back(std::chrono::duration<double,std::milli>(
                        std::chrono::steady_clock::now()-host_start).count()/depth);
                    device_samples.push_back(ms/depth);
                }
                const double d=median(device_samples),w=median(wall_samples);
                std::cout<<"  depth "<<depth<<": device "<<d<<" ms/launch  wall "<<w<<" ms/launch  host_is_limit="
                         <<(w>d*0.9?"yes":"no")<<'\n';
            }
        }
        if(node_probe) {
            const uint64_t small=total_vectors/64;
            std::cout<<"\nnode attribution, 8 MiB of reading per graph, same total work split across nodes\n";
            std::cout<<"nodes device_ms  ms_per_extra_node\n";
            double previous=0;
            for(const int nodes:{1,2,4,8}) {
                auto graph=compile(device,source,sink,small,1024,nodes,8,0,"nodes"+std::to_string(nodes));
                const std::span<const Binding> bindings(graph.bindings);
                Stream stream(device);graph.recipe->enqueue(stream,bindings);stream.synchronize();
                TimedEvent start(device),end(device);std::vector<double> samples;
                for(int pass=0;pass<5;++pass) samples.push_back(device_ms(stream,start,end,
                    [&]{for(int i=0;i<256;++i) graph.recipe->enqueue(stream,bindings);})/256.0);
                const double ms=median(samples);
                std::cout<<nodes<<"  "<<ms<<"  "<<(previous?ms-previous:0.0)<<'\n';
                previous=ms;
            }
        }
        std::cout<<"PASS bw_read\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
