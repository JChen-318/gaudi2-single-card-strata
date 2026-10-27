#include "strata/hpu/routing_capture.hpp"
#include <iostream>
#include <limits>
using namespace strata::hpu;
namespace {
void need(bool ok,const char* text) {if(!ok) throw std::runtime_error(text);}
template<class F> void rejects(F f) {bool failed=false;try {f();} catch(const std::exception&) {failed=true;}need(failed,"invalid routing capture accepted");}
Routes routes(uint32_t rows,uint32_t layer) {
    Routes out;out.tokens=rows;out.experts=16;out.top_k=3;
    for(uint32_t row=0;row<rows;++row) for(uint32_t k=0;k<3;++k) {out.ids.push_back(int32_t((layer*3+row+k)%16));out.weights.push_back(float(row*3+k+1)/512);}
    return out;
}
void fill(RoutingCapture& trace,uint32_t rows) {for(uint32_t layer=0;layer<3;++layer) trace.capture(layer,routes(rows,layer));}
void observer_failures() {
    RoutingCapture trace;trace.configure(3,2,16,3);const auto storage=trace.storage_bytes();
    for(int fault=0;fault<3;++fault) {
        {auto previous=trace.round(2);fill(trace,2);previous.finish();}
        need(trace.records().size()==6,"previous committed round missing");
        bool failed=false;
        try {
            auto round=trace.round(1);fill(trace,1);round.finish();
            need(trace.rows()==1,"finished routing not visible to synchronous observer");
            if(fault==0) throw std::runtime_error("post-commit routing IO");
            if(fault==1) throw Cancelled();
            throw SessionFatal("post-commit observer contract");
        } catch(const std::exception&) {failed=true;}
        need(failed && trace.records().empty() && !trace.physical_view().layers && trace.storage_bytes()==storage,"post-finish observer exception leaked current/prior routing or changed storage");
        {auto next=trace.round(1);fill(trace,1);next.finish();}
        need(trace.records().size()==3,"observer failure prevented subsequent capture");
    }
    {
        auto round=trace.round(2);fill(trace,2);round.finish();
        try {throw std::runtime_error("handled observer retry");} catch(const std::runtime_error&) {}
    }
    need(trace.rows()==2 && trace.records().size()==6,"internally handled error erased successful capture");
    // A successful capture can be made by a cleanup callback while some
    // OTHER exception is already unwinding. Only a newly escaping error may
    // invalidate it; checking uncaught_exceptions()!=0 is incorrect here.
    for(bool fail_inside:{false,true}) {
        bool during_unwind=false,caught_inside=false;uint32_t final_rows=99;
        struct Cleanup {
            RoutingCapture& trace;bool fail;bool& during;bool& caught;uint32_t& rows;
            ~Cleanup() noexcept {
                during=std::uncaught_exceptions()>0;
                try {auto round=trace.round(1);fill(trace,1);round.finish();if(fail) throw std::runtime_error("new nested observer failure");}
                catch(const std::exception&) {caught=true;}
                rows=trace.rows();
            }
        };
        try {Cleanup cleanup{trace,fail_inside,during_unwind,caught_inside,final_rows};throw std::runtime_error("original caller error");}
        catch(const std::runtime_error& error) {need(std::string(error.what())=="original caller error","capture cleanup replaced the caller error");}
        need(during_unwind && caught_inside==fail_inside && final_rows==(fail_inside?0u:1u),"round ignored its entry exception count during nested unwind");
    }
    RoutingCapture off;
    try {auto round=off.round(999);round.finish();throw std::runtime_error("disabled observer");} catch(const std::runtime_error&) {}
    need(!off.enabled() && !off.storage_bytes() && off.records().empty(),"disabled unwind allocated routing storage");
    std::cout<<"PASS routing post-finish IO/cancel/fatal cleanup, prior-trace exclusion/recovery, handled errors, pre-existing/nested unwind and disabled capture; no numeric rollback claim\n";
}
}
int main() {
    try {
        RoutingCapture off;off.capture(999,{});{auto round=off.round(999);round.finish();}
        need(!off.enabled() && !off.storage_bytes() && off.records().empty(),"disabled routing allocated or recorded");
        for(uint32_t maximum:{1u,2u,7u,128u}) {
            RoutingCapture trace;trace.configure(3,maximum,16,3);const auto capacity=trace.storage_bytes();
            need(capacity==uint64_t(3)*maximum*3*8 && capacity<RoutingCapture::host_bytes(3,maximum,16,3),"capture storage outside admitted envelope");
            for(uint32_t repeat=0;repeat<6;++repeat) {
                const auto rows=repeat%2?1:maximum;
                auto round=trace.round(rows);
                for(uint32_t layer=0;layer<3;++layer) trace.capture(layer,routes(rows,layer));
                // Native rejected-prefix replay occurs before round finish.
                // Its scalar records must not append/replace the initial pass.
                for(uint32_t layer=0;layer<3;++layer) trace.capture(layer,routes(1,layer+4));
                round.finish();
                const auto records=trace.records();need(records.size()==3*rows && trace.rows()==rows,"window routing records lost rows or duplicated replay");
                for(uint32_t layer=0;layer<3;++layer) for(uint32_t row=0;row<rows;++row) {
                    const auto expected=routes(rows,layer);const auto& record=records[layer*rows+row];
                    need(record.layer==int32_t(layer) && record.ids.size()==3 && record.weights.size()==3,"wrong layer/record shape");
                    for(uint32_t k=0;k<3;++k) need(record.ids[k]==expected.ids[row*3+k] && record.weights[k]==expected.weights[row*3+k],"layer/token order or true router weights changed");
                }
                uint32_t visited=0;trace.visit([&](int32_t layer,std::span<const int32_t> ids,std::span<const float> weights){need(layer==int32_t(visited/rows) && ids.size()==3 && weights.size()==3,"borrowed routing view differs from export");++visited;});
                need(visited==records.size() && trace.storage_bytes()==capacity,"routing visits grew persistent storage");
            }
            const auto before=trace.records();rejects([&]{auto bad=trace.round(maximum+1);});need(trace.records().size()==before.size(),"invalid preflight destroyed prior trace");
            trace.clear();need(trace.records().empty() && trace.storage_bytes()==capacity,"clear leaked stale routing or grew buffers");
        }
        RoutingCapture trace;trace.configure(3,2,16,3);
        for(int fault=0;fault<8;++fault) {
            rejects([&] {
                auto round=trace.round(2);trace.capture(0,routes(2,0));auto bad=routes(2,1);
                switch(fault) {
                    case 0:bad.tokens=1;break;
                    case 1:bad.top_k=2;break;
                    case 2:bad.experts=15;break;
                    case 3:bad.ids.pop_back();break;
                    case 4:bad.weights.pop_back();break;
                    case 5:bad.ids[0]=16;break;
                    case 6:bad.weights[0]=std::numeric_limits<float>::infinity();break;
                    case 7:bad.weights[0]=-1;break;
                }
                trace.capture(1,bad);round.finish();
            });need(trace.records().empty(),"failed physical pass exposed old/partial routing");
        }
        rejects([&]{auto round=trace.round(1);trace.capture(1,routes(1,1));});
        bool fatal=false;try {auto round=trace.round(1);trace.capture(0,routes(1,0));round.finish();} catch(const SessionFatal&) {fatal=true;}
        need(fatal && trace.records().empty(),"missing target layer after physical pass was not fatal/cleared");
        rejects([&]{RoutingCapture::host_bytes(1,129,16,3);});rejects([&]{RoutingCapture::host_bytes(0,1,16,3);});rejects([&]{RoutingCapture::host_bytes(1,1,16,17);});
        {
            RoutingCapture wide;wide.configure(48,128,512,64);auto round=wide.round(128);
            Routes input;input.tokens=128;input.experts=512;input.top_k=64;
            for(uint32_t at=0;at<128*64;++at) {input.ids.push_back(int32_t(at%64));input.weights.push_back(1.0f/64);}
            for(uint32_t layer=0;layer<48;++layer) wide.capture(layer,input);
            round.finish();const auto records=wide.records();uint64_t held=wide.storage_bytes()+records.capacity()*sizeof(RoutingRecord);
            for(const auto& record:records) held+=record.ids.capacity()*sizeof(int32_t)+record.weights.capacity()*sizeof(float);
            need(records.size()==48*128 && held<=RoutingCapture::host_bytes(48,128,512,64),"persistent capture plus real exported vector capacities exceed admitted host envelope");
        }
        observer_failures();
        std::cout<<"PASS bounded physical-window routing capture, layer/token order, true weights, replay exclusion, export envelope, invalid/failure cleanup and disabled zero allocation\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
