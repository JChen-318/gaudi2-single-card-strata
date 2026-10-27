#include "strata/hpu/session_memory.hpp"
#include <iostream>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
}
int main() {
    try {
        std::optional<uint64_t> free;bool captured=false,published=false,applied=false;unsigned probes=0;
        SessionMemoryProbe probe=[&] {++probes;return free;};
        auto transaction=[&](bool load,bool pressure) {
            check_session_memory(400,100,probe);captured=true;
            // A complete file parse/CRC or snapshot capture has now succeeded.
            if(pressure) free=99;
            check_session_memory(0,100,probe);
            if(load) applied=true;else published=true;
        };
        for(auto bytes:{std::optional<uint64_t>{},std::optional<uint64_t>{0},std::optional<uint64_t>{499}}) {
            free=bytes;bool refused=false;try {transaction(false,false);} catch(const SessionMemoryRefused&) {refused=true;}
            require(refused && !captured && !published,"session allocated/published below allocation plus RAM floor");
        }
        free=500;transaction(false,false);require(captured && published,"exact session allocation plus floor refused");
        captured=published=false;free=500;bool refused=false;try {transaction(true,true);} catch(const SessionMemoryRefused&) {refused=true;}
        require(refused && captured && !applied && !published,"pressure after parse applied an unsafe session");
        free=500;transaction(true,false);require(applied,"restoration with sufficient final RAM refused");
        free=UINT64_MAX;check_session_memory(UINT64_MAX,0,probe);
        refused=false;try {check_session_memory(UINT64_MAX,1,probe);} catch(const SessionMemoryRefused&) {refused=true;}require(refused,"session memory addition overflow accepted");
        free={};refused=false;try {check_session_memory(0,0,probe);} catch(const SessionMemoryRefused&) {refused=true;}require(refused,"unknown telemetry treated as a zero floor bypass");
        require(probes==12,"session memory probe pre/post contract changed");
        std::cout<<"PASS session memory admission before allocation and before publish/apply, missing/zero telemetry, exact floor, overflow and pressure after capture\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
