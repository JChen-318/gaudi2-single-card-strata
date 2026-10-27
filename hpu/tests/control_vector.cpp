#include "strata/hpu/control_vector.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
using namespace strata::hpu;
namespace {
void require(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
template<class F> void rejects(F&& call) {bool failed=false;try {call();} catch(const std::exception&) {failed=true;}require(failed,"invalid control vector accepted");}
void close(float a,float b) {require(std::abs(a-b)<4e-6,"control vector differs from independent oracle");}
}
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    try {
        const std::filesystem::path root(argv[1]);ControlVectorOptions options;
        options.files={{(root/"a.gguf").string(),2},{(root/"b.gguf").string(),-1}};
        ControlVectors add(options,4,64);require(!add.covers(0) && add.covers(1) && add.covers(2) && add.covers(3),"layer-0/range rule");
        close(add.direction(1)[0],5);close(add.direction(1)[1],10);close(add.direction(2)[2],3);
        std::vector<float> rows(128,1);add.apply_reference(1,rows);close(rows[0],6);close(rows[1],11);close(rows[64],6);close(rows[65],11);
        const auto old=rows;add.apply_reference(1,rows,false);require(rows==old,"disabled control vector changed a bit");add.apply_reference(0,rows);require(rows==old,"layer 0 steered");
        options.files={{(root/"a.gguf").string(),1}};options.mode=ControlVectorMode::Project;
        ControlVectors project(options,4,64);close(project.scale(1),5);close(project.direction(1)[0],.6f);close(project.direction(1)[1],.8f);
        rows.assign(128,0);rows[0]=1;rows[1]=2;rows[64]=-1;rows[65]=1;project.apply_reference(1,rows);
        close(rows[0],-5.6f);close(rows[1],-6.8f);close(rows[64],-1.6f);close(rows[65],.2f);
        options.single=2;options.first=1;options.last=3;ControlVectors single(options,4,64);rows.assign(64,0);rows[2]=3;
        single.apply_reference(1,rows);close(rows[2],-3);require(single.direction(1)[2]==1 && single.scale(3)==2,"single source not replicated across selected layers");
        require(project.signature()!=single.signature() && project.signature()!=add.signature() && ControlVectors(options,4,64).signature()==single.signature(),"resolved identity unstable or omitted settings");
        require(!single.file_identity().empty() && single.host_bytes()==(4*64+4)*4,"bounded control vector identity/host charge");
        for(const auto* name:{"wrong_arch","wrong_type","wrong_shape","nan","inf","duplicate","malformed","truncated","zero","overflow"}) {
            auto invalid=options;invalid.single=-1;invalid.files={{(root/(std::string(name)+".gguf")).string(),1}};
            rejects([&]{ControlVectors bad(invalid,4,64);});
        }
        auto invalid=options;invalid.single=0;rejects([&]{ControlVectors bad(invalid,4,64);});
        invalid=options;invalid.files.front().second=std::numeric_limits<float>::infinity();rejects([&]{ControlVectors bad(invalid,4,64);});
        invalid=options;invalid.first=3;invalid.last=2;rejects([&]{ControlVectors bad(invalid,4,64);});
        std::cout<<"PASS bounded original controlvector GGUF, multi-file FP32 sum, add/project/per-stream oracle, single source/range, resolved identity and malformed/overflow guards\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
