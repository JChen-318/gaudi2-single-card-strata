// Hardware-only F32 steering; never run while another process owns the card.
#include "strata/hpu/control_graph.hpp"
#include "strata/hpu/host_recipe.hpp"
#include <cmath>
#include <iostream>
using namespace strata::hpu;
int main() {
    try {
        Device device;
        for(uint32_t hidden:{64u,2560u}) for(auto mode:{ControlVectorMode::Add,ControlVectorMode::Project}) {
            GraphBuilder graph(device);define_control_vector(graph,hidden,4,mode);
            HostRecipe recipe(device,graph.compile("strata_control_hardware"),{}, {"out"});
            std::vector<float> direction(hidden),input(uint64_t(hidden)*4),expected(input.size());
            direction[0]=.6f;direction[1]=.8f;
            for(size_t i=0;i<input.size();++i) input[i]=float(int(i%31)-15)/16;
            for(size_t row=0;row<4;++row) {
                const double dot=double(input[row*hidden])*.6+double(input[row*hidden+1])*.8;
                for(size_t column=0;column<hidden;++column) expected[row*hidden+column]=mode==ControlVectorMode::Add?input[row*hidden+column]+direction[column]:float(input[row*hidden+column]-5*dot*direction[column]);
            }
            recipe.set("residual",input);recipe.set("direction",direction);
            if(mode==ControlVectorMode::Project) {const std::array<float,1> scale{-5};recipe.set("negative_scale",scale);}
            recipe.run();const auto output=recipe.output("out");double peak=0;
            for(size_t i=0;i<input.size();++i) {const auto error=std::abs(double(output[i])-expected[i]);peak=std::max(peak,error);if(!std::isfinite(output[i]) || error>4e-4) throw std::runtime_error("HPU control vector differs from per-stream scalar oracle");}
            std::cout<<"PASS real HPU control vector hidden="<<hidden<<" mode="<<uint32_t(mode)<<" max_abs_error="<<peak<<'\n';
        }
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
