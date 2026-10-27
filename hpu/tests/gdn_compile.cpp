#include "strata/hpu/gdn.hpp"
#include <iostream>

using namespace strata::hpu;
namespace {
template<class F> void rejects(F&& call) {
    bool rejected = false;
    try { call(); } catch (const std::exception&) { rejected = true; }
    if (!rejected) throw std::runtime_error("invalid graph contract accepted");
}
strata::core::ModelGeometry geometry(int s, int hk, int hv) {
    strata::core::ModelGeometry g;
    g.ssm_state_size=s; g.ssm_k_heads=hk; g.ssm_v_heads=hv;
    g.ssm_conv_channels=s*(2*hk+hv); g.ssm_value_dim=s*hv;
    return g;
}
void compile(const strata::core::ModelGeometry& g, const char* mode, bool debug=false) {
    GraphBuilder graph(OfflineCompilation{});
    const std::string operation(mode);
    if (operation=="step") define_gdn_step(graph,g);
    else if (operation=="body") define_gdn_body(graph,g,debug);
    else define_gdn_block(graph,g);
    rejects([&] { (void)graph.compile("must_not_acquire_device"); });
    const auto name="strata_gdn_"+operation+"_"+std::to_string(g.ssm_state_size)+(debug?"_debug":"");
    const auto bytes=graph.compile_offline(name);
    rejects([&] { (void)graph.compile_offline(name); });
    std::cout << "PASS offline GDN " << mode << " S=" << g.ssm_state_size << " key_heads=" << g.ssm_k_heads
              << " value_heads=" << g.ssm_v_heads << " debug=" << debug
              << " workspace_bytes=" << bytes << " no_device_acquired\n";
}
void window(const strata::core::ModelGeometry& g,uint32_t rows) {
    GraphBuilder graph(OfflineCompilation{});define_gdn_window(graph,g,rows);
    const auto bytes=graph.compile_offline("strata_gdn_window_"+std::to_string(g.n_embd)+"_"+std::to_string(rows));
    std::cout<<"PASS offline GDN window H="<<g.n_embd<<" S="<<g.ssm_state_size<<" rows="<<rows
             <<" workspace_bytes="<<bytes<<" no_device_acquired\n";
}
}
int main() {
    bool initialized=false;
    try {
        configure_kernel_libraries();
        check(synInitialize(),"offline synInitialize"); initialized=true;
        compile(geometry(64,2,6),"step");
        compile(geometry(128,16,48),"step");
        compile(geometry(256,2,6),"step");
        compile(geometry(64,1,3),"body",true);
        compile(geometry(128,16,48),"body");
        compile(geometry(128,16,48),"body",true);
        compile(geometry(128,16,48),"block");
        auto small=geometry(64,2,6);small.n_embd=32;
        for(auto rows:{1u,2u,5u,8u}) window(small,rows);
        window(geometry(128,16,48),8);
        for(auto rows:{0u,9u,128u}) {GraphBuilder graph(OfflineCompilation{});rejects([&] {define_gdn_window(graph,small,rows);});}
        // The SDK row axis is reversed from the public logical row-major axis.
        // Invalid operations fail before a split/concat node can be created.
        {
            GraphBuilder graph(OfflineCompilation{});
            const auto matrix=graph.tensor({"matrix",DType::BF16,{2,64}});
            const auto row=graph.tensor({"row",DType::BF16,{1,64}});
            const auto wrong=graph.tensor({"wrong",DType::F32,{1,64}});
            const TensorId duplicate[]={row,row},bad_type[]={row,wrong},short_list[]={row};
            rejects([&] {graph.split_rows(matrix,duplicate);});rejects([&] {graph.split_rows(matrix,bad_type);});
            rejects([&] {graph.split_rows(matrix,short_list);});rejects([&] {graph.concatenate_rows(duplicate,matrix);});
            rejects([&] {graph.concatenate_rows(bad_type,matrix);});rejects([&] {graph.concatenate_rows(short_list,matrix);});
        }
        for (int kind=0;kind<3;++kind) {
            GraphBuilder graph(OfflineCompilation{});
            auto g=geometry(128,16,48);
            if (kind==0) g.ssm_conv_channels++;
            if (kind==1) g.rms_eps=0;
            if (kind==2) g.ssm_d_conv=1;
            rejects([&] { define_gdn_body(graph,g); });
        }
        check(synDestroy(),"offline synDestroy"); initialized=false;
    } catch (const std::exception& e) {
        if (initialized) (void)synDestroy();
        std::cerr << e.what() << '\n'; return 1;
    }
}
