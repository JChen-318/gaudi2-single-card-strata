// STRATA bandwidth probe (hpu/tests/bw_read.cpp; hardware diagnostic, no model path uses this kernel).
// The first version read with a single accumulator per member and measured 365 GB/s - BELOW the resident IQ4
// GEMV's rate - because a TPC issues in order, so `acc += load` keeps exactly one request in flight per vector
// unit. The number that matters is therefore the curve against in-flight requests: `chains` register-resident
// independent accumulators, each fed by its own load, with all loads of a group issued before any of their math.
//   chains = 1, 2, 4 or 8 (chosen by graph params, so the group really is unrolled);
//   ops    = 0 -> one add per load (the memory ceiling);
//            7 -> ~8 vector instructions per 256-byte load, the ratio tpc/iq4q8_gemv_packed.c runs at.
// Reading ops=7 at the chain depth that saturates memory tells us whether the GEMV is limited by in-flight
// requests or by instruction issue - the two things the existing "logical GB/s" numbers cannot separate.
// Each member reads `vectors` 256-byte vectors (64 F32 lanes, the same burst the GEMV loads), so bytes per
// launch are exactly members*vectors*256.
static inline float64 mix(float64 acc,float64 value,int ops) {
    acc+=value;
    if(ops==7) {
        acc+=value;acc=v_f32_max_b(acc,value);acc+=value;acc=v_f32_min_b(acc,value);
        acc+=value;acc=v_f32_max_b(acc,value);acc+=value;
    }
    return acc;
}

void main(tensor source,tensor sink,int members,int vectors,int chains,int ops,int offset) {
    const int5 begin=get_index_space_offset(),end=begin+get_index_space_size();
    (void)members;
    for(int member=begin[0];member<end[0];++member) {
        // `offset` is in member units and applies to the shared source only: several nodes can split one weight
        // buffer inside a single graph, which is how tests/bw_read.cpp --node-attribution asks the per-graph versus
        // per-node question. Each node owns its own sink slice, so the store stays member-relative.
        const int base=(offset+member)*vectors;
        float64 total=0;
        if(chains==1) {
            float64 a=0;
            for(int i=0;i<vectors;++i) {
                int5 p={(base+i)*64,0,0,0,0};
                a=mix(a,v_f32_ld_tnsr_b(p,source),ops);
            }
            total=a;
        } else if(chains==2) {
            float64 a=0,b=0;
            for(int i=0;i+2<=vectors;i+=2) {
                int5 p={(base+i)*64,0,0,0,0};
                const float64 v0=v_f32_ld_tnsr_b(p,source);p[0]+=64;
                const float64 v1=v_f32_ld_tnsr_b(p,source);
                a=mix(a,v0,ops);b=mix(b,v1,ops);
            }
            total=a+b;
        } else if(chains==4) {
            float64 a=0,b=0,c=0,d=0;
            for(int i=0;i+4<=vectors;i+=4) {
                int5 p={(base+i)*64,0,0,0,0};
                const float64 v0=v_f32_ld_tnsr_b(p,source);p[0]+=64;
                const float64 v1=v_f32_ld_tnsr_b(p,source);p[0]+=64;
                const float64 v2=v_f32_ld_tnsr_b(p,source);p[0]+=64;
                const float64 v3=v_f32_ld_tnsr_b(p,source);
                a=mix(a,v0,ops);b=mix(b,v1,ops);c=mix(c,v2,ops);d=mix(d,v3,ops);
            }
            total=(a+b)+(c+d);
        } else {
            float64 a=0,b=0,c=0,d=0,e=0,f=0,g=0,h=0;
            for(int i=0;i+8<=vectors;i+=8) {
                int5 p={(base+i)*64,0,0,0,0};
                const float64 v0=v_f32_ld_tnsr_b(p,source);p[0]+=64;
                const float64 v1=v_f32_ld_tnsr_b(p,source);p[0]+=64;
                const float64 v2=v_f32_ld_tnsr_b(p,source);p[0]+=64;
                const float64 v3=v_f32_ld_tnsr_b(p,source);p[0]+=64;
                const float64 v4=v_f32_ld_tnsr_b(p,source);p[0]+=64;
                const float64 v5=v_f32_ld_tnsr_b(p,source);p[0]+=64;
                const float64 v6=v_f32_ld_tnsr_b(p,source);p[0]+=64;
                const float64 v7=v_f32_ld_tnsr_b(p,source);
                a=mix(a,v0,ops);b=mix(b,v1,ops);c=mix(c,v2,ops);d=mix(d,v3,ops);
                e=mix(e,v4,ops);f=mix(f,v5,ops);g=mix(g,v6,ops);h=mix(h,v7,ops);
            }
            total=((a+b)+(c+d))+((e+f)+(g+h));
        }
        int5 sp={member*64,0,0,0,0};
        v_f32_st_tnsr(sp,sink,total);
    }
}
