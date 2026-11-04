#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "shmem.h"
#include "layout.hpp"
#include "plan.hpp"
#include "exp_guard.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" void launch_barrier(void*);
extern "C" void launch_demo(uint32_t,uint32_t,void*,const Demo::Args*);
static void Check(int code,const char* op) {
    if(code) throw std::runtime_error(std::string(op)+": "+std::to_string(code));
}
#define CHECK(call) Check((call),#call)
int main(int argc,char** argv) {
  try {
    std::map<std::string,std::string> o;
    for(int i=1;i<argc;i+=2) {
        if(i+1==argc || !o.emplace(argv[i],argv[i+1]).second) throw std::runtime_error("Invalid arguments");
    }
    auto number=[&](const char* key)->uint32_t {
        const auto& v=o.at(key);size_t used=0;auto n=std::stoull(v,&used);
        if(used!=v.size() || n>UINT32_MAX) throw std::runtime_error("Invalid integer");
        return n;
    };
    Demo::Args a{};
    a.rank=number("--rank");a.world=number("--ep-size");a.bs=number("--bs");
    a.hidden=number("--hidden");a.topk=number("--topk");a.intermediate=number("--intermediate");
    a.sharedExperts=number("--shared-experts");a.sharedIntermediate=number("--shared-intermediate");a.computeEnable=number("--compute-enable");
    a.communicationEnable=number("--communication-enable");
    a.completionNotify=number("--completion-notify");
    const uint32_t warmup=number("--warmup"),iters=number("--iters"),rounds=warmup+iters;
    if((a.world!=4 && a.world!=8 && a.world!=16) || a.rank>=a.world ||
       a.bs<1024 || a.bs>MegaMoeUrma::kMaxTokensPerRank || a.bs%128 || a.computeEnable>1 || a.communicationEnable>1 || a.completionNotify>1 || (!a.computeEnable && !a.communicationEnable) ||
       !iters || iters>1000 || warmup>1000 || a.hidden<256 || a.hidden>8192 || a.hidden%256 ||
       a.intermediate<256 || a.intermediate>16384 || a.intermediate%256 || a.topk<1 || a.topk>8 ||
       a.sharedExperts>1 || a.sharedIntermediate<256 || a.sharedIntermediate>16384 || a.sharedIntermediate%256)
        throw std::runtime_error("Invalid case/configuration");
    const uint32_t rows=a.bs*a.topk/a.world,cvs=(rows+255)/256;
    const uint64_t slot=uint64_t(rows)*a.hidden*2,storage=a.world*slot;
    const uint64_t scaleSlot=uint64_t(rows)*(a.hidden/32),scaleStorage=a.world*scaleSlot;
    const uint64_t dispatchStorage=storage/2+scaleStorage;
    const uint32_t recordCount=2*Demo::cores+a.world-1;
    const uint64_t recordBytes=uint64_t(rounds)*recordCount*Demo::words*8;
    const auto plan=Demo::MakePlan(a.bs,a.world,a.rank,a.hidden,a.intermediate,a.topk);
    if(plan.config.receiveRows!=uint64_t(a.bs)*a.topk)
        throw std::runtime_error("MFU requires the balanced per-card receive-row count");
    a.compute.config=plan.config;
    CHECK(aclInit(nullptr));CHECK(aclrtSetDevice(number("--device")));
    const char* soc=aclrtGetSocName();
    if(!soc || std::string(soc).find("950")==std::string::npos) throw std::runtime_error("Ascend950 required");
    aclrtStream stream{};CHECK(aclrtCreateStream(&stream));
    auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();
    if(!platform || platform->GetCoreNumAic()<28 || platform->GetCoreNumAiv()<56)
        throw std::runtime_error("Needs 28 AIC / 56 AIV");
    uint64_t ub=0,l1=0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1,l1);
    if(l1<512*1024 || ub<MegaMoeUrma::kRequiredUbBytes) throw std::runtime_error("Needs production L1/UB capacity");
    const auto blocks=platform->CalcTschBlockDim(56,28,56);
    aclshmemx_init_attr_t attr{};aclshmemx_uniqueid_t uid{};
    attr.my_pe=a.rank;attr.n_pes=a.world;attr.local_mem_size=std::max<uint64_t>(2ULL<<30,storage*2+scaleStorage+dispatchStorage+(64ULL<<20));
    if(o.at("--endpoint").size()>=sizeof(attr.ip_port)) throw std::runtime_error("Endpoint too long");
    std::memcpy(attr.ip_port,o.at("--endpoint").c_str(),o.at("--endpoint").size()+1);
    attr.comm_args=&uid;
    attr.option_attr.data_op_engine_type=static_cast<data_op_engine_type_t>(ACLSHMEM_DATA_OP_MTE|ACLSHMEM_DATA_OP_UDMA);
    attr.option_attr.shm_init_timeout=120;attr.option_attr.shm_create_timeout=120;
    attr.option_attr.control_operation_timeout=120;
    attr.runtime_config.udma_shared_jetty=ACLSHMEM_CONFIG_DISABLED;
    CHECK(aclshmemx_set_qp_num(ACLSHMEM_DATA_OP_UDMA,1));
    CHECK(aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT,&attr));
    for(uint32_t peer=0;peer<a.world;++peer) if(peer!=a.rank) {
        aclshmemx_net_layer_t layer{};
        CHECK(aclshmemx_get_qp_net_layer(ACLSHMEM_DATA_OP_UDMA,peer,0,&layer));
        std::cout<<"[ROUTE] rank="<<a.rank<<" peer="<<peer<<" qp=0 layer="<<uint32_t(layer)<<'\n';
    }
    auto* source=static_cast<uint8_t*>(aclshmem_malloc(storage));
    auto* sourceScale=static_cast<uint8_t*>(aclshmem_malloc(scaleStorage));
    auto* dispatch=static_cast<uint8_t*>(aclshmem_malloc(dispatchStorage));
    auto* destination=static_cast<uint8_t*>(aclshmem_malloc(storage));
    if(!source || !sourceScale || !dispatch || !destination) throw std::runtime_error("Symmetric allocation failed");
    std::vector<void*> owned;
    auto allocate=[&](uint64_t n,int pattern)->uint64_t {
        void* ptr=nullptr;CHECK(aclrtMalloc(&ptr,n,ACL_MEM_MALLOC_HUGE_FIRST));owned.push_back(ptr);
        CHECK(aclrtMemset(ptr,n,pattern,n));return uintptr_t(ptr);
    };
    auto upload=[&](const auto& values)->uint64_t {
        const uint64_t bytes=values.size()*sizeof(values[0]);
        uint64_t dst=allocate(bytes,0);
        CHECK(aclrtMemcpy(reinterpret_cast<void*>(dst),bytes,values.data(),bytes,ACL_MEMCPY_HOST_TO_DEVICE));
        return dst;
    };
    // All modes allocate exactly the same operands. These are independent of
    // communication buffers and of each other's producer outputs.
    auto& c=a.compute.config;auto& g=a.compute;
    const uint64_t r=c.receiveRows,h=a.hidden,i=a.intermediate,e=c.localExperts;
    c.receiveXOffset=0;c.returnYOffset=r*h;c.receiveScaleOffset=0;
    g.symmetric=allocate(r*h+r*h*2,0x38); // GMM1 input + private self-GMM2 return
    g.gatheredInputScale=allocate(r*(h/32),0x7f);
    g.activation=allocate(r*i,0x38);g.activationScale=allocate(r*(i/32),0x7f);
    g.weight1=allocate(e*h*2*i/2,0x11);g.weightScale1=allocate(e*(h/32)*2*i,0x7f);
    g.weight2=allocate(e*i*h/2,0x11);g.weightScale2=allocate(e*(i/32)*h,0x7f);
    g.gmm1Output=allocate(r*2*i*2,0);g.gmm2Output=allocate(r*h*2,0);
    g.sharedExperts=a.sharedExperts;g.sharedIntermediate=a.sharedIntermediate;
    if(a.sharedExperts) {
        const uint64_t si=a.sharedIntermediate;
        g.x=allocate(uint64_t(a.bs)*h,0x38);g.xScale=allocate(uint64_t(a.bs)*(h/32),0x7f);
        g.sharedWeight1=allocate(h*si,0x11);g.sharedScale1=allocate((h/32)*2*si,0x7f);
        g.sharedWeight2=allocate(si*h/2,0x11);g.sharedScale2=allocate((si/32)*h,0x7f);
        g.sharedGmm1=allocate(uint64_t(a.bs)*2*si*2,0);
        g.sharedActivation=allocate(uint64_t(a.bs)*si,0x38);g.sharedActivationScale=allocate(uint64_t(a.bs)*(si/32),0x7f);
        g.sharedOutput=allocate(uint64_t(a.bs)*h*2,0);
    }
    const uint64_t g1FlagBytes=uint64_t(c.gmm1Tiles)*MegaMoeUrma::kFlagBytes;
    const uint64_t g2FlagBytes=uint64_t(c.fragmentCount)*MegaMoeUrma::kFlagBytes;
    const uint64_t sharedFlagBytes=a.sharedExperts?uint64_t((a.bs+255)/256)*(a.sharedIntermediate/256)*MegaMoeUrma::kFlagBytes:0;
    g.gmm1Done=allocate(g1FlagBytes,0);g.gmm2Done=allocate(g2FlagBytes,0);
    if(sharedFlagBytes)g.sharedGmm1Done=allocate(sharedFlagBytes,0);
    g.fragments=upload(plan.fragments);g.segments=upload(plan.segments);
    g.computeBatches=upload(plan.computeBatches);g.computeTasks=upload(plan.computeTasks);
    auto* records=reinterpret_cast<void*>(allocate(recordBytes,0));
    a.workspace=allocate(platform->GetLibApiWorkSpaceSize(),0);a.ffts=shmemx_get_ffts_config();
    a.source=uintptr_t(source);a.sourceScale=uintptr_t(sourceScale);
    a.dispatchDestination=uintptr_t(dispatch);a.destination=uintptr_t(destination);
    g.systemWorkspace=a.workspace;g.ffts=a.ffts;
    std::vector<uint8_t> host(slot);
    for(uint32_t peer=0;peer<a.world;++peer) {
        for(uint64_t i=0;i<slot;++i)host[i]=Demo::Pattern(a.rank,peer,i);
        CHECK(aclrtMemcpy(source+peer*slot,slot,host.data(),slot,ACL_MEMCPY_HOST_TO_DEVICE));
    }
    for(uint32_t peer=0;peer<a.world;++peer) {
        for(uint64_t j=0;j<scaleSlot;++j)host[j]=Demo::Pattern(a.rank,peer,j);
        CHECK(aclrtMemcpy(sourceScale+peer*scaleSlot,scaleSlot,host.data(),scaleSlot,ACL_MEMCPY_HOST_TO_DEVICE));
    }
    CHECK(aclrtMemset(destination,storage,0xCD,storage));
    CHECK(aclrtMemset(dispatch,dispatchStorage,0xCD,dispatchStorage));
    ExpGuard guard;guard.Init(number("--exp-rows"),number("--exp-cols"),number("--exp-repeat"));guard.Prepare();
    std::cout<<"[CONFIG] rank="<<a.rank<<" compute+dequant="<<(a.computeEnable?"on":"off")
             <<" completion_notify="<<(a.completionNotify?"on":"off")<<" communication="<<(a.communicationEnable?"on":"off")<<" rows/peer="<<rows<<" CVs/peer="<<cvs<<" SO_flags=0"
             <<" dispatch_bytes/card="<<(a.world-1)*(slot/2+scaleSlot)
             <<" combine_bytes/card="<<(a.world-1)*slot
             <<" compute_input=independent-prepared L2=production-default"<<std::endl;
    aclshmem_barrier_all();launch_barrier(stream);guard.Enqueue(stream);
    for(uint32_t round=0;round<rounds;++round) {
        // Completion flags are write-only to this benchmark: no consumer waits.
        // Reset outside the measured kernel, identically in both variants.
        CHECK(aclrtMemsetAsync(reinterpret_cast<void*>(g.gmm1Done),g1FlagBytes,0,g1FlagBytes,stream));
        CHECK(aclrtMemsetAsync(reinterpret_cast<void*>(g.gmm2Done),g2FlagBytes,0,g2FlagBytes,stream));
        if(sharedFlagBytes)CHECK(aclrtMemsetAsync(reinterpret_cast<void*>(g.sharedGmm1Done),sharedFlagBytes,0,sharedFlagBytes,stream));
        launch_barrier(stream);
        a.records=uintptr_t(records)+uint64_t(round)*recordCount*Demo::words*8;
        launch_demo(blocks,ub,stream,&a);
        launch_barrier(stream);
    }
    CHECK(aclrtSynchronizeStream(stream));aclshmem_barrier_all();
    std::vector<uint64_t> hr(recordBytes/8);
    CHECK(aclrtMemcpy(hr.data(),recordBytes,records,recordBytes,ACL_MEMCPY_DEVICE_TO_HOST));
    for(uint32_t peer=0;peer<a.world;++peer) {
        CHECK(aclrtMemcpy(host.data(),slot,destination+peer*slot,slot,ACL_MEMCPY_DEVICE_TO_HOST));
        for(uint64_t i=0;i<slot;++i) if(host[i]!=((!a.communicationEnable || peer==a.rank)?0xCD:Demo::Pattern(peer,a.rank,i)))
            throw std::runtime_error("Payload mismatch peer="+std::to_string(peer)+" byte="+std::to_string(i));
        for(unsigned part=0;part<2;++part) {
            uint64_t n=part?scaleSlot:slot/2;
            uint8_t* ptr=dispatch+(part?storage/2:0)+peer*n;
            CHECK(aclrtMemcpy(host.data(),n,ptr,n,ACL_MEMCPY_DEVICE_TO_HOST));
            for(uint64_t j=0;j<n;++j)if(host[j]!=((!a.communicationEnable || peer==a.rank)?0xCD:Demo::Pattern(peer,a.rank,j)))
                throw std::runtime_error("Dispatch payload mismatch");
        }
    }
    if(a.computeEnable) {
        auto checkNonzero=[&](uint64_t pointer,uint64_t bytes,const char* name) {
            std::vector<uint16_t> values(bytes/2);
            CHECK(aclrtMemcpy(values.data(),bytes,reinterpret_cast<void*>(pointer),bytes,ACL_MEMCPY_DEVICE_TO_HOST));
            bool nonzero=false;
            for(auto v:values){if((v&0x7f80U)==0x7f80U)throw std::runtime_error(std::string(name)+" nonfinite output");nonzero|=(v&0x7fffU)!=0;}
            if(!nonzero)throw std::runtime_error(std::string(name)+" all-zero output");
        };
        checkNonzero(g.gmm1Output,r*2*i*2,"routed GMM1");checkNonzero(g.gmm2Output,r*h*2,"routed GMM2");
        if(a.sharedExperts){checkNonzero(g.sharedGmm1,uint64_t(a.bs)*2*a.sharedIntermediate*2,"shared GMM1");checkNonzero(g.sharedOutput,uint64_t(a.bs)*h*2,"shared GMM2");}
        std::cout<<"[NONZERO] FP8 input=0x38 W4 packed=0x11 scale=0x7f; all GMM outputs finite and nonzero (not Golden)\n";
    }
    auto verifyFlags=[&](uint64_t address,uint64_t bytes,bool gmm2) {
        if(!bytes)return;
        std::vector<int32_t> values(bytes/sizeof(int32_t));
        CHECK(aclrtMemcpy(values.data(),bytes,reinterpret_cast<void*>(address),bytes,ACL_MEMCPY_DEVICE_TO_HOST));
        const uint64_t slots=bytes/MegaMoeUrma::kFlagBytes;
        for(uint64_t j=0;j<slots;++j) {
            const int32_t expected=a.computeEnable && a.completionNotify ?
                (gmm2?static_cast<int32_t>(MegaMoeUrma::ComputeTileCount(plan.fragments[j].rows,a.hidden)):1):0;
            if(values[j*(MegaMoeUrma::kFlagBytes/sizeof(int32_t))]!=expected)
                throw std::runtime_error("Incorrect final GMM completion flag/count at slot="+std::to_string(j));
        }
    };
    verifyFlags(g.gmm1Done,g1FlagBytes,false);verifyFlags(g.gmm2Done,g2FlagBytes,true);
    verifyFlags(g.sharedGmm1Done,sharedFlagBytes,false);
    std::cout<<"[FLAGS] last-round completion publications verified; consumers never wait on them"<<std::endl;
    std::ofstream csv(o.at("--output")+"/rank"+std::to_string(a.rank)+".csv");
    if(!csv) throw std::runtime_error("Cannot write results");
    csv<<"iteration,kind,worker,peer,start_tick,end_tick,bytes\n";
    for(uint32_t round=warmup;round<rounds;++round) for(uint32_t worker=0;worker<recordCount;++worker) {
        if(worker<56 && !a.computeEnable)continue;
        if(worker>=56 && !a.communicationEnable)continue;
        const auto* v=hr.data()+(uint64_t(round)*recordCount+worker)*Demo::words;
        if(v[1]<=v[0])throw std::runtime_error("Invalid timing record");
        auto row=[&](const char* kind,int peer,uint64_t start,uint64_t end,uint64_t bytes) {
            csv<<round-warmup<<','<<kind<<','<<worker<<','<<peer<<','<<start<<','<<end<<','<<bytes<<'\n';
        };
        if(worker<56) row(worker<28?"gmm":"dequant",-1,v[0],v[1],0);
        else {
            int peer=worker-56;if(peer>=int(a.rank))++peer;
            row("total",peer,v[0],v[1],v[2]+v[3]);
            row("dispatch_submit",peer,v[0],v[4],v[2]);
            row("combine_shared_tail",peer,v[5],v[1],v[3]);
        }
    }
    csv.close();aclshmem_barrier_all();guard.Close();
    for(void* ptr:owned)CHECK(aclrtFree(ptr));
    aclshmem_free(destination);aclshmem_free(dispatch);aclshmem_free(sourceScale);aclshmem_free(source);
    aclshmem_finalize();CHECK(aclrtDestroyStream(stream));CHECK(aclFinalize());
    std::cout<<"[PASSED] Dispatch/Combine payload (GMM numerical accuracy not checked) rank="<<a.rank<<std::endl;
    return 0;
  }catch(const std::exception& e){std::cerr<<"[FAILED] "<<e.what()<<std::endl;return 1;}
}
