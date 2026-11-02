#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "shmem.h"
#include "layout.hpp"
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
    a.hidden=3584;a.topk=8;a.l1Enable=number("--l1-enable");
    a.l1Repeat=number("--l1-repeat");a.gapCycles=number("--l1-gap-cycles");
    const uint32_t warmup=number("--warmup"),iters=number("--iters"),rounds=warmup+iters;
    if((a.world!=4 && a.world!=8 && a.world!=16) || a.rank>=a.world ||
       a.bs<1024 || a.bs>8192 || a.bs%128 || a.l1Enable>1 || !a.l1Repeat || a.l1Repeat>64 ||
       !iters || iters>1000 || warmup>1000 || a.gapCycles>10000000)
        throw std::runtime_error("Invalid case/configuration");
    const uint32_t rows=a.bs*a.topk/a.world,cvs=(rows+255)/256;
    const uint64_t slot=uint64_t(rows)*a.hidden*2,storage=a.world*slot;
    const uint64_t flagBytes=uint64_t(a.world)*cvs*128;
    const uint32_t recordCount=Demo::cores+a.world-1;
    const uint64_t recordBytes=uint64_t(rounds)*recordCount*Demo::words*8;
    a.l1Bytes=Demo::L1Bytes(a.bs,a.hidden,1280,a.topk);
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
    if(l1<Demo::copyBytes*Demo::slots) throw std::runtime_error("Needs 256KiB L1");
    const auto blocks=platform->CalcTschBlockDim(56,28,56);
    aclshmemx_init_attr_t attr{};aclshmemx_uniqueid_t uid{};
    attr.my_pe=a.rank;attr.n_pes=a.world;attr.local_mem_size=2ULL<<30;
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
    auto* destination=static_cast<uint8_t*>(aclshmem_malloc(storage));
    auto* flags=static_cast<uint8_t*>(aclshmem_malloc(flagBytes));
    if(!source || !destination || !flags) throw std::runtime_error("Symmetric allocation failed");
    void *l1Source=nullptr,*records=nullptr,*workspace=nullptr;
    // Both off/on allocate and initialize the SAME buffers (allocation footprint control).
    CHECK(aclrtMalloc(&l1Source,a.l1Bytes,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemset(l1Source,a.l1Bytes,0x35,a.l1Bytes));
    CHECK(aclrtMalloc(&records,recordBytes,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemset(records,recordBytes,0,recordBytes));
    CHECK(aclrtMalloc(&workspace,platform->GetLibApiWorkSpaceSize(),ACL_MEM_MALLOC_HUGE_FIRST));
    a.source=uintptr_t(source);a.destination=uintptr_t(destination);a.flags=uintptr_t(flags);
    a.l1Source=uintptr_t(l1Source);a.workspace=uintptr_t(workspace);a.ffts=shmemx_get_ffts_config();
    std::vector<uint8_t> host(slot);
    for(uint32_t peer=0;peer<a.world;++peer) {
        for(uint64_t i=0;i<slot;++i)host[i]=Demo::Pattern(a.rank,peer,i);
        CHECK(aclrtMemcpy(source+peer*slot,slot,host.data(),slot,ACL_MEMCPY_HOST_TO_DEVICE));
    }
    CHECK(aclrtMemset(destination,storage,0xCD,storage));
    ExpGuard guard;guard.Init(number("--exp-rows"),number("--exp-cols"),number("--exp-repeat"));guard.Prepare();
    std::cout<<"[CONFIG] rank="<<a.rank<<" mode="<<(a.l1Enable?"on":"off")
             <<" rows/peer="<<rows<<" puts+SO/peer="<<cvs<<" combine_bytes/card="<<(a.world-1)*slot
             <<" l1_bytes/pass="<<a.l1Bytes<<" l1_repeat="<<a.l1Repeat<<" gap_cycles="<<a.gapCycles<<std::endl;
    aclshmem_barrier_all();launch_barrier(stream);guard.Enqueue(stream);
    for(uint32_t round=0;round<rounds;++round) {
        // Previous round's post-barrier prevents reset racing a late old SO add.
        CHECK(aclrtMemsetAsync(flags,flagBytes,0,flagBytes,stream));
        launch_barrier(stream);
        a.records=uintptr_t(records)+uint64_t(round)*recordCount*Demo::words*8;
        launch_demo(blocks,ub,stream,&a);
        launch_barrier(stream);
    }
    CHECK(aclrtSynchronizeStream(stream));aclshmem_barrier_all();
    std::vector<uint64_t> hr(recordBytes/8),hf(flagBytes/8);
    CHECK(aclrtMemcpy(hr.data(),recordBytes,records,recordBytes,ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK(aclrtMemcpy(hf.data(),flagBytes,flags,flagBytes,ACL_MEMCPY_DEVICE_TO_HOST));
    for(uint32_t peer=0;peer<a.world;++peer) {
        CHECK(aclrtMemcpy(host.data(),slot,destination+peer*slot,slot,ACL_MEMCPY_DEVICE_TO_HOST));
        for(uint64_t i=0;i<slot;++i) if(host[i]!=(peer==a.rank?0xCD:Demo::Pattern(peer,a.rank,i)))
            throw std::runtime_error("Payload mismatch peer="+std::to_string(peer)+" byte="+std::to_string(i));
        for(uint32_t word=0;word<cvs;++word) for(uint32_t lane=0;lane<16;++lane) {
            uint64_t expected=0;
            if(peer!=a.rank && lane==0 && word*64<cvs) {
                uint32_t bits=std::min(64U,cvs-word*64);expected=bits==64?~uint64_t{0}:(uint64_t{1}<<bits)-1;
            }
            if(hf[(uint64_t(peer)*cvs+word)*16+lane]!=expected) throw std::runtime_error("SO bitmap mismatch");
        }
    }
    std::ofstream csv(o.at("--output")+"/rank"+std::to_string(a.rank)+".csv");
    if(!csv) throw std::runtime_error("Cannot write results");
    csv<<"iteration,kind,worker,peer,start_tick,end_tick,bytes\n";
    for(uint32_t round=warmup;round<rounds;++round) for(uint32_t worker=0;worker<recordCount;++worker) {
        if(worker<28 && !a.l1Enable)continue;
        const auto* p=hr.data()+(uint64_t(round)*recordCount+worker)*Demo::words;
        if(p[1]<=p[0] || !p[2])throw std::runtime_error("Invalid timing record");
        int peer=-1;if(worker>=28){peer=worker-28;if(peer>=int(a.rank))++peer;}
        csv<<round-warmup<<','<<(worker<28?"l1":"combine")<<','<<worker<<','<<peer<<','<<p[0]<<','<<p[1]<<','<<p[2]<<'\n';
    }
    csv.close();aclshmem_barrier_all();
    guard.Close();CHECK(aclrtFree(l1Source));CHECK(aclrtFree(records));CHECK(aclrtFree(workspace));
    aclshmem_free(flags);aclshmem_free(destination);aclshmem_free(source);
    aclshmem_finalize();CHECK(aclrtDestroyStream(stream));CHECK(aclFinalize());
    std::cout<<"[PASSED] payload+SO bitmap rank="<<a.rank<<std::endl;
    return 0;
  }catch(const std::exception& e){std::cerr<<"[FAILED] "<<e.what()<<std::endl;return 1;}
}
