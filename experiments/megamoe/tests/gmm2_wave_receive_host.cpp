// Execute the production AIV1 planner with delayed issued state on every physical owner.
#include "common/mega_moe_wave_schedule.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#define __gm__
using namespace MegaMoeImpl;
constexpr uint32_t INT_CACHELINE=16, L1_TILE_N=256;
using GM_ADDR=const std::vector<uint32_t>*;
using BlockWorkspaceContext=int;
namespace Ops::Base { template<class T>T CeilDiv(T a,T b){return (a+b-1)/b;} }
namespace AscendTimer { constexpr int AIV_COMBINE_WAIT=1; }
struct Timer {
 int64_t wave=-1; bool cleared=false;
 bool Enabled(){return true;} void SetWave(int64_t w){wave=w;} void ClearSlice(){cleared=true;}
};
struct AscendTimerWaitScope { AscendTimerWaitScope(Timer*,int){} };
struct Gmm2AivCompletion {
 static inline bool issued=false; static inline unsigned waits=0,releases=0;
 static bool IsIssued(int32_t*,int32_t){return issued;}
 static void WaitConsumer(){assert(issued);++waits;}
 static void ReleaseConsumer(){assert(waits==releases+1);++releases;}
};
struct Harness {
 static constexpr uint32_t GMM1_TILE_M=256;
 static constexpr bool GMM2_REVERSE_BLOCKS=false;
 struct {uint32_t moeExpertPerRank,tokenHiddenDim=7168;}commonConfig_;
 struct {struct{uint32_t jobIndex,totalJobs=32;}blockJob;}gmmExecutionConfig_;
 struct {struct{GM_ADDR expertRevTokenNumsPtr; void* flagGmm2IssuedPtr;}workspaceInfo;}params_;
 int32_t issued[32*INT_CACHELINE]{};
 int countWorkspace_=0;uint32_t mGroupsPerWave_;Timer timer_;
 std::vector<std::array<uint32_t,2>> notified;
 Harness(const std::vector<uint32_t>&c,uint32_t core,uint32_t groups):commonConfig_{uint32_t(c.size())},
  gmmExecutionConfig_{{core}},params_{{&c,issued}},mGroupsPerWave_(groups){}
 uint32_t GetExpertTokenCountFromWorkspace(GM_ADDR c,int,uint32_t,uint32_t e){return c->at(e);}
 uint32_t GetMGroupCountForRows(uint32_t n,uint32_t m){return (n+m-1)/m;}
 uint32_t GetGmmFirstOwnedTile(uint32_t core,uint32_t jobs,uint32_t start,bool reverse){
  assert(!reverse);return (core+jobs-start)%jobs;
 }
 bool HasWaveCapacity(const ExpertTokenPosition&p,uint32_t groups){return MegaMoeImpl::HasWaveCapacity(p,commonConfig_.moeExpertPerRank,groups,mGroupsPerWave_);}
 template<bool Aiv>void NotifyTokenCombineSliceReady(decltype(params_)&,uint32_t e,uint32_t row){
  static_assert(Aiv);assert(Gmm2AivCompletion::waits==Gmm2AivCompletion::releases+1);notified.push_back({e,row});
 }
#include "gmm2_wave_receive.inc"
};
void Check(const std::vector<uint32_t>&c,uint32_t groups){
 for(uint32_t core=0;core<32;++core){
  Harness h(c,core,groups);ExpertTokenPosition pos{};uint32_t start=0;int32_t seq=0;
  Gmm2AivCompletion::waits=Gmm2AivCompletion::releases=0;
  uint32_t e=0,row=0,owner=0,waves=0;
  while(e<c.size()){
   std::vector<std::array<uint32_t,2>> expected;uint32_t used=0;
   while(e<c.size()&&used<groups){
    if(row==c[e]){++e;row=0;continue;}
    const uint32_t n=std::min(c[e]-row,(groups-used)*256),tiles=((n+255)/256)*28;
    if((core+32-owner)%32<tiles)expected.push_back({e,row});
    owner=(owner+tiles)%32;used+=(n+255)/256;row+=n;if(row==c[e]){++e;row=0;}
   }
   const auto before=pos;const auto previousStart=start;
   Gmm2AivCompletion::issued=false;
   assert(!h.TryReceiveGmm2Completion(pos,start,seq));
   assert(pos.expertIdx==before.expertIdx&&pos.tokenIndexInExpert==before.tokenIndexInExpert&&start==previousStart);
   assert(Gmm2AivCompletion::waits==waves);
   Gmm2AivCompletion::issued=true;h.notified.clear();
   assert(h.TryReceiveGmm2Completion(pos,start,seq));
   assert(h.notified==expected&&start==owner&&seq==int32_t(++waves));
   assert(pos.expertIdx==e&&pos.tokenIndexInExpert==row);
   assert(h.timer_.wave==seq-1&&h.timer_.cleared);
  }
  assert(!h.TryReceiveGmm2Completion(pos,start,seq));
  assert(Gmm2AivCompletion::waits==waves&&Gmm2AivCompletion::releases==waves);
 }
}
int main(){
 for(uint32_t groups:{1U,2U,5U,32U}){
  Check({},groups);Check({0,0,0},groups);Check({1,255,256,257,0,1025,0,0},groups);
  Check(std::vector<uint32_t>(48,1536),groups);
  uint32_t seed=71;for(unsigned trial=0;trial<80;++trial){std::vector<uint32_t>c(48);
   for(auto&n:c){seed=seed*1664525U+1013904223U;n=(seed%4)?seed%2800:0;}Check(c,groups);}
 }
}
