// Model delayed FIX and Scalar queues, not CANN implementation or NPU execution.
#include "kernel_operator.h"
#include <array>
#include <cassert>
#include <deque>
#include <functional>
#include <vector>
#define g_coreType AscendC::AIC
#define Gmm2AivCompletion AicCompletion
#include "common/gmm2_aiv_completion.hpp"
#undef Gmm2AivCompletion
#undef g_coreType
#undef MEGA_MOE_GMM2_AIV_COMPLETION_HPP
#define g_coreType AscendC::AIV
#define Gmm2AivCompletion AivCompletion
#include "common/gmm2_aiv_completion.hpp"
#undef Gmm2AivCompletion
#undef g_coreType

unsigned activeCore=0,activeSub=0;
struct Op { unsigned kind,flag;std::function<void()> action; };
std::deque<Op> queues[2][2];
unsigned tokens[2][32],fullArrivals,emptyArrivals,fullWaits,emptyWaits,hardWaits,completedWave,expectedWave;
bool hardComplete;
void Record(unsigned pipe,unsigned kind,unsigned flag) {
 assert(activeCore<2&&pipe<2&&flag<32);
 queues[activeCore][pipe].push_back({kind,flag,{}});
 if(kind==4)++hardWaits;
}
struct Params { mutable int32_t ready=0; };
int32_t* GetTokenCombineReadyCounter(const Params&p,unsigned expert,unsigned row){assert(expert==0&&row==0);return &p.ready;}
unsigned GetSubBlockIdx(){return activeSub;}
namespace AicStage {constexpr int g_coreType=AIC;
#include "notify_slice_ready.inc"
}
namespace AivStage {constexpr int g_coreType=AIV;
#include "notify_slice_ready.inc"
}
void AscendC::WriteGmByPassDCache(int32_t*p,int32_t value) {
 queues[activeCore][PIPE_S].push_back({2,0,[=]{*p=value;}});
}
void AscendC::AtomicAdd(int32_t*p,int value) {
 assert(value==1);const auto wave=expectedWave;
 queues[activeCore][PIPE_S].push_back({2,0,[=]{assert(completedWave>=wave);++*p;}});
}
bool Step(unsigned core,unsigned pipe) {
 auto&q=queues[core][pipe];if(q.empty())return false;
 auto op=q.front();
 if(op.kind==1&&tokens[core][op.flag]==0)return false;
 if(op.kind==4&&!hardComplete)return false;
 q.pop_front();
 if(op.kind==0) {
  auto&count=tokens[1-core][core==AIC ? op.flag-16 : op.flag+16];++count;assert(count<=1);
  if(core==AIC){assert(op.flag==(16U | MegaMoeImpl::AicCompletion::FULL));++fullArrivals;}
  else{assert(op.flag==MegaMoeImpl::AivCompletion::EMPTY);++emptyArrivals;}
 }else if(op.kind==1){
  --tokens[core][op.flag];if(core==AIC)++emptyWaits;else++fullWaits;
 }else if(op.kind==2)op.action();
 else if(op.kind==3)hardComplete=true;
 return true;
}
void Pump(unsigned salt) {
 unsigned stalled=0;
 for(unsigned step=0;step<100000;++step){
  bool pending=false;for(auto&c:queues)for(auto&q:c)pending|=!q.empty();if(!pending)return;
  // Alternate pipeline priorities to cover producer and consumer skew.
  bool progress=false;for(unsigned i=0;i<4;++i){unsigned x=(i+salt+step)%4;progress|=Step(x/2,x%2);}
  if(!progress){++stalled;assert(stalled<2);}else stalled=0;
 }
 assert(false);
}
void Check(unsigned waves,unsigned salt) {
 for(auto&c:queues)for(auto&q:c)q.clear();for(auto&c:tokens)for(auto&x:c)x=0;
 fullArrivals=emptyArrivals=fullWaits=emptyWaits=hardWaits=completedWave=expectedWave=0;hardComplete=false;
 Params p;int32_t issued=0;activeCore=AIV;activeSub=1;MegaMoeImpl::AivCompletion::InitConsumer();
 unsigned publishers=0;
 for(unsigned wave=1;wave<=waves;++wave){
  activeCore=AIC;expectedWave=wave;
  queues[AIC][PIPE_FIX].push_back({2,0,[wave]{completedWave=wave;}}); // Issued output DMA.
  MegaMoeImpl::AicCompletion::PublishCompleted(&issued,wave);
  AicStage::NotifyTokenCombineSliceReady<true>(p,0,0); // AIC must not publish AIV-completed ready.
  assert(hardWaits==0&&p.ready==0); // Wave publish does not drain FIX into Scalar.
  assert(!MegaMoeImpl::AivCompletion::IsIssued(&issued,wave));
  // Scalar issued may advance ahead of FIX; never use it as payload readiness.
  assert(Step(AIC,PIPE_S));
  assert(MegaMoeImpl::AivCompletion::IsIssued(&issued,wave));
  activeCore=AIV;activeSub=0;
  MegaMoeImpl::AivCompletion::InitConsumer();MegaMoeImpl::AivCompletion::WaitConsumer();
  AivStage::NotifyTokenCombineSliceReady<true>(p,0,0);MegaMoeImpl::AivCompletion::ReleaseConsumer();
  activeSub=1;MegaMoeImpl::AivCompletion::WaitConsumer();
  if(wave%3!=2){AivStage::NotifyTokenCombineSliceReady<true>(p,0,0);++publishers;}
  AivStage::NotifyTokenCombineSliceReady<false>(p,0,0); // Default path remains AIC-only.
  MegaMoeImpl::AivCompletion::ReleaseConsumer();
 }
 activeCore=AIC;MegaMoeImpl::AicCompletion::DrainProducer();assert(hardWaits==1);
 Pump(salt);
 assert(p.ready==int32_t(publishers)&&completedWave==waves);
 assert(fullArrivals==waves&&fullWaits==waves&&emptyArrivals==waves+1&&emptyWaits==waves+1);
 for(auto&c:tokens)for(auto x:c)assert(x==0); // No stale credit across repeated kernel calls.
 // The synchronous standalone-slice API is still restricted to AIC.
 activeCore=AIC;expectedWave=waves;AicStage::NotifyTokenCombineSliceReady(p,0,0);Pump(salt);
 assert(p.ready==int32_t(publishers+1));
}
int main(){for(unsigned waves:{0U,1U,2U,3U,16U,127U})for(unsigned salt=0;salt<4;++salt)Check(waves,salt);}
