#ifdef NDEBUG
#undef NDEBUG
#endif
#include "routed_fix_channel.hpp"
#include <array>
#include <deque>
#include <iostream>
#include <stdexcept>
#include <vector>

// Uses the production channel methods; no NPU ordering or bandwidth claim.
namespace CreditModel {
enum Kind { Wait, Arrive, Payload, FinishConsumer };
struct Command { Kind kind; unsigned flag=0, seq=0; bool ub=false; };
std::array<std::deque<Command>,3> queues;
std::array<unsigned,16> credits{};
unsigned produced=0,consumed=0,read=0,lastPayload=~0U;
bool consumerBusy=false;
void Check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
bool Pump() {
    bool progress=false;
    for(auto& q:queues) {
        if(q.empty())continue;
        const auto c=q.front();
        if(c.kind==Wait&&!credits[c.flag])continue;
        if(c.kind==Wait){--credits[c.flag];if(c.flag<14)++consumed;}
        if(c.kind==Arrive){Check(!credits[c.flag],"event overwrite");++credits[c.flag];if(c.flag<14)++produced;}
        if(c.kind==Payload){
            Check(!consumerBusy,"next FIX overlaps current low-UB consumer");
            lastPayload=c.seq;
        }
        if(c.kind==FinishConsumer){
            Check(consumerBusy && lastPayload==c.seq,"UB changed before final consumer read");
            consumerBusy=false;++read;
        }
        q.pop_front();progress=true;
    }
    return progress;
}
void PumpAll(){while(Pump()){};}
void Enqueue(unsigned pipe,unsigned flag,bool arrive) {
    const unsigned id=flag&15;
    Check(pipe<3,"pipe");
    if(pipe==PIPE_FIX)Check(flag>=16 && id>=12 && arrive==(id<14),"FIX targeting");
    if(pipe==PIPE_MTE2)Check(flag<16 && id>=12 && id<14 && !arrive,"AIV targeting");
    if(pipe==PIPE_MTE3)Check(flag>=16 && id>=14 && arrive,"empty targeting");
    queues[pipe].push_back({arrive?Arrive:Wait,id});PumpAll();
}
void Drain(unsigned pipe){PumpAll();Check(queues[pipe].empty(),"credit deadlock");}
void Reset(){for(auto& q:queues)q.clear();credits.fill(0);produced=consumed=read=0;lastPayload=~0U;consumerBusy=false;}
}
int main() {
    using MegaMoeUrma::RoutedFixChannel;
    unsigned cases=0;
    for(unsigned count:{0U,1U,2U,3U,5U,28U,57U,1025U})for(unsigned pattern=0;pattern<4;++pattern)
    for(unsigned initLate=0;initLate<2;++initLate) {
        using namespace CreditModel;
        Reset();
        if(!initLate)RoutedFixChannel::InitConsumer();
        for(unsigned i=0;i<count;++i) {
            RoutedFixChannel::Acquire(i);
            // GMM1 may still output GM in step1, but its AIV1 activation uses
            // the same low UB; its logical credit must be held equally long.
            bool toUb=pattern==3 || (pattern==1 && i%2) || (pattern==2 && !(i%2));
            queues[PIPE_FIX].push_back({Payload,0U,i,toUb});PumpAll();
            RoutedFixChannel::Publish(i);
        }
        if(initLate){Check(produced==0,"FIX started before initial grant");RoutedFixChannel::InitConsumer();}
        Check(produced==(count?1U:0U),"one physical UB bank requires exactly one outstanding payload");
        for(unsigned i=0;i<count;++i) {
            RoutedFixChannel::Wait(i);
            Check(lastPayload==i,"completion order mismatch");consumerBusy=true;
            for(unsigned delay=0;delay<13;++delay)PumpAll();
            Check(produced==i+1 && lastPayload==i,"FIX escaped while consumer owns UB");
            queues[PIPE_MTE3].push_back({FinishConsumer,0U,i});
            RoutedFixChannel::Release(i); // ordered after actual final UB read
        }
        RoutedFixChannel::DrainProducer();
        Check(produced==count&&consumed==count&&read==count,"lost task");
        for(auto& q:queues)Check(q.empty(),"unretired pipe");
        for(auto credit:credits)Check(!credit,"leaked credit between calls");
        ++cases;
    }
    std::cout<<"single shared UB credit/liveness cases="<<cases<<" passed\n";
}
