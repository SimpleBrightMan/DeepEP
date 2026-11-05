#include "routed_fix_channel.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <deque>
#include <iostream>
#include <stdexcept>

// Exercise the exact production channel methods against ordered hardware-pipe
// queues. This is a credit/liveness model, not NPU ordering validation.
namespace CreditModel {
struct Command { unsigned flag; bool arrive; };
std::array<std::deque<Command>,3> queues;
std::array<unsigned,16> credits{};
unsigned produced=0,consumed=0;
bool Pump() {
    bool progressed=false;
    for(auto& queue:queues) {
        if(queue.empty())continue;
        const auto op=queue.front();
        if(!op.arrive&&!credits[op.flag])continue;
        if(op.arrive) { assert(credits[op.flag]==0U); ++credits[op.flag]; if(op.flag<14U)++produced; }
        else { --credits[op.flag]; if(op.flag<14U)++consumed; }
        queue.pop_front();progressed=true;
    }
    return progressed;
}
void PumpAll() { while(Pump()){} }
void Enqueue(unsigned pipe,unsigned flag,bool arrive) {
    assert(pipe<=2U);
    const unsigned id=flag&15U;
    if(pipe==PIPE_FIX)assert(flag>=16U&&id>=12U&&arrive==(id<14U));
    if(pipe==PIPE_MTE2)assert(flag<16U&&id>=12U&&id<14U&&!arrive);
    if(pipe==PIPE_MTE3)assert(flag>=16U&&id>=14U&&arrive);
    queues[pipe].push_back({id,arrive});PumpAll();
}
void Drain(unsigned pipe) {
    PumpAll();
    if(!queues[pipe].empty())throw std::runtime_error("credit deadlock");
}
void Reset() { for(auto& q:queues)q.clear();credits.fill(0);produced=consumed=0; }
}
int main() {
    using MegaMoeUrma::RoutedFixChannel;
    for(unsigned count: {0U,1U,2U,3U,5U,28U,57U,1025U}) {
        CreditModel::Reset();RoutedFixChannel::InitConsumer();
        // Scalar can enqueue multiple tiles before the consumer runs. FIX is
        // held by the two credits and must never reuse a full flag early.
        for(unsigned i=0;i<count;++i) { RoutedFixChannel::Acquire(i);RoutedFixChannel::Publish(i); }
        assert(CreditModel::produced==std::min(count,2U));
        for(unsigned i=0;i<count;++i) { RoutedFixChannel::Wait(i);RoutedFixChannel::Release(i); }
        RoutedFixChannel::DrainProducer();
        assert(CreditModel::produced==count&&CreditModel::consumed==count);
        for(auto& q:CreditModel::queues)assert(q.empty());
        for(auto credit:CreditModel::credits)assert(!credit);
    }
    std::cout<<"routed FIX channel credit and zero-work cases passed\n";
}
