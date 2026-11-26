// Exercise the actual header: destinations, credit wraparound and tail consumption.
// CPU event model checks protocol accounting, not NPU visibility or execution timing.
#include <cassert>
#include "../csrc/kernels/mega_moe/common/gmm1_async_completion.hpp"
using Channel = MegaMoeImpl::Gmm1AsyncCompletion;
int main() {
    for (unsigned tasks : {0U, 1U, 2U, 3U, 36U, 113U}) {
        recorded.clear();
        drains = 0;
        Channel::InitConsumer();
        for (unsigned task = 0; task < tasks; ++task) {
            Channel::WaitConsumer(task);
            Channel::ReleaseConsumer(task);
        }
        auto consumer = recorded;
        recorded.clear();
        for (unsigned task = 0; task < tasks; ++task) {
            Channel::AcquireProducer(task);
            Channel::PublishCompleted(task);
        }
        Channel::DrainProducer();
        auto producer = recorded;
        assert(drains == 2);
        unsigned tokens[32] = {};
        unsigned consumed = 0, completed = 0;
        unsigned ci = 0, pi = 0;
        while (ci != consumer.size() || pi != producer.size()) {
            bool progress = false;
            auto step = [&](const std::vector<Event>& events, unsigned& index, bool isProducer) {
                if (index == events.size()) return;
                auto e = events[index];
                assert(e.pipe == (isProducer ? PIPE_FIX : e.arrive ? PIPE_MTE3 : PIPE_MTE2));
                // AIC/full notify and AIV/empty notify select AIV1. AIV wait uses local low bits.
                assert((e.flag >= 16) == (isProducer || e.arrive));
                unsigned flag = e.flag | 16;
                assert(flag >= 28 && flag <= 31);
                if (e.arrive) {
                    assert(tokens[flag] == 0); // No second publication before the prior token is consumed.
                    ++tokens[flag];
                    if (isProducer) ++completed;
                } else {
                    if (!tokens[flag]) return;
                    --tokens[flag];
                    if (!isProducer) { ++consumed; assert(consumed <= completed); }
                }
                ++index;
                progress = true;
            };
            step(producer, pi, true);
            step(consumer, ci, false);
            assert(progress); // Detect a missing initial credit or unmatched final wait.
        }
        assert(consumed == tasks && completed == tasks);
        for (auto token : tokens) assert(token == 0);
    }
}
