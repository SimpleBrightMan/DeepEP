#include <cassert>
#include <vector>
#include "common/small_linear_task_schedule.hpp"
int main()
{
    using namespace MegaMoeImpl;
    SmallLinearTaskSchedule example{48,24,28,32};
    const SmallLinearTask expected[]{{true,0},{true,32},{true,64},{false,0},{true,96},{false,32},{false,64}};
    SmallLinearTask task{};
    for (auto e : expected) { assert(example.Next(0,task)); assert(task.gmm1==e.gmm1 && task.index==e.index); }
    for (unsigned blocks=0; blocks<=64; ++blocks) {
        for (unsigned cores : {1U,8U,32U}) {
            for (unsigned n1 : {1U,24U,32U}) for (unsigned n2 : {1U,28U,32U}) {
                std::vector<unsigned> first(blocks*n1), second(blocks*n2);
                for (unsigned core=0; core<cores; ++core) {
                    SmallLinearTaskSchedule s{blocks,n1,n2,cores};
                    unsigned count=0;
                    while (s.Next(core,task)) {
                        auto& target=task.gmm1 ? first : second;
                        assert(task.index<target.size() && task.index%cores==core);
                        ++target[task.index]; assert(++count<=blocks*(n1+n2));
                    }
                }
                for (auto n : first) assert(n==1);
                for (auto n : second) assert(n==1);
            }
        }
    }
    assert((!SmallLinearTaskSchedule{48,24,28,0}.Next(0,task)));
}
