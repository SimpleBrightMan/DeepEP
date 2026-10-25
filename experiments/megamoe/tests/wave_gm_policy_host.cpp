#include <cassert>
#include <initializer_list>
#include "common/mega_moe_aiv1_ub_layout.hpp"
using namespace MegaMoeImpl;
int main()
{
    for (unsigned ep : {8,32,64,128})
        for (unsigned bs : {1,32,96,256,512,1024,2048,4096,65536}) {
            assert(!UseGmm1CvForLaunch(bs,6,ep,384/ep,4096));
            const auto gm = MakeAiv1ResidentUbLayout(7168,ep,384/ep,4096,6,7392,2,false);
            assert(gm.valid && gm.dispatchBase==Aiv1ResidentUbLayout::GmActivationBytes);
            assert(gm.dispatchEnd<=gm.combineRowsBase && gm.combineRowsBase<=gm.combineMetaBase);
            assert(gm.end<=256*1024 && gm.combineRowCount==2);
        }
    assert(!UseGmm1CvForLaunch(0,0,0,0,0));
}
