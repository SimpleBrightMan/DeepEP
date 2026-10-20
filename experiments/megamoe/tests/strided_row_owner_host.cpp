#include <cassert>
#include <cstdint>
#include <vector>
#include "common/strided_row_owner.hpp"
int main() {
    for(uint32_t rows=0;rows<=128;++rows)for(uint32_t cores:{1U,8U,32U,64U}) {
        std::vector<uint32_t> seen(rows,0);
        uint32_t published=0;
        for(uint32_t core=0;core<cores;++core) {
            MegaMoeImpl::StridedRowOwner owner{core,rows,cores};
            uint32_t row=0,contribution=0;
            while(owner.Next(row)) {
                assert(row<rows && row%cores==core);
                assert(row/256==0); // 本协议的ready贡献只能用于单M-group。
                ++seen[row];++contribution;
            }
            published+=contribution;
        }
        assert(published==rows);
        for(auto count:seen)assert(count==1);
    }
    uint32_t row=0;
    MegaMoeImpl::StridedRowOwner invalid{0,128,0};
    assert(!invalid.Next(row));
}
