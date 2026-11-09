#pragma once
#include "timer.hpp"
#include "transport.hpp"

namespace MegaMoeUrma {
// Diagnostic-only read observer for the pinned SHMEM 9479427 direct QP ABI.
// Freeze the last Dispatch SO's CQ target BEFORE any Combine can be posted.
// Never consume CQEs, update queue credits or ring a CQ doorbell here. The
// regular SHMEM overflow handling/final quiet remains the only CQ consumer.
class DispatchCompletionObserver {
public:
    __aicore__ inline void Init(Timer& timer) {
#ifdef DEEPEP_MEGAMOE_TIMER
        if(timer.DiagnosticEnabled())timer_=&timer;
#else
        (void)timer;
#endif
    }
    __aicore__ inline void Arm(uint32_t peer,uint32_t lastCv) {
#ifdef DEEPEP_MEGAMOE_TIMER
        if(!timer_)return;
        static_assert(!ACLSHMEM_RELAY_SUPPORTED,"Diagnostic CQ observer requires direct QP mode");
        auto* info=aclshmemi_udma_qp_info_fetch();
        auto* table=aclshmemi_udma_active_table(info);
        auto* cq=reinterpret_cast<__gm__ aclshmemi_udma_cq_ctx_t*>(table->scq_ptr+
            (peer*info->qp_num+kCommunicationQp)*sizeof(aclshmemi_udma_cq_ctx_t));
        state_=reinterpret_cast<__gm__ aclshmemi_udma_queue_state_t*>(cq->state_addr);
        // Same QP-owning Scalar has just updated this cacheable producer state.
        // Bypassing its dirty cache would incorrectly read an older counter.
        target_=state_->cqe_cnt;
        const uint32_t ordinal=target_-1U;
        owner_=(ordinal/shm::UDMA_CQ_DEPTH_DEFAULT)&1U;
        word_=reinterpret_cast<__gm__ uint32_t*>(cq->buf_addr+
            cq->cqe_size*(ordinal&(shm::UDMA_CQ_DEPTH_DEFAULT-1U)));
        peer_=peer;cv_=lastCv;pending_=true;
#else
        (void)peer;(void)lastCv;
#endif
    }
    __aicore__ inline void Poll() {
#ifdef DEEPEP_MEGAMOE_TIMER
        if(!pending_)return;
        // A later normal SHMEM poll may already have retired this target (and
        // reused its ring slot). Check the owner-maintained CQ tail FIRST.
        bool complete=static_cast<int32_t>(state_->cq_tail-target_)>=0;
        if(!complete) {
            // DW0 in pinned aclshmemi_jfc_cqe_ctx_t: owner bit2, substatus
            // bits16..23, status bits24..31. One uncached load per negative
            // probe, no retry loop or pipeline fence.
            const uint32_t word=AscendC::ReadGmByPassDCache(word_);
            if(((word>>2U)&1U)==owner_)return;
            if((word>>16U)!=0U) {
                aclshmemi_kernel_abort("Dispatch diagnostic CQ error: peer=%u dw0=%u\n",peer_,word);
                return;
            }
            complete=true;
        }
        if(complete) {
            timer_->RecordDispatchCompletion(peer_,cv_);
            pending_=false;
        }
#endif
    }
private:
#ifdef DEEPEP_MEGAMOE_TIMER
    Timer* timer_=nullptr;
    __gm__ aclshmemi_udma_queue_state_t* state_=nullptr;
    __gm__ uint32_t* word_=nullptr;
    uint32_t target_=0U,owner_=0U,peer_=0U,cv_=0U;
    bool pending_=false;
#endif
};
} // namespace MegaMoeUrma
