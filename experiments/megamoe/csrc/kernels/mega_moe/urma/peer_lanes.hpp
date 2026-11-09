#pragma once
#include "abi.hpp"
namespace MegaMoeUrma {
#if defined(__NPU_DEVICE__)
#define URMA_PEER_INLINE __aicore__ inline
#else
#define URMA_PEER_INLINE inline
#endif
URMA_PEER_INLINE constexpr uint32_t PeerLanes(uint32_t world){return world==4U||world==8U?2U:1U;}
URMA_PEER_INLINE constexpr bool DualPeerMode(uint32_t world){return PeerLanes(world)==2U;}
URMA_PEER_INLINE constexpr uint32_t CommunicationRoles(uint32_t world){return (world-1U)*PeerLanes(world);}
URMA_PEER_INLINE constexpr uint32_t OwnerQp(uint32_t owner,uint32_t world){return owner%PeerLanes(world);}
URMA_PEER_INLINE constexpr bool PeerLeader(uint32_t owner,uint32_t world){return OwnerQp(owner,world)==0U;}
URMA_PEER_INLINE constexpr uint32_t PeerForOwner(uint32_t rank,uint32_t owner,uint32_t world=kWorld){
    const uint32_t peer=owner/PeerLanes(world);return peer<rank?peer:peer+1U;
}
// Round-robin CV ownership exposes both startup Pack CVs immediately.
URMA_PEER_INLINE constexpr uint32_t PeerOwnedCvs(uint32_t total,uint32_t lane,uint32_t lanes){
    return lane<total?(total-1U-lane)/lanes+1U:0U;
}
URMA_PEER_INLINE constexpr uint32_t PeerLastCv(uint32_t total,uint32_t lane,uint32_t lanes){
    return lane+(PeerOwnedCvs(total,lane,lanes)-1U)*lanes;
}
#undef URMA_PEER_INLINE
} // namespace MegaMoeUrma
