/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#pragma once

template <typename T>
__aicore__ inline void NativeLoad(LocalTensor<T> ub, __gm__ T* source, uint32_t count)
{
    if (!count) {
        return;
    }
    GlobalTensor<T> gm;
    gm.SetGlobalBuffer(source);
    SetFlag<HardEvent::S_MTE2>(EVENT_ID2);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID2);
    DataCopyPad(ub, gm, DataCopyExtParams(1, count * sizeof(T), 0, 0, 0), DataCopyPadExtParams<T>{false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
}
template <typename T>
__aicore__ inline void NativeStore(__gm__ T* destination, LocalTensor<T> ub, uint32_t count)
{
    if (!count) {
        return;
    }
    GlobalTensor<T> gm;
    gm.SetGlobalBuffer(destination);
    SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
    DataCopyPad(gm, ub, DataCopyExtParams(1, count * sizeof(T), 0, 0, 0));
    SetFlag<HardEvent::MTE3_S>(EVENT_ID3);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID3);
}
