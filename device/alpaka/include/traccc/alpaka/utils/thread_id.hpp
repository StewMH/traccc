/**
 * traccc library, part of the ACTS project (R&D line)
 *
 * (c) 2024 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include "traccc/definitions/qualifiers.hpp"

namespace traccc::alpaka {
template <typename TAcc>
struct thread_id1 {
    ALPAKA_FN_INLINE ALPAKA_FN_ACC thread_id1(const TAcc* acc) : m_acc(acc) {}

    auto inline ALPAKA_FN_ACC getLocalThreadId() const {
        return ::alpaka::getIdx<::alpaka::Block, ::alpaka::Threads>(*m_acc)[0u];
    }

    auto inline ALPAKA_FN_ACC getLocalThreadIdX() const {
        return getLocalThreadId();
    }

    auto inline ALPAKA_FN_ACC getGlobalThreadId() const {
        return getLocalThreadId() + getBlockIdX() * getBlockDimX();
    }

    auto inline ALPAKA_FN_ACC getGlobalThreadIdX() const {
        return getLocalThreadId() + getBlockIdX() * getBlockDimX();
    }

    auto inline ALPAKA_FN_ACC getBlockIdX() const {
        return ::alpaka::getIdx<::alpaka::Grid, ::alpaka::Blocks>(*m_acc)[0u];
    }

    auto inline ALPAKA_FN_ACC getBlockDimX() const {
        return ::alpaka::getWorkDiv<::alpaka::Block, ::alpaka::Threads>(
            *m_acc)[0u];
    }

    auto inline ALPAKA_FN_ACC getGridDimX() const {
        return ::alpaka::getWorkDiv<::alpaka::Grid, ::alpaka::Blocks>(
            *m_acc)[0u];
    }

    private:
    const TAcc* m_acc;
};
}  // namespace traccc::alpaka
