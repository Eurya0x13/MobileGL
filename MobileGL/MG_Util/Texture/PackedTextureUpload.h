// MobileGL - MobileGL/MG_Util/Texture/PackedTextureUpload.h
// Copyright (c) 2025-2026 MobileGL-Dev
// Licensed under the GNU Lesser General Public License v3.0:
//   https://www.gnu.org/licenses/gpl-3.0.txt
//   https://www.gnu.org/licenses/lgpl-3.0.txt
// SPDX-License-Identifier: LGPL-3.0-only
// End of Source File Header

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace MobileGL::MG_Util::PackedTextureUpload {
    // A small RGBA8 page bounds both temporary storage and the packing search.
    constexpr int PageSide = 128;
    constexpr std::size_t MaxRegions = 32;
    constexpr std::size_t BytesPerTexel = 4;

    struct Region {
        const std::uint8_t* source;
        std::size_t sourcePitch;
        int level, x, y, width, height;
    };

    struct Point {
        int x, y;
    };

    struct Batch {
        std::array<Region, MaxRegions> regions;
        std::array<Point, MaxRegions> offsets;
        std::size_t count = 0;
        int width = 0;
        int height = 0;

        std::size_t ByteSize() const { return BytesPerTexel * width * height; }
    };

    struct DirectUploadCost {
        std::uint64_t bytes = 0;
        std::size_t uploads = 0;
    };

    namespace Detail {
        struct Rect {
            int x, y, width, height;
            int Area() const { return width * height; }
        };

        // Best-area-fit guillotine packing. Splitting on the shorter remainder
        // leaves the larger free rectangle available for the next sprite.
        inline bool TryWidth(const Batch& batch, int width, Batch& result) {
            std::array<Rect, MaxRegions + 1> free;
            std::size_t freeCount = 1;
            free[0] = {0, 0, width, PageSide};
            int usedWidth = 0, usedHeight = 0;

            for (std::size_t i = 0; i < batch.count; ++i) {
                const Region& region = batch.regions[i];
                std::size_t best = freeCount;
                for (std::size_t j = 0; j < freeCount; ++j) {
                    const Rect& rect = free[j];
                    if (region.width <= rect.width && region.height <= rect.height &&
                        (best == freeCount || rect.Area() < free[best].Area())) {
                        best = j;
                    }
                }
                if (best == freeCount) return false;

                const Rect rect = free[best];
                free[best] = free[--freeCount];
                result.offsets[i] = {rect.x, rect.y};
                usedWidth = std::max(usedWidth, rect.x + region.width);
                usedHeight = std::max(usedHeight, rect.y + region.height);

                const int right = rect.width - region.width;
                const int below = rect.height - region.height;
                if (right > 0) {
                    free[freeCount++] = {rect.x + region.width, rect.y, right,
                                         right <= below ? region.height : rect.height};
                }
                if (below > 0) {
                    free[freeCount++] = {rect.x, rect.y + region.height, right <= below ? rect.width : region.width,
                                         below};
                }
            }

            result.width = usedWidth;
            result.height = usedHeight;
            return true;
        }
    } // namespace Detail

    inline bool Pack(Batch& batch, DirectUploadCost direct) {
        batch.width = 0;
        batch.height = 0;
        if (batch.count < 2 || batch.count > batch.regions.size()) return false;
        int widest = 1, dirtyTexels = 0;
        for (std::size_t i = 0; i < batch.count; ++i) {
            const Region& region = batch.regions[i];
            if (region.source == nullptr || region.width <= 0 || region.width > PageSide || region.height <= 0 ||
                region.height > PageSide) {
                return false;
            }
            widest = std::max(widest, region.width);
            dirtyTexels += region.width * region.height;
        }
        if (dirtyTexels > PageSide * PageSide) return false;

        // Direct uploads stage and transfer every byte. Packing clears and
        // transfers the staging page, stages the dirty texels, and reads and
        // writes them on the GPU. Reserve byte-equivalent headroom for one
        // upload, two read FBO binds, and one copy per region.
        constexpr std::uint64_t ExtraCommandCostBytes = 1024;
        constexpr std::uint64_t MinimumSavingsBytes = 2 * 1024;
        const std::size_t packedCommands = batch.count + 3;
        const std::size_t extraCommands = packedCommands - std::min(packedCommands, direct.uploads);
        const std::uint64_t directWork = 2 * direct.bytes;
        const std::uint64_t dirtyBytes = dirtyTexels * BytesPerTexel;
        const std::uint64_t reserve = 3 * dirtyBytes + extraCommands * ExtraCommandCostBytes + MinimumSavingsBytes;
        if (directWork <= 2 * dirtyBytes + reserve) return false;

        std::sort(batch.regions.begin(), batch.regions.begin() + batch.count, [](const Region& a, const Region& b) {
            const int aArea = a.width * a.height;
            const int bArea = b.width * b.height;
            if (aArea != bArea) return aArea > bArea;
            return std::max(a.width, a.height) > std::max(b.width, b.height);
        });

        int width = 1;
        while (width < widest)
            width *= 2;
        Batch layout;
        for (; width <= PageSide; width *= 2) {
            if (!Detail::TryWidth(batch, width, layout)) continue;
            if (batch.width > 0 && layout.ByteSize() >= batch.ByteSize()) continue;
            std::copy_n(layout.offsets.begin(), batch.count, batch.offsets.begin());
            batch.width = layout.width;
            batch.height = layout.height;
        }
        return batch.width > 0 && directWork > 2 * batch.ByteSize() + reserve;
    }

    inline void WritePage(const Batch& batch, std::uint8_t* page) {
        std::memset(page, 0, batch.ByteSize());
        const std::size_t pitch = batch.width * BytesPerTexel;
        for (std::size_t i = 0; i < batch.count; ++i) {
            const Region& region = batch.regions[i];
            const Point offset = batch.offsets[i];
            auto* dest = page + offset.y * pitch + offset.x * BytesPerTexel;
            for (int y = 0; y < region.height; ++y) {
                std::memcpy(dest + y * pitch, region.source + y * region.sourcePitch, region.width * BytesPerTexel);
            }
        }
    }
} // namespace MobileGL::MG_Util::PackedTextureUpload
