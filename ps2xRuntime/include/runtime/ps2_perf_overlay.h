#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace ps2x::perfoverlay
{
    void addCounter(std::string label, std::function<uint64_t()> read, uint64_t wrapMask = ~uint64_t{0});
    void setVisible(bool visible);
    bool visible();
    void toggle();
    void recordPresent(double intervalSeconds, uint64_t field, bool sharedPath, uint64_t sharedSequence);
    void draw();
}
