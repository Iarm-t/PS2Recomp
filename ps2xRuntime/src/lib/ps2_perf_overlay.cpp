#include "runtime/ps2_perf_overlay.h"
#include "raylib.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <vector>

namespace ps2x::perfoverlay
{
    namespace
    {
        constexpr double kSampleSeconds = 0.5;
        constexpr size_t kHistory = 240;
        constexpr float kGraphCeilingMs = 50.0f;

        struct Counter
        {
            std::string label;
            std::function<uint64_t()> read;
            uint64_t wrapMask = ~uint64_t{0};
            uint64_t last = 0;
            bool primed = false;
            double rate = 0.0;
        };

        struct State
        {
            std::mutex mutex;
            std::vector<Counter> counters;
            bool visible = false;
            std::array<float, kHistory> history{};
            size_t historyHead = 0;
            size_t historyCount = 0;
            double windowSeconds = 0.0;
            double windowMax = 0.0;
            uint32_t windowPresents = 0;
            uint64_t windowField = 0;
            uint64_t windowSequence = 0;
            bool windowPrimed = false;
            double hostFps = 0.0;
            double avgMs = 0.0;
            double maxMs = 0.0;
            double fieldRate = 0.0;
            double gsRate = 0.0;
            bool shared = false;
        };

        State &state()
        {
            static State s;
            return s;
        }

        void sampleCounters(State &s, double elapsed)
        {
            for (Counter &counter : s.counters)
            {
                const uint64_t value = counter.read();
                counter.rate = counter.primed && elapsed > 0.0 ? static_cast<double>((value - counter.last) & counter.wrapMask) / elapsed : 0.0;
                counter.last = value;
                counter.primed = true;
            }
        }

        Color frameColor(float ms)
        {
            if (ms <= 17.5f)
                return Color{96, 220, 120, 230};
            if (ms <= 34.5f)
                return Color{240, 200, 80, 230};
            return Color{240, 90, 80, 230};
        }
    }

    void addCounter(std::string label, std::function<uint64_t()> read, uint64_t wrapMask)
    {
        State &s = state();
        std::lock_guard lock(s.mutex);
        s.counters.push_back(Counter{std::move(label), std::move(read), wrapMask});
    }

    void setVisible(bool visible)
    {
        State &s = state();
        std::lock_guard lock(s.mutex);
        s.visible = visible;
    }

    bool visible()
    {
        State &s = state();
        std::lock_guard lock(s.mutex);
        return s.visible;
    }

    void toggle()
    {
        State &s = state();
        std::lock_guard lock(s.mutex);
        s.visible = !s.visible;
    }

    void recordPresent(double intervalSeconds, uint64_t field, bool sharedPath, uint64_t sharedSequence)
    {
        State &s = state();
        std::lock_guard lock(s.mutex);
        s.shared = sharedPath;
        if (!s.windowPrimed)
        {
            s.windowPrimed = true;
            s.windowField = field;
            s.windowSequence = sharedSequence;
            sampleCounters(s, 0.0);
            return;
        }
        s.history[s.historyHead] = static_cast<float>(intervalSeconds * 1000.0);
        s.historyHead = (s.historyHead + 1) % kHistory;
        s.historyCount = std::min(s.historyCount + 1, kHistory);
        s.windowSeconds += intervalSeconds;
        s.windowMax = std::max(s.windowMax, intervalSeconds);
        ++s.windowPresents;
        if (s.windowSeconds < kSampleSeconds)
            return;
        const double elapsed = s.windowSeconds;
        s.hostFps = s.windowPresents / elapsed;
        s.avgMs = elapsed * 1000.0 / s.windowPresents;
        s.maxMs = s.windowMax * 1000.0;
        s.fieldRate = static_cast<double>(field - s.windowField) / elapsed;
        s.gsRate = sharedSequence >= s.windowSequence ? static_cast<double>(sharedSequence - s.windowSequence) / elapsed : 0.0;
        sampleCounters(s, elapsed);
        s.windowField = field;
        s.windowSequence = sharedSequence;
        s.windowSeconds = 0.0;
        s.windowMax = 0.0;
        s.windowPresents = 0;
    }

    void draw()
    {
        State &s = state();
        std::lock_guard lock(s.mutex);
        if (!s.visible)
            return;

        const int screenWidth = GetScreenWidth();
        const int screenHeight = GetScreenHeight();
        const float scale = std::max(1.0f, static_cast<float>(screenHeight) / 720.0f);
        const int fontSize = static_cast<int>(std::lround(18.0f * scale));
        const int lineHeight = fontSize + static_cast<int>(std::lround(4.0f * scale));
        const int pad = static_cast<int>(std::lround(10.0f * scale));

        std::vector<std::string> lines;
        char buffer[160];
        std::snprintf(buffer, sizeof(buffer), "FPS %.1f", s.hostFps);
        lines.emplace_back(buffer);
        std::snprintf(buffer, sizeof(buffer), "Frame %.2f ms  max %.2f ms", s.avgMs, s.maxMs);
        lines.emplace_back(buffer);
        for (const Counter &counter : s.counters)
        {
            std::snprintf(buffer, sizeof(buffer), "%s %.1f/s", counter.label.c_str(), counter.rate);
            lines.emplace_back(buffer);
        }
        if (s.shared)
        {
            std::snprintf(buffer, sizeof(buffer), "GS frames %.1f/s", s.gsRate);
            lines.emplace_back(buffer);
        }
        std::snprintf(buffer, sizeof(buffer), "VBlank %.2f/s", s.fieldRate);
        lines.emplace_back(buffer);
        std::snprintf(buffer, sizeof(buffer), "VSync %s  %dx%d", IsWindowState(FLAG_VSYNC_HINT) ? "on" : "off", screenWidth, screenHeight);
        lines.emplace_back(buffer);

        const int graphWidth = static_cast<int>(std::lround(static_cast<float>(kHistory) * scale));
        const int graphHeight = static_cast<int>(std::lround(64.0f * scale));
        int textWidth = 0;
        for (const std::string &line : lines)
            textWidth = std::max(textWidth, MeasureText(line.c_str(), fontSize));
        const int panelWidth = std::max(textWidth, graphWidth) + pad * 2;
        const int panelHeight = pad * 3 + lineHeight * static_cast<int>(lines.size()) + graphHeight;
        const int panelX = screenWidth - panelWidth - pad;
        const int panelY = pad;

        DrawRectangle(panelX, panelY, panelWidth, panelHeight, Fade(BLACK, 0.65f));
        int y = panelY + pad;
        for (size_t i = 0; i < lines.size(); ++i)
        {
            const Color color = i == 0 ? frameColor(static_cast<float>(s.avgMs)) : RAYWHITE;
            DrawText(lines[i].c_str(), panelX + pad, y, fontSize, color);
            y += lineHeight;
        }

        const int graphX = panelX + pad;
        const int graphY = y + pad;
        DrawRectangle(graphX, graphY, graphWidth, graphHeight, Fade(DARKGRAY, 0.5f));
        const float barWidth = static_cast<float>(graphWidth) / static_cast<float>(kHistory);
        for (size_t i = 0; i < s.historyCount; ++i)
        {
            const size_t index = (s.historyHead + kHistory - s.historyCount + i) % kHistory;
            const float ms = s.history[index];
            const float height = std::min(ms, kGraphCeilingMs) / kGraphCeilingMs * static_cast<float>(graphHeight);
            const float x = static_cast<float>(graphX) + barWidth * static_cast<float>(kHistory - s.historyCount + i);
            DrawRectangleRec(Rectangle{x, static_cast<float>(graphY + graphHeight) - height, std::max(1.0f, barWidth), height}, frameColor(ms));
        }
        for (const float target : {1000.0f / 60.0f, 1000.0f / 30.0f})
        {
            const int lineY = graphY + graphHeight - static_cast<int>(std::lround(target / kGraphCeilingMs * static_cast<float>(graphHeight)));
            DrawLine(graphX, lineY, graphX + graphWidth, lineY, Fade(RAYWHITE, 0.45f));
        }
    }
}
