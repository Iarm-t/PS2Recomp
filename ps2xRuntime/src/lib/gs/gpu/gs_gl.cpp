#include "gs_gl.h"
#include "runtime/gs/gs_shared_present.h"

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <string_view>

namespace
{
    template <typename T, typename Loader>
    bool loadEntry(T &target, const char *name, std::string &missing, Loader loadProc)
    {
        target = reinterpret_cast<T>(loadProc(name));
        if (!target)
        {
            missing += missing.empty() ? name : std::string(", ") + name;
            return false;
        }
        return true;
    }

    // Resolves every entry point the GPU backend uses through the platform's GL loader.
    template <typename Loader>
    void loadContextApi(GSGL::Api &a, std::string &missing, Loader loadProc)
    {
        const auto load = [&](auto &target, const char *name, std::string &list) { return loadEntry(target, name, list, loadProc); };
        load(a.GenBuffers, "glGenBuffers", missing);
        load(a.DeleteBuffers, "glDeleteBuffers", missing);
        load(a.BindBuffer, "glBindBuffer", missing);
        load(a.BufferData, "glBufferData", missing);
        load(a.BufferSubData, "glBufferSubData", missing);
        load(a.GetBufferSubData, "glGetBufferSubData", missing);
        load(a.BindBufferBase, "glBindBufferBase", missing);
        load(a.BindBufferRange, "glBindBufferRange", missing);
        load(a.ClearBufferData, "glClearBufferData", missing);
        load(a.CopyBufferSubData, "glCopyBufferSubData", missing);
        load(a.BufferStorage, "glBufferStorage", missing);
        load(a.MapBufferRange, "glMapBufferRange", missing);
        load(a.UnmapBuffer, "glUnmapBuffer", missing);
        load(a.GenQueries, "glGenQueries", missing);
        load(a.DeleteQueries, "glDeleteQueries", missing);
        load(a.BeginQuery, "glBeginQuery", missing);
        load(a.EndQuery, "glEndQuery", missing);
        load(a.GetQueryObjectui64v, "glGetQueryObjectui64v", missing);
        load(a.QueryCounter, "glQueryCounter", missing);
        load(a.CreateShader, "glCreateShader", missing);
        load(a.ShaderSource, "glShaderSource", missing);
        load(a.CompileShader, "glCompileShader", missing);
        load(a.GetShaderiv, "glGetShaderiv", missing);
        load(a.GetShaderInfoLog, "glGetShaderInfoLog", missing);
        load(a.DeleteShader, "glDeleteShader", missing);
        load(a.CreateProgram, "glCreateProgram", missing);
        load(a.AttachShader, "glAttachShader", missing);
        load(a.LinkProgram, "glLinkProgram", missing);
        load(a.GetProgramiv, "glGetProgramiv", missing);
        load(a.GetProgramInfoLog, "glGetProgramInfoLog", missing);
        load(a.ProgramParameteri, "glProgramParameteri", missing);
        load(a.GetProgramBinary, "glGetProgramBinary", missing);
        load(a.ProgramBinary, "glProgramBinary", missing);
        load(a.DeleteProgram, "glDeleteProgram", missing);
        load(a.UseProgram, "glUseProgram", missing);
        load(a.GetUniformLocation, "glGetUniformLocation", missing);
        load(a.Uniform1ui, "glUniform1ui", missing);
        load(a.Uniform4ui, "glUniform4ui", missing);
        load(a.Uniform1uiv, "glUniform1uiv", missing);
        load(a.DispatchCompute, "glDispatchCompute", missing);
        load(a.MemBarrier, "glMemoryBarrier", missing);
        load(a.FenceSync, "glFenceSync", missing);
        load(a.ClientWaitSync, "glClientWaitSync", missing);
        load(a.DeleteSync, "glDeleteSync", missing);
        load(a.Finish, "glFinish", missing);
        load(a.Flush, "glFlush", missing);
        load(a.GetError, "glGetError", missing);
        load(a.GetString, "glGetString", missing);
        load(a.GenTextures, "glGenTextures", missing);
        load(a.DeleteTextures, "glDeleteTextures", missing);
        load(a.BindTexture, "glBindTexture", missing);
        load(a.TexParameteri, "glTexParameteri", missing);
        load(a.TexImage2D, "glTexImage2D", missing);
        load(a.TexSubImage2D, "glTexSubImage2D", missing);
        {
            std::string graphicsMissing;
            load(a.GenFramebuffers, "glGenFramebuffers", graphicsMissing);
            load(a.DeleteFramebuffers, "glDeleteFramebuffers", graphicsMissing);
            load(a.BindFramebuffer, "glBindFramebuffer", graphicsMissing);
            load(a.FramebufferParameteri, "glFramebufferParameteri", graphicsMissing);
            load(a.Viewport, "glViewport", graphicsMissing);
            load(a.DrawArrays, "glDrawArrays", graphicsMissing);
            load(a.GenVertexArrays, "glGenVertexArrays", graphicsMissing);
            load(a.DeleteVertexArrays, "glDeleteVertexArrays", graphicsMissing);
            load(a.BindVertexArray, "glBindVertexArray", graphicsMissing);
            load(a.Enable, "glEnable", graphicsMissing);
            load(a.Disable, "glDisable", graphicsMissing);
            std::string parallelMissing;
            load(a.MaxShaderCompilerThreads, "glMaxShaderCompilerThreadsARB", parallelMissing);
            if (!a.MaxShaderCompilerThreads)
                load(a.MaxShaderCompilerThreads, "glMaxShaderCompilerThreadsKHR", parallelMissing);
            a.graphics = graphicsMissing.empty();
        }
    }
}

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

namespace
{
    constexpr int kContextMajorVersion = 0x2091;
    constexpr int kContextMinorVersion = 0x2092;
    constexpr int kContextFlags = 0x2094;
    constexpr int kContextProfileMask = 0x9126;
    constexpr int kContextCoreProfileBit = 0x0001;

    using CreateContextAttribs = HGLRC(WINAPI *)(HDC, HGLRC, const int *);

    struct SharedSlot
    {
        GSSharedPresent::Frame frame;
        GSGL::GLsync produced = nullptr;
        GSGL::GLsync consumed = nullptr;
        GSGL::GLuint hashBuffer = 0;
        GSGL::GLuint hashDeviceBuffer = 0;
        const uint32_t *hash = nullptr;
        bool reading = false;
    };

    std::mutex s_sharedMutex;
    std::array<SharedSlot, 8> s_sharedSlots;
    GSGL::Api s_hostApi;
    HGLRC s_hostContext = nullptr;
    HDC s_hostDc = nullptr;
    PIXELFORMATDESCRIPTOR s_hostFormat{};
    int s_hostFormatIndex = 0;
    int s_readingSlot = -1;
    uint64_t s_sharedSequence = 0;
    std::atomic<bool> s_sharedActive{false};

    bool completed(const GSGL::Api &gl, GSGL::GLsync fence)
    {
        if (!fence)
            return true;
        const auto status = gl.ClientWaitSync(fence, 0u, 0u);
        return status == GSGL::kAlreadySignaled || status == GSGL::kConditionSatisfied;
    }

    void releaseReading()
    {
        if (s_readingSlot < 0)
            return;
        SharedSlot &slot = s_sharedSlots[s_readingSlot];
        slot.consumed = s_hostApi.FenceSync(GSGL::kSyncGpuCommandsComplete, 0u);
        s_hostApi.Flush();
        slot.reading = false;
        s_readingSlot = -1;
    }

    HMODULE openGlModule()
    {
        static HMODULE module = LoadLibraryA("opengl32.dll");
        return module;
    }

    void *loadProc(const char *name)
    {
        void *proc = reinterpret_cast<void *>(wglGetProcAddress(name));
        const auto value = reinterpret_cast<intptr_t>(proc);
        if (value == 0 || value == 1 || value == 2 || value == 3 || value == -1)
            proc = reinterpret_cast<void *>(GetProcAddress(openGlModule(), name));
        return proc;
    }

    template <typename T>
    bool load(T &target, const char *name, std::string &missing)
    {
        target = reinterpret_cast<T>(loadProc(name));
        if (!target)
        {
            missing += missing.empty() ? name : std::string(", ") + name;
            return false;
        }
        return true;
    }

    LRESULT CALLBACK windowProc(HWND window, UINT message, WPARAM wparam, LPARAM lparam)
    {
        return DefWindowProcA(window, message, wparam, lparam);
    }
}

namespace GSSharedPresent
{
    void CaptureHostContext()
    {
        const char *value = std::getenv("PS2X_GS_DIRECT_PRESENT");
        if (!value || *value != '1')
            return;
        const HDC dc = wglGetCurrentDC();
        const HGLRC context = wglGetCurrentContext();
        if (!dc || !context)
            return;
        std::string missing;
        GSGL::Api api;
        load(api.FenceSync, "glFenceSync", missing);
        load(api.ClientWaitSync, "glClientWaitSync", missing);
        load(api.DeleteSync, "glDeleteSync", missing);
        load(api.Flush, "glFlush", missing);
        if (!missing.empty())
        {
            std::fprintf(stderr, "[gs-present] shared presentation unavailable: %s\n", missing.c_str());
            return;
        }
        std::lock_guard<std::mutex> lock(s_sharedMutex);
        s_hostFormatIndex = GetPixelFormat(dc);
        if (!DescribePixelFormat(dc, s_hostFormatIndex, sizeof(s_hostFormat), &s_hostFormat))
            return;
        s_hostApi = api;
        s_hostContext = context;
        s_hostDc = dc;
        wglMakeCurrent(nullptr, nullptr);
    }

    void RestoreHostContext()
    {
        if (s_hostDc && s_hostContext)
            wglMakeCurrent(s_hostDc, s_hostContext);
    }

    bool Active()
    {
        return s_sharedActive.load(std::memory_order_acquire);
    }

    bool Acquire(Frame &frame)
    {
        if (!Active())
            return false;
        std::lock_guard<std::mutex> lock(s_sharedMutex);
        int newest = -1;
        uint64_t sequence = s_readingSlot >= 0 ? s_sharedSlots[s_readingSlot].frame.sequence : 0u;
        for (int i = 0; i < static_cast<int>(s_sharedSlots.size()); ++i)
        {
            const SharedSlot &slot = s_sharedSlots[i];
            if (!slot.reading && slot.frame.sequence > sequence && slot.produced && completed(s_hostApi, slot.produced))
            {
                newest = i;
                sequence = slot.frame.sequence;
            }
        }
        if (newest >= 0)
        {
            releaseReading();
            SharedSlot &slot = s_sharedSlots[newest];
            s_hostApi.DeleteSync(slot.produced);
            slot.produced = nullptr;
            if (slot.frame.hashValid)
                slot.frame.contentHash = static_cast<uint64_t>(slot.hash[0]) | (static_cast<uint64_t>(slot.hash[1]) << 32u);
            slot.reading = true;
            s_readingSlot = newest;
        }
        if (s_readingSlot < 0)
            return false;
        frame = s_sharedSlots[s_readingSlot].frame;
        return true;
    }

    void ShutdownHost()
    {
        std::lock_guard<std::mutex> lock(s_sharedMutex);
        releaseReading();
        s_sharedActive.store(false, std::memory_order_release);
        s_hostContext = nullptr;
        s_hostDc = nullptr;
    }
}

namespace GSGL
{
    bool SharedAvailable(const Api &gl)
    {
        std::lock_guard<std::mutex> lock(s_sharedMutex);
        for (const SharedSlot &slot : s_sharedSlots)
            if (!slot.reading && completed(gl, slot.produced) && completed(gl, slot.consumed))
                return true;
        return false;
    }

    bool PublishShared(const Api &gl, GLuint buffer, uint32_t width, uint32_t height, uint64_t renderSequence,
                       GLuint hashProgram, GLint hashLocation)
    {
        std::lock_guard<std::mutex> lock(s_sharedMutex);
        if (!s_sharedActive.load(std::memory_order_relaxed))
            return false;
        SharedSlot *available = nullptr;
        for (SharedSlot &slot : s_sharedSlots)
        {
            if (slot.reading || !completed(gl, slot.produced) || !completed(gl, slot.consumed))
                continue;
            if (!available || slot.frame.sequence < available->frame.sequence)
                available = &slot;
        }
        if (!available)
            return false;
        SharedSlot &slot = *available;
        if (slot.produced)
            gl.DeleteSync(slot.produced);
        if (slot.consumed)
            gl.DeleteSync(slot.consumed);
        slot.produced = slot.consumed = nullptr;
        slot.frame.hashValid = false;
        if (hashProgram)
        {
            if (!slot.hashBuffer)
            {
                gl.GenBuffers(1, &slot.hashBuffer);
                gl.BindBuffer(kShaderStorageBuffer, slot.hashBuffer);
                const GLbitfield flags = kMapReadBit | kMapPersistentBit | kMapCoherentBit;
                gl.BufferStorage(kShaderStorageBuffer, 8, nullptr, flags);
                slot.hash = static_cast<const uint32_t *>(gl.MapBufferRange(kShaderStorageBuffer, 0, 8, flags));
            }
            if (slot.hash)
            {
                static const bool deviceHash = [] {
                    const char *setting = std::getenv("PS2X_FRAME_HASH_DEVICE");
                    return !setting || std::string_view(setting) != "0";
                }();
                if (deviceHash && !slot.hashDeviceBuffer)
                {
                    gl.GenBuffers(1, &slot.hashDeviceBuffer);
                    gl.BindBuffer(kShaderStorageBuffer, slot.hashDeviceBuffer);
                    gl.BufferData(kShaderStorageBuffer, 8, nullptr, kDynamicDraw);
                }
                const GLuint output = deviceHash ? slot.hashDeviceBuffer : slot.hashBuffer;
                const uint32_t zero = 0u;
                gl.BindBuffer(kShaderStorageBuffer, output);
                gl.ClearBufferData(kShaderStorageBuffer, kR32UI, kRedInteger, kUnsignedInt, &zero);
                gl.BindBufferBase(kShaderStorageBuffer, 7u, buffer);
                gl.BindBufferBase(kShaderStorageBuffer, 10u, output);
                gl.MemBarrier(kAllBarrierBits);
                gl.UseProgram(hashProgram);
                gl.Uniform1ui(hashLocation, width * height);
                gl.DispatchCompute((width * height + 255u) / 256u, 1u, 1u);
                gl.MemBarrier(kAllBarrierBits);
                if (deviceHash)
                {
                    gl.BindBuffer(kCopyReadBuffer, output);
                    gl.BindBuffer(kCopyWriteBuffer, slot.hashBuffer);
                    gl.CopyBufferSubData(kCopyReadBuffer, kCopyWriteBuffer, 0, 0, 8);
                    gl.MemBarrier(kAllBarrierBits);
                }
                slot.frame.hashValid = true;
            }
        }
        gl.BindBuffer(kPixelUnpackBuffer, 0u);
        if (!slot.frame.texture)
            gl.GenTextures(1, &slot.frame.texture);
        gl.BindTexture(kTexture2D, slot.frame.texture);
        if (slot.frame.width != width || slot.frame.height != height)
        {
            gl.TexImage2D(kTexture2D, 0, kRgba8, width, height, 0, kRgba, kUnsignedByte, nullptr);
            gl.TexParameteri(kTexture2D, kTextureMinFilter, kLinear);
            gl.TexParameteri(kTexture2D, kTextureMagFilter, kLinear);
            gl.TexParameteri(kTexture2D, kTextureWrapS, kClampToEdge);
            gl.TexParameteri(kTexture2D, kTextureWrapT, kClampToEdge);
        }
        gl.MemBarrier(kAllBarrierBits);
        gl.BindBuffer(kPixelUnpackBuffer, buffer);
        gl.TexSubImage2D(kTexture2D, 0, 0, 0, width, height, kRgba, kUnsignedByte, nullptr);
        gl.MemBarrier(kAllBarrierBits);
        gl.BindBuffer(kPixelUnpackBuffer, 0u);
        gl.BindTexture(kTexture2D, 0u);
        slot.frame.width = width;
        slot.frame.height = height;
        slot.frame.sequence = ++s_sharedSequence;
        slot.frame.renderSequence = renderSequence;
        slot.produced = gl.FenceSync(kSyncGpuCommandsComplete, 0u);
        gl.Flush();
        return true;
    }

    void DestroyShared(const Api &gl)
    {
        std::lock_guard<std::mutex> lock(s_sharedMutex);
        s_sharedActive.store(false, std::memory_order_release);
        for (SharedSlot &slot : s_sharedSlots)
        {
            if (slot.produced)
                gl.DeleteSync(slot.produced);
            if (slot.consumed)
                gl.DeleteSync(slot.consumed);
            if (slot.frame.texture)
                gl.DeleteTextures(1, &slot.frame.texture);
            if (slot.hashBuffer)
                gl.DeleteBuffers(1, &slot.hashBuffer);
            if (slot.hashDeviceBuffer)
                gl.DeleteBuffers(1, &slot.hashDeviceBuffer);
            slot = {};
        }
        s_readingSlot = -1;
    }

    Context::~Context()
    {
        Destroy();
    }

    bool Context::Create(std::string &error)
    {
        Destroy();
        WNDCLASSA windowClass{};
        windowClass.style = CS_OWNDC;
        windowClass.lpfnWndProc = windowProc;
        windowClass.hInstance = GetModuleHandleA(nullptr);
        windowClass.lpszClassName = "PS2XGSGpuContext";
        RegisterClassA(&windowClass);
        HWND window = CreateWindowExA(0, windowClass.lpszClassName, "gs", WS_OVERLAPPEDWINDOW, 0, 0, 16, 16, nullptr, nullptr,
                                      windowClass.hInstance, nullptr);
        if (!window)
        {
            error = "CreateWindowEx failed";
            return false;
        }
        HDC dc = GetDC(window);
        PIXELFORMATDESCRIPTOR descriptor{};
        descriptor.nSize = sizeof(descriptor);
        descriptor.nVersion = 1;
        descriptor.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
        descriptor.iPixelType = PFD_TYPE_RGBA;
        descriptor.cColorBits = 32;
        const int format = s_hostContext ? s_hostFormatIndex : ChoosePixelFormat(dc, &descriptor);
        if (s_hostContext)
            descriptor = s_hostFormat;
        if (format == 0 || !SetPixelFormat(dc, format, &descriptor))
        {
            ReleaseDC(window, dc);
            DestroyWindow(window);
            error = "no pixel format";
            return false;
        }
        HGLRC legacy = wglCreateContext(dc);
        if (!legacy || !wglMakeCurrent(dc, legacy))
        {
            if (legacy)
                wglDeleteContext(legacy);
            ReleaseDC(window, dc);
            DestroyWindow(window);
            error = "wglCreateContext failed";
            return false;
        }
        const auto createAttribs = reinterpret_cast<CreateContextAttribs>(wglGetProcAddress("wglCreateContextAttribsARB"));
        HGLRC context = nullptr;
        if (createAttribs)
        {
            const int attributes[] = {kContextMajorVersion, 4, kContextMinorVersion, 6, kContextProfileMask, kContextCoreProfileBit,
                                      kContextFlags, 0, 0};
            context = createAttribs(dc, s_hostContext, attributes);
            m_shared = context && s_hostContext;
            if (!context && s_hostContext)
            {
                std::fprintf(stderr, "[gs-present] WGL sharing failed (%lu); using RAM presentation\n", GetLastError());
                context = createAttribs(dc, nullptr, attributes);
            }
        }
        wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(legacy);
        if (!context || !wglMakeCurrent(dc, context))
        {
            if (context)
                wglDeleteContext(context);
            ReleaseDC(window, dc);
            DestroyWindow(window);
            error = "OpenGL 4.6 core context unavailable";
            return false;
        }
        m_window = window;
        m_dc = dc;
        m_context = context;

        std::string missing;
        loadContextApi(m_api, missing, loadProc);
        if (!missing.empty())
        {
            error = "missing GL functions: " + missing;
            Destroy();
            return false;
        }
        if (m_shared)
        {
            s_sharedActive.store(true, std::memory_order_release);
            std::fprintf(stderr, "[gs-present] shared GPU textures active\n");
        }
        return true;
    }

    void *Context::CreateWorker() const
    {
        const auto createAttribs = reinterpret_cast<CreateContextAttribs>(wglGetProcAddress("wglCreateContextAttribsARB"));
        if (!createAttribs || !m_context || !m_dc)
            return nullptr;
        const int attributes[] = {kContextMajorVersion, 4, kContextMinorVersion, 6, kContextProfileMask, kContextCoreProfileBit,
                                  kContextFlags, 0, 0};
        return createAttribs(static_cast<HDC>(m_dc), static_cast<HGLRC>(m_context), attributes);
    }

    bool Context::MakeWorkerCurrent(void *worker) const
    {
        return worker && m_dc && wglMakeCurrent(static_cast<HDC>(m_dc), static_cast<HGLRC>(worker));
    }

    void Context::DestroyWorker(void *worker)
    {
        if (!worker)
            return;
        if (wglGetCurrentContext() == static_cast<HGLRC>(worker))
            wglMakeCurrent(nullptr, nullptr);
        wglDeleteContext(static_cast<HGLRC>(worker));
    }

    void Context::Destroy()
    {
        if (m_context)
        {
            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(static_cast<HGLRC>(m_context));
            m_context = nullptr;
        }
        if (m_dc && m_window)
            ReleaseDC(static_cast<HWND>(m_window), static_cast<HDC>(m_dc));
        m_dc = nullptr;
        if (m_window)
            DestroyWindow(static_cast<HWND>(m_window));
        m_window = nullptr;
        m_api = {};
        m_shared = false;
    }
}
#elif defined(__linux__)
#include <EGL/egl.h>
#include <EGL/eglext.h>

namespace
{
    // One EGL display for the process; the GS contexts are surfaceless, so they never touch the host window's
    // GLX/EGL context and frames reach the window through RAM (no shared-texture presentation).
    std::mutex s_eglMutex;
    EGLDisplay s_eglDisplay = EGL_NO_DISPLAY;

    void *loadProc(const char *name)
    {
        return reinterpret_cast<void *>(eglGetProcAddress(name));
    }

    bool hasExtension(const char *list, std::string_view name)
    {
        if (!list)
            return false;
        const std::string_view extensions(list);
        size_t at = 0;
        while ((at = extensions.find(name, at)) != std::string_view::npos)
        {
            const size_t end = at + name.size();
            if ((at == 0 || extensions[at - 1] == ' ') && (end == extensions.size() || extensions[end] == ' '))
                return true;
            at = end;
        }
        return false;
    }

    EGLDisplay openDisplay(std::string &error)
    {
        std::lock_guard<std::mutex> lock(s_eglMutex);
        if (s_eglDisplay != EGL_NO_DISPLAY)
            return s_eglDisplay;
        EGLDisplay display = EGL_NO_DISPLAY;
        if (hasExtension(eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS), "EGL_MESA_platform_surfaceless"))
            display = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
        if (display == EGL_NO_DISPLAY)
            display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (display == EGL_NO_DISPLAY || !eglInitialize(display, nullptr, nullptr))
        {
            error = "no EGL display";
            return EGL_NO_DISPLAY;
        }
        if (!hasExtension(eglQueryString(display, EGL_EXTENSIONS), "EGL_KHR_surfaceless_context"))
        {
            error = "EGL_KHR_surfaceless_context unavailable";
            return EGL_NO_DISPLAY;
        }
        s_eglDisplay = display;
        return display;
    }

    EGLConfig chooseConfig(EGLDisplay display)
    {
        const EGLint attributes[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE};
        EGLConfig config = nullptr;
        EGLint count = 0;
        if (!eglChooseConfig(display, attributes, &config, 1, &count) || count < 1)
            return nullptr;
        return config;
    }

    EGLContext createCoreContext(EGLDisplay display, EGLConfig config, EGLContext share)
    {
        const EGLint attributes[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 6, EGL_CONTEXT_OPENGL_PROFILE_MASK,
                                     EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
        return eglCreateContext(display, config, share, attributes);
    }
}

namespace GSGL
{
    bool PublishShared(const Api &, GLuint, uint32_t, uint32_t, uint64_t, GLuint, GLint) { return false; }
    bool SharedAvailable(const Api &) { return false; }
    void DestroyShared(const Api &) {}

    Context::~Context()
    {
        Destroy();
    }

    bool Context::Create(std::string &error)
    {
        Destroy();
        EGLDisplay display = openDisplay(error);
        if (display == EGL_NO_DISPLAY)
            return false;
        if (!eglBindAPI(EGL_OPENGL_API))
        {
            error = "EGL cannot bind the desktop OpenGL API";
            return false;
        }
        EGLConfig config = chooseConfig(display);
        EGLContext context = config ? createCoreContext(display, config, EGL_NO_CONTEXT) : EGL_NO_CONTEXT;
        if (context == EGL_NO_CONTEXT || !eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context))
        {
            if (context != EGL_NO_CONTEXT)
                eglDestroyContext(display, context);
            char code[16];
            std::snprintf(code, sizeof(code), "0x%04X", static_cast<unsigned>(eglGetError()));
            error = std::string("OpenGL 4.6 core context unavailable (EGL error ") + code + ")";
            return false;
        }
        m_dc = display;
        m_window = config;
        m_context = context;

        std::string missing;
        loadContextApi(m_api, missing, loadProc);
        if (!missing.empty())
        {
            error = "missing GL functions: " + missing;
            Destroy();
            return false;
        }
        return true;
    }

    void *Context::CreateWorker() const
    {
        if (!m_context || !m_dc)
            return nullptr;
        eglBindAPI(EGL_OPENGL_API);
        EGLContext worker = createCoreContext(static_cast<EGLDisplay>(m_dc), static_cast<EGLConfig>(m_window),
                                              static_cast<EGLContext>(m_context));
        return worker == EGL_NO_CONTEXT ? nullptr : worker;
    }

    bool Context::MakeWorkerCurrent(void *worker) const
    {
        if (!worker || !m_dc)
            return false;
        eglBindAPI(EGL_OPENGL_API);
        return eglMakeCurrent(static_cast<EGLDisplay>(m_dc), EGL_NO_SURFACE, EGL_NO_SURFACE, static_cast<EGLContext>(worker));
    }

    void Context::DestroyWorker(void *worker)
    {
        if (!worker || s_eglDisplay == EGL_NO_DISPLAY)
            return;
        if (eglGetCurrentContext() == static_cast<EGLContext>(worker))
            eglMakeCurrent(s_eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(s_eglDisplay, static_cast<EGLContext>(worker));
    }

    void Context::Destroy()
    {
        if (m_context && m_dc)
        {
            if (eglGetCurrentContext() == static_cast<EGLContext>(m_context))
                eglMakeCurrent(static_cast<EGLDisplay>(m_dc), EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            eglDestroyContext(static_cast<EGLDisplay>(m_dc), static_cast<EGLContext>(m_context));
        }
        m_context = nullptr;
        m_dc = nullptr;
        m_window = nullptr;
        m_api = {};
        m_shared = false;
    }
}
namespace GSSharedPresent
{
    void CaptureHostContext() {}
    void RestoreHostContext() {}
    bool Active() { return false; }
    bool Acquire(Frame &) { return false; }
    void ShutdownHost() {}
}
#else
namespace GSGL
{
    bool PublishShared(const Api &, GLuint, uint32_t, uint32_t, uint64_t, GLuint, GLint) { return false; }
    bool SharedAvailable(const Api &) { return false; }
    void DestroyShared(const Api &) {}
    Context::~Context() = default;

    bool Context::Create(std::string &error)
    {
        error = "GPU GS backend is only implemented for Windows";
        return false;
    }

    void *Context::CreateWorker() const { return nullptr; }
    bool Context::MakeWorkerCurrent(void *) const { return false; }
    void Context::DestroyWorker(void *) {}

    void Context::Destroy()
    {
    }
}
namespace GSSharedPresent
{
    void CaptureHostContext() {}
    void RestoreHostContext() {}
    bool Active() { return false; }
    bool Acquire(Frame &) { return false; }
    void ShutdownHost() {}
}
#endif
