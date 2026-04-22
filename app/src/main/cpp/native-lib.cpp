/**
 * renderer.cpp
 *
 * Native OpenGL ES 3.0 renderer for Android using the NDK.
 * Renders a rotating triangle, driven by tap input from the Java layer.
 *
 * Architecture overview:
 *   - Java/Kotlin calls initGL(), render(), setViewport(), and onTap() via JNI.
 *   - initGL()     : Sets up EGL display/surface/context and compiles shaders.
 *   - render()     : Called every frame; smoothly interpolates rotation and draws.
 *   - setViewport(): Called on surface resize events.
 *   - onTap()      : Increments the target rotation angle by 90°.
 *
 * Threading note:
 *   targetAngle is an atomic<float> because onTap() may be called from the UI
 *   thread while render() runs on the GL/render thread.
 */

#include <jni.h>
#include <android/native_window_jni.h>
#include <android/log.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <cmath>
#include <atomic>

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

#define LOG_TAG "GAME"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// width/height
static int gWidth;
static int gHeight;

// ---------------------------------------------------------------------------
// EGL state
//
// These represent the connection between the Android surface and the OpenGL
// ES driver. They are initialised once in initGLInternal() and remain valid
// for the lifetime of the surface.
// ---------------------------------------------------------------------------

static EGLDisplay sDisplay = EGL_NO_DISPLAY;
static EGLSurface sSurface = EGL_NO_SURFACE;
static EGLContext sContext = EGL_NO_CONTEXT;

// ---------------------------------------------------------------------------
// GL objects
// ---------------------------------------------------------------------------

static GLuint sProgram = 0;  // Linked shader program
static GLuint sVbo     = 0;  // Vertex Buffer Object — stores triangle vertices
static GLuint sVao     = 0;  // Vertex Array Object  — records attribute layout
static GLuint sFbo = 0;
static GLuint sColorTexture = 0;
static GLuint sQuadVao = 0;
static GLuint sQuadVbo = 0;
static GLuint sBlurProgram = 0;

// Location of the "uAngle" uniform inside the vertex shader.
static GLint sUAngleLoc = -1;
static GLint sUScaleLoc = -1;

// ---------------------------------------------------------------------------
// Rotation state
//
// targetAngle  : where we want to end up (written by the UI thread via onTap).
// currentAngle : the angle actually passed to the shader each frame, smoothed
//                toward targetAngle with a simple lerp.
// ---------------------------------------------------------------------------

static std::atomic<float> sTargetAngle{0.0f};
static float sCurrentAngle = 0.0f;

// How fast currentAngle chases targetAngle (0 = never moves, 1 = instant snap).
static constexpr float kRotationLerpSpeed = 0.1f;

// Amount added to targetAngle per tap, in radians (90°).
static constexpr float kTapRotationStep = static_cast<float>(M_PI) / 2.0f;

// ---------------------------------------------------------------------------
// Magnification state
//
// sMagnified      : toggled by double-tap; true = zoomed in.
// sTargetScale    : 2.0 when magnified, 1.0 when normal.
// sCurrentScale   : smoothly lerps toward sTargetScale each frame.
// kScaleLerpSpeed : how fast the zoom animates (matches rotation feel).
// kMagnifiedScale : the zoom factor applied when magnified.
// ---------------------------------------------------------------------------
static std::atomic<bool> sMagnified{false};
static float sTargetScale  = 1.0f;
static float sCurrentScale = 1.0f;

static constexpr float kScaleLerpSpeed = 0.1f;
static constexpr float kMagnifiedScale = 2.0f;

// ---------------------------------------------------------------------------
// Pending viewport resize — written by setViewport (any thread),
// consumed by render() (GL thread). Using atomics for thread safety.
// ---------------------------------------------------------------------------
static std::atomic<int> sPendingWidth{0};
static std::atomic<int> sPendingHeight{0};
static std::atomic<bool> sPendingResize{false};

// ---------------------------------------------------------------------------
// Forward declaration
// ---------------------------------------------------------------------------
static void initPostProcessing(int width, int height);
static void resizePostProcessing(int width, int height);

// ---------------------------------------------------------------------------
// Triangle geometry  (clip space, counter-clockwise winding)
// ---------------------------------------------------------------------------

static const GLfloat kVertices[] = {
        0.0f,  0.5f, 0.0f,   // top
        -0.5f, -0.5f, 0.0f,   // bottom-left
        0.5f, -0.5f, 0.0f,   // bottom-right
};
static constexpr GLsizei kVertexCount = 3;

// ---------------------------------------------------------------------------
// GLSL shaders
//
// Using C++11 raw string literals (R"(...)") keeps the GLSL readable and
// avoids manual escaping. This is the idiomatic modern approach for embedded
// shaders in Android NDK projects — a full asset pipeline (loading from
// APK assets) is the alternative, but is overkill for simple shaders.
// ---------------------------------------------------------------------------

// Vertex shader: rotates each vertex around the origin by uAngle (radians).
//
// NOTE: The #version directive must have no leading whitespace — the GLSL spec
// forbids anything before it. Do not reformat or re-indent these raw strings.
static const char* kVertexShaderSrc = R"GLSL(#version 300 es

layout(location = 0) in vec3 aPos;

uniform float uAngle;
uniform float uScale;

void main() {
    float c = cos(uAngle);
    float s = sin(uAngle);

    // 2-D rotation matrix applied to the XY plane
    vec2 rotated = vec2(
        aPos.x * c - aPos.y * s,
        aPos.x * s + aPos.y * c
    );

    gl_Position = vec4(rotated * uScale, 0.0, 1.0);
}
)GLSL";

// Fragment shader: outputs a solid red colour for every fragment.
static const char* kFragmentShaderSrc = R"GLSL(#version 300 es

precision mediump float;
out vec4 FragColor;

void main() {
    FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
)GLSL";

static const char* kBlurVertexShaderSrc = R"GLSL(#version 300 es
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aTex;

out vec2 TexCoords;

void main() {
    TexCoords = aTex;
    gl_Position = vec4(aPos, 0.0, 1.0);
}
)GLSL";

static const char* kBlurFragmentShaderSrc = R"GLSL(#version 300 es
precision mediump float;

in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D screenTexture;
uniform vec2 uTexelSize;

void main() {
    vec2 offsets[9] = vec2[](
        vec2(-uTexelSize.x,  uTexelSize.y),  // top-left
        vec2( 0.0,           uTexelSize.y),  // top-center
        vec2( uTexelSize.x,  uTexelSize.y),  // top-right
        vec2(-uTexelSize.x,  0.0),           // center-left
        vec2( 0.0,           0.0),           // center
        vec2( uTexelSize.x,  0.0),           // center-right
        vec2(-uTexelSize.x, -uTexelSize.y),  // bottom-left
        vec2( 0.0,          -uTexelSize.y),  // bottom-center
        vec2( uTexelSize.x, -uTexelSize.y)   // bottom-right
    );

    float kernel[9] = float[](
        1.0/16.0, 2.0/16.0, 1.0/16.0,
        2.0/16.0, 4.0/16.0, 2.0/16.0,
        1.0/16.0, 2.0/16.0, 1.0/16.0
    );

    vec3 col = vec3(0.0);

    for (int i = 0; i < 9; i++) {
        col += texture(screenTexture, TexCoords + offsets[i]).rgb * kernel[i];
    }

    FragColor = vec4(col, 1.0);
}
)GLSL";

// ---------------------------------------------------------------------------
// Shader helpers
// ---------------------------------------------------------------------------

/**
 * Compiles a single shader stage and returns its handle.
 * Logs the info log on failure; the caller should check GL_COMPILE_STATUS
 * if stricter error handling is needed.
 */
static GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint success = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[512];
        glGetShaderInfoLog(shader, sizeof(info), nullptr, info);
        LOGE("Shader compile error (type=0x%x): %s", type, info);
    }
    return shader;
}

/**
 * Links a vertex and fragment shader into a program.
 * Returns the program handle, or 0 on link failure.
 * Intermediate shader objects are deleted regardless of outcome.
 */
static GLuint buildProgram(const char* vertSrc, const char* fragSrc) {
    GLuint vert = compileShader(GL_VERTEX_SHADER,   vertSrc);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragSrc);

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);

    // Shader objects are no longer needed once linked into the program.
    glDeleteShader(vert);
    glDeleteShader(frag);

    GLint linked = GL_FALSE;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        char info[512];
        glGetProgramInfoLog(prog, sizeof(info), nullptr, info);
        LOGE("Program link error: %s", info);
        glDeleteProgram(prog);
        return 0;
    }

    return prog;
}

// ---------------------------------------------------------------------------
// Core initialisation — one function per responsibility
// ---------------------------------------------------------------------------

/**
 * initEGL()
 * Creates the EGL display, picks a config, and makes the context current.
 *
 * EGL is the platform glue between Android's windowing system and OpenGL ES.
 * This must succeed before any GL call is valid.
 */
static void initEGL(ANativeWindow* window) {
    sDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    eglInitialize(sDisplay, nullptr, nullptr);

    // Request an OpenGL ES 3 capable, RGB-888 window surface.
    const EGLint configAttribs[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
            EGL_SURFACE_TYPE,    EGL_WINDOW_BIT,
            EGL_RED_SIZE,        8,
            EGL_GREEN_SIZE,      8,
            EGL_BLUE_SIZE,       8,
            EGL_NONE
    };
    EGLConfig config;
    EGLint    numConfigs = 0;
    eglChooseConfig(sDisplay, configAttribs, &config, 1, &numConfigs);

    sSurface = eglCreateWindowSurface(sDisplay, config, window, nullptr);

    const EGLint ctxAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    sContext = eglCreateContext(sDisplay, config, EGL_NO_CONTEXT, ctxAttribs);

    eglMakeCurrent(sDisplay, sSurface, sSurface, sContext);
    LOGI("EGL initialised.");
}

/**
 * initGeometry()
 * Uploads the triangle's vertex data to the GPU and records the attribute
 * layout inside a VAO.
 *
 * The VAO captures the VBO binding and glVertexAttribPointer state so that
 * render() only needs a single glBindVertexArray() call per frame.
 */
static void initGeometry() {
    glGenVertexArrays(1, &sVao);
    glBindVertexArray(sVao);

    glGenBuffers(1, &sVbo);
    glBindBuffer(GL_ARRAY_BUFFER, sVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kVertices), kVertices, GL_STATIC_DRAW);

    // Attribute 0: vec3 position, tightly packed, no offset.
    glVertexAttribPointer(
            /*index=*/      0,
            /*size=*/       3,
            /*type=*/       GL_FLOAT,
            /*normalized=*/ GL_FALSE,
            /*stride=*/     3 * sizeof(GLfloat),
            /*offset=*/     nullptr
    );
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);  // Unbind so later code cannot accidentally mutate the VAO.
    LOGI("Geometry uploaded (%d vertices).", kVertexCount);
}

/**
 * initShaders()
 * Compiles and links the shader program, then caches uniform locations.
 *
 * Uniform locations are looked up once here rather than every frame;
 * glGetUniformLocation is not free and should never sit in the render loop.
 */
static void initShaders() {
    sProgram   = buildProgram(kVertexShaderSrc, kFragmentShaderSrc);
    sUAngleLoc = glGetUniformLocation(sProgram, "uAngle");
    sUScaleLoc = glGetUniformLocation(sProgram, "uScale");
    LOGI("Shaders compiled. uAngle location: %d", sUAngleLoc);
}

static void initPostProcessing(int width, int height) {
    // -----------------------------
    // 1. Create framebuffer
    // -----------------------------
    glGenFramebuffers(1, &sFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, sFbo);

    // -----------------------------
    // 2. Create color texture
    // -----------------------------
    glGenTextures(1, &sColorTexture);
    glBindTexture(GL_TEXTURE_2D, sColorTexture);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    // Filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // prevent edge artifacts
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Attach texture to framebuffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, sColorTexture, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        LOGE("Framebuffer not complete!");
    }

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        LOGE("FBO incomplete! status=0x%x", status);
    } else {
        LOGI("FBO complete at %dx%d", width, height);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // -----------------------------
    // 3. Fullscreen quad
    // -----------------------------
    float quadVertices[] = {
            -1.0f,  1.0f,  0.0f, 1.0f,
            -1.0f, -1.0f,  0.0f, 0.0f,
            1.0f, -1.0f,  1.0f, 0.0f,
            -1.0f,  1.0f,  0.0f, 1.0f,
            1.0f, -1.0f,  1.0f, 0.0f,
            1.0f,  1.0f,  1.0f, 1.0f
    };

    glGenVertexArrays(1, &sQuadVao);
    glGenBuffers(1, &sQuadVbo);

    glBindVertexArray(sQuadVao);
    glBindBuffer(GL_ARRAY_BUFFER, sQuadVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    // position (location = 0)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)nullptr);
    glEnableVertexAttribArray(0);

    // (location = 1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float),
                          (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    // -----------------------------
    // 4. Blur shader
    // -----------------------------
    sBlurProgram = buildProgram(kBlurVertexShaderSrc, kBlurFragmentShaderSrc);

    glUseProgram(sBlurProgram);

    glUniform1i(glGetUniformLocation(sBlurProgram, "screenTexture"), 0);
    glUniform2f(glGetUniformLocation(sBlurProgram, "uTexelSize"),
                1.0f / (float)width, 1.0f / (float)height);

    LOGI("Post-processing initialised at %dx%d.", width, height);
}

// Called on subsequent resizes only.
static void resizePostProcessing(int width, int height) {
    glBindTexture(GL_TEXTURE_2D, sColorTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                 GL_RGB, GL_UNSIGNED_BYTE, nullptr);

    glUseProgram(sBlurProgram);
    glUniform2f(glGetUniformLocation(sBlurProgram, "uTexelSize"),
                1.0f / (float)width, 1.0f / (float)height);

    LOGI("Post-processing resized to %dx%d.", width, height);
}

/**
 * initGLInternal()
 * Top-level coordinator: calls the three initialisation stages in order.
 * Each stage has a single, clearly named responsibility.
 *
 * Must be called from the GL/render thread.
 */
static void initGLInternal(ANativeWindow* window) {
    initEGL(window);       // 1. Connect to the display, create GL context
    initGeometry();        // 2. Upload mesh data to the GPU
    initShaders();         // 3. Compile shaders, cache uniform locations

    LOGI("GL initialisation complete. Renderer: %s", glGetString(GL_RENDERER));
}

// ---------------------------------------------------------------------------
// JNI entry points
// Called by MainActivity on the render/UI threads respectively.
// ---------------------------------------------------------------------------

/**
 * initGL(Surface surface)
 * Sets up EGL and OpenGL for the given Android Surface.
 * Called once when the SurfaceView is ready.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_initGL(JNIEnv* env, jobject /*thiz*/, jobject surfaceObj) {
    ANativeWindow* window = ANativeWindow_fromSurface(env, surfaceObj);
    initGLInternal(window);
    ANativeWindow_release(window);  // EGL holds its own reference; we can release ours.
}

/**
 * render()
 * Advances the rotation lerp, clears the screen, and draws the triangle.
 * Called every frame from the render loop (typically a Choreographer callback).
 */
extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_render(JNIEnv*, jobject) {
    // handle render resize/initialization
    if (sPendingResize.load(std::memory_order_acquire)) {
        sPendingResize.store(false, std::memory_order_relaxed);

        int w = sPendingWidth.load(std::memory_order_relaxed);
        int h = sPendingHeight.load(std::memory_order_relaxed);

        if (w > 0 && h > 0) {
            gWidth  = w;
            gHeight = h;
            glViewport(0, 0, w, h);

            if (sFbo == 0) {
                // create all post-processing resources.
                initPostProcessing(w, h);
            } else {
                // update texture and uniforms only.
                resizePostProcessing(w, h);
            }
        }
    }

    // don't draw until post-processing is ready.
    if (sFbo == 0 || gWidth == 0 || gHeight == 0) return;

    // Prepare rotation
    float target = sTargetAngle.load(std::memory_order_relaxed);
    sCurrentAngle += (target - sCurrentAngle) * kRotationLerpSpeed;

    // interpolate to scale
    sTargetScale   = sMagnified.load(std::memory_order_relaxed) ? kMagnifiedScale : 1.0f;
    sCurrentScale += (sTargetScale - sCurrentScale) * kScaleLerpSpeed;

    // ── Pass 1: render triangle
    glBindFramebuffer(GL_FRAMEBUFFER, sFbo);
    glViewport(0, 0, gWidth, gHeight);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(sProgram);
    glUniform1f(sUAngleLoc, sCurrentAngle);
    glUniform1f(sUScaleLoc, sCurrentScale);

    glBindVertexArray(sVao);
    glDrawArrays(GL_TRIANGLES, 0, kVertexCount);

    // ── Pass 2: apply blur
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, gWidth, gHeight);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(sBlurProgram);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, sColorTexture);
    glBindVertexArray(sQuadVao);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    eglSwapBuffers(sDisplay, sSurface);
}

/**
 * setViewport(int w, int h)
 * Updates the GL viewport to match the current surface dimensions.
 * Called whenever the surface is created or resized.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_setViewport(JNIEnv* /*env*/, jobject /*thiz*/, jint w, jint h) {
    // Store new Viewport Dimensions
    sPendingWidth.store(w,  std::memory_order_relaxed);
    sPendingHeight.store(h, std::memory_order_relaxed);
    sPendingResize.store(true, std::memory_order_release);
    LOGI("setViewport: queued resize to %d x %d", w, h);
}

/**
 * onTap()
 * Advances the target rotation by 90° each time the screen is tapped.
 * Safe to call from the UI thread; uses atomic store.
 */
extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_onSingleTap(JNIEnv* /*env*/, jobject /*thiz*/) {
    float next = sTargetAngle.load(std::memory_order_relaxed) + kTapRotationStep;
    sTargetAngle.store(next, std::memory_order_relaxed);
    LOGI("Tap! Target angle now: %.4f rad (%.1f deg)", next, next * (180.0f / M_PI));
}

extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_onDoubleTap(JNIEnv* /*env*/, jobject /*thiz*/) {
    bool next = !sMagnified.load(std::memory_order_relaxed);
    sMagnified.store(next, std::memory_order_relaxed);
    LOGI("Double tap! Magnification %s", next ? "ON" : "OFF");
}