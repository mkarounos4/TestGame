/**
 * renderer.cpp
 *
 * Native OpenGL ES 3.0 renderer for Android using the NDK.
 * Renders a rotating triangle/image, driven by gestures from the Java layer.
 *
 * Architecture overview:
 *   - Java/Kotlin calls initGL(), render(), setViewport(), onTap(), onDoubleTap(), and onFling() via JNI.
 *   - initGL()     : Sets up EGL display/surface/context and compiles shaders.
 *   - render()     : Called every frame; smoothly interpolates rotation, scale, and translation.
 *   - setViewport(): Called on surface resize events.
 *   - onTap()      : Increments the target rotation angle by 90°.
 *   - onDoubleTap(): Toggles between normal and magnified scale.
 *   - onFling()    : Updates translation velocity for viewpoint movement.
 *
 * Threading note:
 *   targetAngle is an atomic<float> because onTap() may be called from the UI
 *   thread while render() runs on the GL/render thread.
 */

#include <jni.h>
#include <android/native_window_jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <cmath>
#include <atomic>
#include <algorithm>

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
// ---------------------------------------------------------------------------

static EGLDisplay sDisplay = EGL_NO_DISPLAY;
static EGLSurface sSurface = EGL_NO_SURFACE;
static EGLContext sContext = EGL_NO_CONTEXT;

// ---------------------------------------------------------------------------
// GL objects
// ---------------------------------------------------------------------------

static GLuint sProgram = 0;  // Linked shader program
static GLuint sVbo     = 0;  // Vertex Buffer Object
static GLuint sVao     = 0;  // Vertex Array Object
static GLuint sFbo = 0;
static GLuint sColorTexture = 0;
static GLuint sQuadVao = 0;
static GLuint sQuadVbo = 0;
static GLuint sBlurProgram = 0;
static GLuint sImageTexture = 0;

// Uniform locations
static GLint sUTransformLoc = -1;
static GLint sUTexLoc = -1;

// ---------------------------------------------------------------------------
// Matrix Helpers (Column-major for OpenGL)
// ---------------------------------------------------------------------------

static void matrixIdentity(float* m) {
    for (int i = 0; i < 16; i++) m[i] = 0;
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static void matrixMultiply(float* out, const float* a, const float* b) {
    float tmp[16];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            tmp[j * 4 + i] = 0;
            for (int k = 0; k < 4; k++) {
                tmp[j * 4 + i] += a[k * 4 + i] * b[j * 4 + k];
            }
        }
    }
    for (int i = 0; i < 16; i++) out[i] = tmp[i];
}

static void matrixTranslate(float* m, float tx, float ty, float tz) {
    float t[16]; matrixIdentity(t);
    t[12] = tx; t[13] = ty; t[14] = tz;
    matrixMultiply(m, m, t);
}

static void matrixScale(float* m, float sx, float sy, float sz) {
    float s[16]; matrixIdentity(s);
    s[0] = sx; s[5] = sy; s[10] = sz;
    matrixMultiply(m, m, s);
}

static void matrixRotateZ(float* m, float angle) {
    float r[16]; matrixIdentity(r);
    float c = cosf(angle); float s = sinf(angle);
    r[0] = c;  r[1] = s;
    r[4] = -s; r[5] = c;
    matrixMultiply(m, m, r);
}

// ---------------------------------------------------------------------------
// Animation State
// ---------------------------------------------------------------------------

// Rotation
static std::atomic<float> sTargetAngle{0.0f};
static float sCurrentAngle = 0.0f;
static constexpr float kRotationLerpSpeed = 0.1f;
static constexpr float kTapRotationStep = static_cast<float>(M_PI) / 2.0f;

// Scale
static std::atomic<bool> sMagnified{false};
static float sTargetScale  = 1.0f;
static float sCurrentScale = 1.0f;
static constexpr float kScaleLerpSpeed = 0.1f;
static std::atomic<float> sNormalScale{1.0f};
static std::atomic<float> sMagnifiedScale{2.0f};

// Translation (Fling)
static float sOffsetX = 0.0f;
static float sOffsetY = 0.0f;
static std::atomic<float> sVelX{0.0f};
static std::atomic<float> sVelY{0.0f};
static constexpr float kFriction = 0.96f; // Velocity decay per frame
static constexpr float kVelocityScale = 0.00001f; // Convert pixels/sec to GL units/frame

// ---------------------------------------------------------------------------
// Pending viewport resize
// ---------------------------------------------------------------------------
static std::atomic<int> sPendingWidth{0};
static std::atomic<int> sPendingHeight{0};
static std::atomic<bool> sPendingResize{false};

// ---------------------------------------------------------------------------
// Image update state
// ---------------------------------------------------------------------------
static std::atomic<bool> sNewImagePending{false};
static jobject sPendingBitmap = nullptr;

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
static void initPostProcessing(int width, int height);
static void resizePostProcessing(int width, int height);

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

static const GLfloat kVertices[] = {
        // X,     Y,     Z,     U,     V
        -0.5f, 0.5f, 0.0f, 0.0f, 0.0f, // Top-Left
        -0.5f, -0.5f, 0.0f, 0.0f, 1.0f, // Bottom-Left
        0.5f, -0.5f, 0.0f, 1.0f, 1.0f, // Bottom-Right

        -0.5f, 0.5f, 0.0f, 0.0f, 0.0f, // Top-Left
        0.5f, -0.5f, 0.0f, 1.0f, 1.0f, // Bottom-Right
        0.5f, 0.5f, 0.0f, 1.0f, 0.0f, // Top-Right
};
static constexpr GLsizei kVertexCount = 6;

// ---------------------------------------------------------------------------
// GLSL shaders
// ---------------------------------------------------------------------------

static const char* kVertexShaderSrc = R"GLSL(#version 300 es
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aTexCoord;

uniform mat4 uTransform;
out vec2 vTexCoord;

void main() {
    gl_Position = uTransform * vec4(aPos, 1.0);
    vTexCoord = aTexCoord;
}
)GLSL";

static const char* kFragmentShaderSrc = R"GLSL(#version 300 es
precision mediump float;
uniform sampler2D uTexture;
in vec2 vTexCoord;
out vec4 FragColor;

void main() {
    FragColor = texture(uTexture, vTexCoord);
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
        vec2(-uTexelSize.x,  uTexelSize.y), vec2( 0.0,  uTexelSize.y), vec2( uTexelSize.x,  uTexelSize.y),
        vec2(-uTexelSize.x,  0.0),          vec2( 0.0,  0.0),          vec2( uTexelSize.x,  0.0),
        vec2(-uTexelSize.x, -uTexelSize.y), vec2( 0.0, -uTexelSize.y), vec2( uTexelSize.x, -uTexelSize.y)
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
// Helpers
// ---------------------------------------------------------------------------

static GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint success = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[512];
        glGetShaderInfoLog(shader, sizeof(info), nullptr, info);
        LOGE("Shader compile error: %s", info);
    }
    return shader;
}

static GLuint buildProgram(const char* vertSrc, const char* fragSrc) {
    GLuint vert = compileShader(GL_VERTEX_SHADER, vertSrc);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragSrc);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);
    glDeleteShader(vert);
    glDeleteShader(frag);
    return prog;
}

static void initEGL(ANativeWindow* window) {
    sDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    eglInitialize(sDisplay, nullptr, nullptr);
    const EGLint configAttribs[] = { EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE, EGL_WINDOW_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_NONE };
    EGLConfig config; EGLint numConfigs = 0;
    eglChooseConfig(sDisplay, configAttribs, &config, 1, &numConfigs);
    sSurface = eglCreateWindowSurface(sDisplay, config, window, nullptr);
    const EGLint ctxAttribs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    sContext = eglCreateContext(sDisplay, config, EGL_NO_CONTEXT, ctxAttribs);
    eglMakeCurrent(sDisplay, sSurface, sSurface, sContext);
}

static void initGeometry() {
    glGenVertexArrays(1, &sVao);
    glBindVertexArray(sVao);
    glGenBuffers(1, &sVbo);
    glBindBuffer(GL_ARRAY_BUFFER, sVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kVertices), kVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void *)(3 * sizeof(GLfloat)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

static void initShaders() {
    sProgram = buildProgram(kVertexShaderSrc, kFragmentShaderSrc);
    sUTransformLoc = glGetUniformLocation(sProgram, "uTransform");
    sUTexLoc = glGetUniformLocation(sProgram, "uTexture");

    glGenTextures(1, &sImageTexture);
    glBindTexture(GL_TEXTURE_2D, sImageTexture);
    unsigned char pixels[] = {255, 0, 0 , 255};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

static void initPostProcessing(int width, int height) {
    glGenFramebuffers(1, &sFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, sFbo);
    glGenTextures(1, &sColorTexture);
    glBindTexture(GL_TEXTURE_2D, sColorTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sColorTexture, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    float quadVertices[] = { -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f };
    glGenVertexArrays(1, &sQuadVao);
    glGenBuffers(1, &sQuadVbo);
    glBindVertexArray(sQuadVao);
    glBindBuffer(GL_ARRAY_BUFFER, sQuadVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);

    sBlurProgram = buildProgram(kBlurVertexShaderSrc, kBlurFragmentShaderSrc);
    glUseProgram(sBlurProgram);
    glUniform1i(glGetUniformLocation(sBlurProgram, "screenTexture"), 0);
    glUniform2f(glGetUniformLocation(sBlurProgram, "uTexelSize"), 1.0f / (float)width, 1.0f / (float)height);
}

static void resizePostProcessing(int width, int height) {
    glBindTexture(GL_TEXTURE_2D, sColorTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glUseProgram(sBlurProgram);
    glUniform2f(glGetUniformLocation(sBlurProgram, "uTexelSize"), 1.0f / (float)width, 1.0f / (float)height);
}

static void uploadPendingImage(JNIEnv *env) {
    if (!sNewImagePending.load(std::memory_order_acquire)) return;
    AndroidBitmapInfo info; void *pixels = nullptr;
    if (AndroidBitmap_getInfo(env, sPendingBitmap, &info) < 0) return;
    if (AndroidBitmap_lockPixels(env, sPendingBitmap, &pixels) < 0) return;
    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        unsigned char* d = (unsigned char*)pixels;
        for (unsigned int i = 0; i < info.width * info.height * 4; i += 4) {
            if (d[i] > 200 && d[i + 1] < 100 && d[i + 2] < 100) d[i + 2] = 255;
        }
    }
    glBindTexture(GL_TEXTURE_2D, sImageTexture);
    GLint f = (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) ? GL_RGBA : GL_RGB;
    glTexImage2D(GL_TEXTURE_2D, 0, f, (GLsizei)info.width, (GLsizei)info.height, 0, (GLenum)f, GL_UNSIGNED_BYTE, pixels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    AndroidBitmap_unlockPixels(env, sPendingBitmap);
    env->DeleteGlobalRef(sPendingBitmap); sPendingBitmap = nullptr;
    sNewImagePending.store(false, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// JNI entry points
// ---------------------------------------------------------------------------

extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_initGL(JNIEnv* env, jobject /*thiz*/, jobject surfaceObj) {
    sProgram = 0; sVbo = 0; sVao = 0; sFbo = 0; sColorTexture = 0;
    sQuadVao = 0; sQuadVbo = 0; sBlurProgram = 0; sImageTexture = 0;

    ANativeWindow* window = ANativeWindow_fromSurface(env, surfaceObj);
    initEGL(window);
    initGeometry();
    initShaders();
    ANativeWindow_release(window);
    LOGI("GL initialized in new context.");
}

extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_terminateGL(JNIEnv* /*env*/, jobject /*thiz*/) {
    if (sDisplay != EGL_NO_DISPLAY) {
        eglMakeCurrent(sDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (sContext != EGL_NO_CONTEXT) eglDestroyContext(sDisplay, sContext);
        if (sSurface != EGL_NO_SURFACE) eglDestroySurface(sDisplay, sSurface);
        eglTerminate(sDisplay);
    }
    sDisplay = EGL_NO_DISPLAY; sSurface = EGL_NO_SURFACE; sContext = EGL_NO_CONTEXT;
    gWidth = 0; gHeight = 0;
    LOGI("GL terminated.");
}

extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_render(JNIEnv* env, jobject) {
    if (sPendingResize.load(std::memory_order_acquire)) {
        sPendingResize.store(false, std::memory_order_relaxed);
        int w = sPendingWidth.load(std::memory_order_relaxed);
        int h = sPendingHeight.load(std::memory_order_relaxed);
        if (w > 0 && h > 0) {
            gWidth  = w; gHeight = h;
            glViewport(0, 0, w, h);
            if (sFbo == 0) initPostProcessing(w, h);
            else resizePostProcessing(w, h);
        }
    }
    if (sFbo == 0 || gWidth == 0 || gHeight == 0) return;

    uploadPendingImage(env);

    // Update States
    float target = sTargetAngle.load(std::memory_order_relaxed);
    sCurrentAngle += (target - sCurrentAngle) * kRotationLerpSpeed;

    float bS = sNormalScale.load(std::memory_order_relaxed);
    float mS = sMagnifiedScale.load(std::memory_order_relaxed);
    sTargetScale   = sMagnified.load(std::memory_order_relaxed) ? mS : bS;
    sCurrentScale += (sTargetScale - sCurrentScale) * kScaleLerpSpeed;

    // Fling Physics
    float vx = sVelX.load(std::memory_order_relaxed);
    float vy = sVelY.load(std::memory_order_relaxed);
    float aspectX = ((gHeight>gWidth) ? gHeight/gWidth : 1);
    float aspectY = ((gWidth>gHeight) ? gWidth/gHeight : 1);;
    // scale scroll speed back to 1 to 1
    sOffsetX += vx * kVelocityScale * aspectX;
    sOffsetY -= vy * kVelocityScale * aspectY;
    sVelX.store(vx * kFriction, std::memory_order_relaxed);
    sVelY.store(vy * kFriction, std::memory_order_relaxed);
    sOffsetX = std::max(-2.0f, std::min(2.0f, sOffsetX));
    sOffsetY = std::max(-2.0f, std::min(2.0f, sOffsetY));

    // ─── CALCULATE TRANSFORMATION MATRIX ───
    float transform[16];
    matrixIdentity(transform);
    // Order: Translation * Scale * Rotation
    matrixTranslate(transform, sOffsetX, sOffsetY, 0.0f);
    matrixScale(transform, sCurrentScale, sCurrentScale, 1.0f);
    matrixRotateZ(transform, sCurrentAngle);

    // Pass 1: Render Image
    glBindFramebuffer(GL_FRAMEBUFFER, sFbo);
    glViewport(0, 0, gWidth, gHeight);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f); glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(sProgram);
    glUniformMatrix4fv(sUTransformLoc, 1, GL_FALSE, transform);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, sImageTexture);
    glUniform1i(sUTexLoc, 0);

    glBindVertexArray(sVao); glDrawArrays(GL_TRIANGLES, 0, kVertexCount);

    // Pass 2: Blur
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, gWidth, gHeight);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(sBlurProgram);
    glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, sColorTexture);
    glBindVertexArray(sQuadVao); glDrawArrays(GL_TRIANGLES, 0, 6);

    eglSwapBuffers(sDisplay, sSurface);
}

extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_setViewport(JNIEnv* /*env*/, jobject /*thiz*/, jint w, jint h) {
    sPendingWidth.store(w, std::memory_order_relaxed);
    sPendingHeight.store(h, std::memory_order_relaxed);
    sPendingResize.store(true, std::memory_order_release);
}

extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_onSingleTap(JNIEnv* /*env*/, jobject /*thiz*/) {
    float n = sTargetAngle.load(std::memory_order_relaxed) + kTapRotationStep;
    sTargetAngle.store(n, std::memory_order_relaxed);
}

extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_onDoubleTap(JNIEnv* /*env*/, jobject /*thiz*/) {
    sMagnified.store(!sMagnified.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_onFling(JNIEnv* /*env*/, jobject /*thiz*/, jfloat vx, jfloat vy) {
    sVelX.store(vx, std::memory_order_relaxed);
    sVelY.store(vy, std::memory_order_relaxed);
}

extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_setImage(JNIEnv* env, jobject /*thiz*/, jobject bitmap) {
    if (sPendingBitmap) env->DeleteGlobalRef(sPendingBitmap);
    sPendingBitmap = env->NewGlobalRef(bitmap);
    sNewImagePending.store(true, std::memory_order_release);
}

extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_setZoomLevels(JNIEnv* /*env*/, jobject /*thiz*/, jfloat normal, jfloat magnified) {
    sNormalScale.store(normal, std::memory_order_relaxed);
    sMagnifiedScale.store(magnified, std::memory_order_relaxed);
}
