#include <jni.h>
#include <android/native_window_jni.h>
#include <android/log.h>
#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <cmath>
#include <atomic>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "GAME", __VA_ARGS__)

static EGLDisplay display = EGL_NO_DISPLAY;
static EGLSurface eglSurface = EGL_NO_SURFACE;
static EGLContext context = EGL_NO_CONTEXT;
static GLuint program = 0;
static GLuint vbo = 0;
static GLuint vao = 0;

// Rotation state
static std::atomic<float> targetAngle{0.0f};
static float currentAngle = 0.0f;
static GLint uAngleLoc = -1;

static const GLfloat vertices[] = {
        0.0f,  0.5f, 0.0f,
        -0.5f, -0.5f, 0.0f,
        0.5f, -0.5f, 0.0f
};

GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char info[512];
        glGetShaderInfoLog(shader, 512, nullptr, info);
        LOGI("Shader compile error: %s", info);
    }
    return shader;
}

void initGLInternal(ANativeWindow* window) {
    display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    eglInitialize(display, nullptr, nullptr);

    EGLConfig config;
    EGLint numConfigs;
    const EGLint attribs[] = {
            EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
            EGL_SURFACE_TYPE,    EGL_WINDOW_BIT,
            EGL_BLUE_SIZE,  8,
            EGL_GREEN_SIZE, 8,
            EGL_RED_SIZE,   8,
            EGL_NONE
    };
    eglChooseConfig(display, attribs, &config, 1, &numConfigs);
    eglSurface = eglCreateWindowSurface(display, config, window, nullptr);

    const EGLint ctxAttribs[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    context = eglCreateContext(display, config, nullptr, ctxAttribs);
    eglMakeCurrent(display, eglSurface, eglSurface, context);

    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // Vertex shader now accepts a rotation angle uniform
    const char* vs =
            "#version 300 es\n"
            "layout(location = 0) in vec3 aPos;\n"
            "uniform float uAngle;\n"
            "void main() {\n"
            "    float c = cos(uAngle);\n"
            "    float s = sin(uAngle);\n"
            "    vec2 rotated = vec2(\n"
            "        aPos.x * c - aPos.y * s,\n"
            "        aPos.x * s + aPos.y * c\n"
            "    );\n"
            "    gl_Position = vec4(rotated, 0.0, 1.0);\n"
            "}\n";

    const char* fs =
            "#version 300 es\n"
            "precision mediump float;\n"
            "out vec4 FragColor;\n"
            "void main() { FragColor = vec4(1.0, 0.0, 0.0, 1.0); }\n";

    GLuint vert = compileShader(GL_VERTEX_SHADER, vs);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fs);
    program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);

    GLint linked;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        char info[512];
        glGetProgramInfoLog(program, 512, nullptr, info);
        LOGI("Program link error: %s", info);
    }

    glDeleteShader(vert);
    glDeleteShader(frag);

    uAngleLoc = glGetUniformLocation(program, "uAngle");
    LOGI("OpenGL initialized successfully");
}

extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_initGL(JNIEnv* env, jobject, jobject surfaceObj) {
    ANativeWindow* window = ANativeWindow_fromSurface(env, surfaceObj);
    initGLInternal(window);
    ANativeWindow_release(window);
}

extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_render(JNIEnv* env, jobject) {
    // Smoothly interpolate current angle toward target
    float target = targetAngle.load();
    currentAngle += (target - currentAngle) * 0.1f;

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(program);
    glUniform1f(uAngleLoc, currentAngle);

    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    eglSwapBuffers(display, eglSurface);
}

extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_setViewport(JNIEnv* env, jobject, jint w, jint h) {
    glViewport(0, 0, w, h);
}

// Each tap adds 90 degrees (in radians) to the target angle
extern "C" JNIEXPORT void JNICALL
Java_com_test_testgame_MainActivity_onTap(JNIEnv* env, jobject) {
    float current = targetAngle.load();
    targetAngle.store(current + (M_PI / 2.0f));
    LOGI("Tap! Target angle now: %f", targetAngle.load());
}