package com.test.testgame

import android.os.Bundle
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import androidx.appcompat.app.AppCompatActivity
import android.os.Handler
import android.os.Looper

class MainActivity : AppCompatActivity() {

    private lateinit var surfaceView: SurfaceView
    private var renderThread: Thread? = null
    @Volatile private var glReady = false

    // single/double tap handling
    private val handler = Handler(Looper.getMainLooper())
    private val DOUBLE_TAP_TIME = 250L
    private var lastTapTime = 0L
    private var singleTapRunnable: Runnable? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        surfaceView = SurfaceView(this)
        setContentView(surfaceView)

        surfaceView.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                handleTap()
            }
            true
        }

        surfaceView.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                renderThread = Thread {
                    initGL(holder.surface)
                    glReady = true
                    while (!Thread.currentThread().isInterrupted) {
                        render()
                        try {
                            Thread.sleep(16)
                        } catch (e: InterruptedException) {
                            Thread.currentThread().interrupt()
                            break
                        }
                    }
                }
                renderThread?.start()
            }

            override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
                while (!glReady) Thread.sleep(1)
                setViewport(width, height)
            }

            override fun surfaceDestroyed(holder: SurfaceHolder) {
                glReady = false
                renderThread?.interrupt()
                renderThread?.join()
            }
        })
    }

    // Distinguishes between single and double taps
    private fun handleTap() {
        val currentTime = System.currentTimeMillis()

        if (currentTime - lastTapTime < DOUBLE_TAP_TIME) {
            // Double tap detected → cancel single tap
            if (singleTapRunnable != null) {
                handler.removeCallbacks(singleTapRunnable!!)
            }
            singleTapRunnable = null

            onDoubleTap()
            lastTapTime = 0L
        } else {
            lastTapTime = currentTime

            singleTapRunnable = Runnable {
                onSingleTap()
            }

            handler.postDelayed(singleTapRunnable!!, DOUBLE_TAP_TIME)
        }

    }

    external fun initGL(surface: android.view.Surface)
    external fun render()
    external fun setViewport(width: Int, height: Int)
    external fun onSingleTap()
    external fun onDoubleTap()

    companion object {
        init {
            System.loadLibrary("testgame")
        }
    }
}