package com.test.testgame

import android.os.Bundle
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var surfaceView: SurfaceView
    private var renderThread: Thread? = null
    @Volatile private var glReady = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        surfaceView = SurfaceView(this)
        setContentView(surfaceView)

        surfaceView.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                onTap()
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

    external fun initGL(surface: android.view.Surface)
    external fun render()
    external fun setViewport(width: Int, height: Int)
    external fun onTap()

    companion object {
        init {
            System.loadLibrary("testgame")
        }
    }
}