package com.test.testgame

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var surfaceView: SurfaceView
    private var renderThread: Thread? = null
    @Volatile private var glReady = false

    // Image picker
    private val pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) {
            loadBitmapFromUri(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        surfaceView = SurfaceView(this)
        setContentView(surfaceView)

        // GestureDetector handles the complexity of distinguishing between single tap, double tap, and long press.
        val gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(e: MotionEvent): Boolean = true

            // Triggered only after it's confirmed not to be a double tap or long press
            override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
                onSingleTap()
                return true
            }

            override fun onDoubleTap(e: MotionEvent): Boolean {
                onDoubleTap()
                return true
            }

            override fun onLongPress(e: MotionEvent) {
                pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            }
        })

        surfaceView.setOnTouchListener { v, event ->
            val handled = gestureDetector.onTouchEvent(event)
            if (event.action == MotionEvent.ACTION_UP) {
                v.performClick()
            }
            handled || true
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

    private fun loadBitmapFromUri(uri: Uri) {
        contentResolver.openInputStream(uri)?.use { inputStream ->
            val bitmap = BitmapFactory.decodeStream(inputStream)
            if (bitmap != null) {
                setImage(bitmap)
            }
        }
    }

    external fun initGL(surface: android.view.Surface)
    external fun render()
    external fun setViewport(width: Int, height: Int)
    external fun onSingleTap()
    external fun onDoubleTap()
    external fun setImage(bitmap: Bitmap)

    companion object {
        init {
            System.loadLibrary("testgame")
        }
    }
}
