package com.test.testgame

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.GestureDetector
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.FrameLayout
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.ActionBarDrawerToggle
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.Toolbar
import androidx.drawerlayout.widget.DrawerLayout
import com.google.android.material.navigation.NavigationView
import com.google.android.material.slider.RangeSlider

class MainActivity : AppCompatActivity() {

    private lateinit var surfaceView: SurfaceView
    private var renderThread: Thread? = null
    @Volatile private var glReady = false
    private var filterFiles: Array<String> = emptyArray()

    private val pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) {
            loadBitmapFromUri(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val toolbar: Toolbar = findViewById(R.id.toolbar)
        setSupportActionBar(toolbar)

        val drawerLayout: DrawerLayout = findViewById(R.id.drawer_layout)
        val toggle = ActionBarDrawerToggle(
            this, drawerLayout, toolbar,
            R.string.navigation_drawer_open, R.string.navigation_drawer_close
        )
        drawerLayout.addDrawerListener(toggle)
        toggle.syncState()

        val navView: NavigationView = findViewById(R.id.nav_view)
        if (navView.headerCount > 0) {
            setupDrawerControls(navView.getHeaderView(0))
        }

        surfaceView = SurfaceView(this)
        findViewById<FrameLayout>(R.id.container).addView(surfaceView)

        val gestureDetector = GestureDetector(this, object : GestureDetector.SimpleOnGestureListener() {
            override fun onDown(e: MotionEvent): Boolean = true
            override fun onSingleTapConfirmed(e: MotionEvent): Boolean {
                onSingleTap()
                return true
            }
            override fun onDoubleTap(e: MotionEvent): Boolean {
                onDoubleTap()
                return true
            }
            override fun onLongPress(e: MotionEvent) {
                pickMedia.launch(androidx.activity.result.PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            }
            override fun onFling(e1: MotionEvent?, e2: MotionEvent, velocityX: Float, velocityY: Float): Boolean {
                onFling(velocityX, velocityY)
                return true
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
                    
                    lastBitmap?.let { setImage(it) }
                    setZoomLevels(savedNormalZoom, savedMagZoom)
                    
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
                terminateGL()
            }
        })
    }

    private fun applyFilter(fileName: String) {
        try {
            val content = assets.open("filters/$fileName").bufferedReader().use { it.readText() }
            val matrix = content.split(",")
                .map { it.trim().toFloat() }
                .toFloatArray()
            if (matrix.size == 16) {
                setMagnificationMatrix(matrix)
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun setupDrawerControls(header: View) {
        val rangeSlider: RangeSlider? = header.findViewById(R.id.zoom_range_slider)
        val valueText: TextView? = header.findViewById(R.id.zoom_values_text)
        val filterSpinner: Spinner? = header.findViewById(R.id.filter_spinner)

        if (rangeSlider != null && valueText != null) {
            rangeSlider.values = listOf(savedNormalZoom, savedMagZoom)
            valueText.text = String.format("Range: %.1fx - %.1fx", savedNormalZoom, savedMagZoom)
            
            rangeSlider.addOnChangeListener { slider, _, _ ->
                val currentValues = slider.values
                if (currentValues.size >= 2) {
                    val normal = currentValues[0]
                    val magnified = currentValues[1]
                    valueText.text = String.format("Range: %.1fx - %.1fx", normal, magnified)
                    
                    setZoomLevels(normal, magnified)
                    savedNormalZoom = normal
                    savedMagZoom = magnified
                }
            }
        }

        filterSpinner?.let { spinner ->
            filterFiles = assets.list("filters") ?: emptyArray()
            val displayNames = filterFiles.map { it.substringBeforeLast(".") }
            val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, displayNames)
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            spinner.adapter = adapter

            spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                    applyFilter(filterFiles[position])
                }
                override fun onNothingSelected(parent: AdapterView<*>?) {}
            }
        }
    }

    private fun loadBitmapFromUri(uri: Uri) {
        contentResolver.openInputStream(uri)?.use { inputStream ->
            val bitmap = BitmapFactory.decodeStream(inputStream)
            if (bitmap != null) {
                lastBitmap = bitmap
                setImage(bitmap)
            }
        }
    }

    external fun initGL(surface: android.view.Surface)
    external fun terminateGL()
    external fun render()
    external fun setViewport(width: Int, height: Int)
    external fun onSingleTap()
    external fun onDoubleTap()
    external fun onFling(velocityX: Float, velocityY: Float)
    external fun setImage(bitmap: Bitmap)
    external fun setZoomLevels(normal: Float, magnified: Float)
    external fun setMagnificationMatrix(matrix: FloatArray)

    companion object {
        private var lastBitmap: Bitmap? = null
        private var savedNormalZoom: Float = 1.0f
        private var savedMagZoom: Float = 2.0f

        init {
            System.loadLibrary("testgame")
        }
    }
}
