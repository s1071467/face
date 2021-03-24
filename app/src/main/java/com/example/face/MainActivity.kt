package com.example.face

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.service.autofill.Dataset
import android.widget.TextView
import com.example.face.ml.Model1
import org.tensorflow.lite.support.image.TensorImage

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val bitmap: Bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.smile1)

        val model = Model1.newInstance(this)

        // Creates inputs for reference.
        val image = TensorImage.fromBitmap(bitmap)

        // Runs model inference and gets result.
        val outputs = model.process(image)
        val probability = outputs.probabilityAsCategoryList

        // Releases model resources if no longer used.
        model.close()

        val txv: TextView = findViewById(R.id.txv)
        txv.text = probability.toString()

    }
}