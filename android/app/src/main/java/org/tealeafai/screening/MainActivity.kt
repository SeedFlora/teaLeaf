package org.tealeafai.screening

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Surface
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tealeafai.screening.data.FieldLogger
import org.tealeafai.screening.ml.ActivationMap
import org.tealeafai.screening.ml.Preprocessing
import org.tealeafai.screening.ml.TeaLeafClassifier
import org.tealeafai.screening.ui.CameraScreen
import org.tealeafai.screening.ui.ResultScreen
import org.tealeafai.screening.ui.ScreeningResult
import org.tealeafai.screening.ui.TeaLeafTheme

/**
 * Single-activity flow: capture or import, read, explain, record a verdict.
 *
 * Model loading failure is surfaced as UI state rather than thrown. The load verifies the
 * asset's SHA-256 against the digest recorded at export, so a mismatched model produces a
 * visible message instead of quietly running weights whose accuracy nobody measured.
 */
class MainActivity : ComponentActivity() {

    private var classifier: TeaLeafClassifier? = null
    private var loadError: String? = null
    private lateinit var logger: FieldLogger

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        logger = FieldLogger(this)

        try {
            classifier = TeaLeafClassifier.create(this)
        } catch (t: Throwable) {
            loadError = t.message ?: t::class.java.simpleName
        }

        setContent {
            TeaLeafTheme {
                Surface(Modifier.fillMaxSize()) {
                    AppFlow(classifier, loadError, logger)
                }
            }
        }
    }

    override fun onDestroy() {
        classifier?.close()
        super.onDestroy()
    }
}

@Composable
private fun AppFlow(
    classifier: TeaLeafClassifier?,
    loadError: String?,
    logger: FieldLogger,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var result by remember { mutableStateOf<ScreeningResult?>(null) }
    var recordId by remember { mutableStateOf<String?>(null) }
    var busy by remember { mutableStateOf(false) }

    suspend fun analyze(bitmap: Bitmap, source: String, rotation: Int) {
        val c = classifier ?: return
        val prediction = withContext(Dispatchers.Default) { c.classify(bitmap, rotation) }

        // Build the explanation overlay from the model's second output when it has one.
        val upright = Preprocessing.rotateBitmap(bitmap, rotation)
        val display = Preprocessing.resizeForModel(upright)
        val cam = withContext(Dispatchers.Default) { c.lastActivationMap() }

        var heatmap: Bitmap? = null
        var peripheral = 0f
        if (cam != null) {
            heatmap = withContext(Dispatchers.Default) {
                ActivationMap.overlay(display, cam, 7, 7, c.config.classOrder.size, prediction.topIndex)
            }
            peripheral = ActivationMap.peripheralMass(
                ActivationMap.normalize(
                    ActivationMap.extractClassMap(cam, 7, 7, c.config.classOrder.size, prediction.topIndex)
                )
            )
        }

        recordId = withContext(Dispatchers.IO) {
            logger.record(
                photo = display,
                prediction = prediction,
                config = c.config,
                source = source,
                rotationApplied = rotation,
                peripheralMass = peripheral,
                consentGranted = true,
            )
        }

        result = ScreeningResult(
            photo = display,
            heatmap = heatmap,
            prediction = prediction,
            labelsEn = c.config.displayLabelsEn,
            labelsId = c.config.displayLabelsId,
            threshold = c.config.abstentionThreshold,
            modelVariant = c.config.variant,
            modelVersion = c.config.version,
            peripheralMass = peripheral,
        )
    }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        if (uri == null) return@rememberLauncherForActivityResult
        busy = true
        scope.launch {
            try {
                val bitmap = withContext(Dispatchers.IO) {
                    // Two streams: one to read the orientation tag, one to decode. The
                    // decoder ignores EXIF, so an untreated gallery import can arrive
                    // rotated with no error anywhere.
                    context.contentResolver.openInputStream(uri).use { exif ->
                        context.contentResolver.openInputStream(uri).use { img ->
                            Preprocessing.decodeAndUpright(img!!, exif!!)
                        }
                    }
                }
                analyze(bitmap, source = "gallery", rotation = 0)
            } finally {
                busy = false
            }
        }
    }

    val current = result
    if (current == null) {
        CameraScreen(
            modelReady = classifier != null,
            loadError = loadError,
            busy = busy,
            onCaptured = { bitmap, rotationDegrees ->
                busy = true
                scope.launch {
                    try {
                        analyze(bitmap, source = "camera", rotation = rotationDegrees)
                    } finally {
                        busy = false
                    }
                }
            },
            onPickImage = { picker.launch("image/*") },
        )
    } else {
        ResultScreen(
            result = current,
            onRetake = { result = null; recordId = null },
            onFeedback = { verdict, claimed ->
                val id = recordId
                if (id != null) {
                    scope.launch(Dispatchers.IO) {
                        logger.recordVerdict(id, verdict, claimed, expertise = "unstated")
                    }
                }
            },
        )
    }
}

