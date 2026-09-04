package org.tealeafai.screening.ui

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors

/**
 * Live capture screen.
 *
 * The rotation handling here is the part that silently breaks models. CameraX's
 * `setTargetRotation` only writes metadata, and `ImageProxy.toBitmap()` decodes through
 * BitmapFactory, which ignores EXIF. A capture therefore arrives unrotated with no error
 * anywhere, and the model quietly loses accuracy on any photograph not taken in the
 * sensor's native orientation. The clockwise rotation reported by
 * `imageInfo.rotationDegrees` is applied explicitly before anything else touches the
 * bitmap, and the instrumented tests assert that direction on real hardware with a
 * negative control.
 */
@Composable
fun CameraScreen(
    modelReady: Boolean,
    loadError: String?,
    busy: Boolean,
    onCaptured: (Bitmap, Int) -> Unit,
    onPickImage: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    var permissionDenied by remember { mutableStateOf(false) }

    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasPermission = granted
        permissionDenied = !granted
    }

    LaunchedEffect(Unit) {
        if (!hasPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    val executor = remember { Executors.newSingleThreadExecutor() }
    DisposableEffect(Unit) { onDispose { executor.shutdown() } }

    var imageCapture by remember { mutableStateOf<ImageCapture?>(null) }
    var cameraError by remember { mutableStateOf<String?>(null) }

    Box(
        modifier
            .fillMaxSize()
            .background(TeaColors.Base)
    ) {
        // --- viewfinder ---------------------------------------------------------------
        Box(
            Modifier
                .fillMaxSize()
                .padding(horizontal = 14.dp)
                .padding(top = 96.dp, bottom = 156.dp)
                .clip(RoundedCornerShape(TeaShape.card))
                .background(TeaColors.Surface)
                .border(1.dp, TeaColors.Outline, RoundedCornerShape(TeaShape.card)),
            contentAlignment = Alignment.Center,
        ) {
            when {
                loadError != null -> CameraMessage(
                    "Model did not load", loadError, TeaColors.Danger
                )

                permissionDenied && !hasPermission -> Column(
                    Modifier.padding(30.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.spacedBy(14.dp),
                ) {
                    Text(
                        "Camera access is off",
                        style = MaterialTheme.typography.titleLarge,
                        color = TeaColors.Amber,
                    )
                    Text(
                        "Grant camera access to scan a leaf, or import a photo you already " +
                            "have. Images stay on this phone either way.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = TeaColors.TextMid,
                    )
                    Box(
                        Modifier
                            .clip(RoundedCornerShape(TeaShape.pill))
                            .background(TeaColors.SurfaceHigh)
                            .clickable { permissionLauncher.launch(Manifest.permission.CAMERA) }
                            .padding(horizontal = 22.dp, vertical = 13.dp)
                    ) {
                        Text(
                            "Allow camera",
                            style = MaterialTheme.typography.titleMedium,
                            color = TeaColors.Lime,
                        )
                    }
                }

                hasPermission -> {
                    AndroidView(
                        modifier = Modifier.fillMaxSize(),
                        factory = { ctx ->
                            PreviewView(ctx).apply {
                                scaleType = PreviewView.ScaleType.FILL_CENTER
                                implementationMode = PreviewView.ImplementationMode.COMPATIBLE
                            }
                        },
                        update = { view ->
                            bindCamera(view, context, lifecycleOwner) { capture, error ->
                                imageCapture = capture
                                cameraError = error
                            }
                        },
                    )
                    ScanFrame(active = !busy)
                }

                else -> CameraMessage("Waiting for camera access", null, TeaColors.TextLow)
            }

            cameraError?.let {
                Text(
                    it,
                    style = MaterialTheme.typography.bodySmall,
                    color = TeaColors.Danger,
                    modifier = Modifier.align(Alignment.BottomCenter).padding(16.dp),
                )
            }
        }

        // --- header -------------------------------------------------------------------
        Column(Modifier.align(Alignment.TopStart).padding(horizontal = 20.dp, vertical = 22.dp)) {
            Text("TeaLeaf AI", style = MaterialTheme.typography.displayLarge, color = TeaColors.TextHigh)
            Text(
                "One leaf, filling the frame, in even light.",
                style = MaterialTheme.typography.bodyMedium,
                color = TeaColors.TextMid,
            )
        }

        // --- controls -----------------------------------------------------------------
        Row(
            Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
                .padding(horizontal = 18.dp)
                .padding(bottom = 34.dp),
            horizontalArrangement = Arrangement.spacedBy(14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Box(
                Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(TeaShape.pill))
                    .background(TeaColors.SurfaceHigh)
                    .clickable(enabled = modelReady && !busy) { onPickImage() }
                    .padding(vertical = 18.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    "From gallery",
                    style = MaterialTheme.typography.titleMedium,
                    color = if (modelReady && !busy) TeaColors.TextHigh else TeaColors.TextLow,
                )
            }

            val canShoot = modelReady && !busy && hasPermission && imageCapture != null
            Box(
                Modifier
                    .size(80.dp)
                    .clip(CircleShape)
                    .background(if (canShoot) TeaColors.Lime else TeaColors.LimeDim)
                    .clickable(enabled = canShoot) {
                        takePhoto(imageCapture, executor, onCaptured) { cameraError = it }
                    },
                contentAlignment = Alignment.Center,
            ) {
                if (busy) {
                    Text(
                        "…",
                        style = MaterialTheme.typography.displayLarge,
                        color = TeaColors.Base,
                        fontWeight = FontWeight.Black,
                    )
                } else {
                    Box(Modifier.size(28.dp).clip(CircleShape).background(TeaColors.Base))
                }
            }
        }
    }
}

@Composable
private fun CameraMessage(title: String, detail: String?, accent: androidx.compose.ui.graphics.Color) {
    Column(
        Modifier.padding(30.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text(title, style = MaterialTheme.typography.titleLarge, color = accent)
        detail?.let {
            Text(it, style = MaterialTheme.typography.bodySmall, color = TeaColors.TextMid)
        }
    }
}

/**
 * Corner brackets with a slow sweep.
 *
 * The sweep is the only motion in the app and it exists to signal that the camera is live
 * rather than frozen. It stops while a reading is in flight, so the interface is never
 * animating during the one moment the user is waiting on a result.
 */
@Composable
private fun ScanFrame(active: Boolean) {
    val transition = rememberInfiniteTransition(label = "scan")
    val sweep by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(2600, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "sweep",
    )

    Box(Modifier.fillMaxSize()) {
        if (active) {
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(120.dp)
                    .align(Alignment.TopCenter)
                    .offset(y = (sweep * 420).dp)
                    .alpha(0.16f)
                    .background(
                        Brush.verticalGradient(
                            listOf(
                                TeaColors.Base.copy(alpha = 0f),
                                TeaColors.Lime,
                                TeaColors.Base.copy(alpha = 0f),
                            )
                        )
                    )
            )
        }
        CornerBrackets(
            Modifier.fillMaxSize().padding(18.dp),
            if (active) TeaColors.Lime else TeaColors.TextLow,
        )
    }
}

private fun bindCamera(
    view: PreviewView,
    context: Context,
    lifecycleOwner: androidx.lifecycle.LifecycleOwner,
    onReady: (ImageCapture?, String?) -> Unit,
) {
    val future = ProcessCameraProvider.getInstance(context)
    future.addListener({
        try {
            val provider = future.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(view.surfaceProvider)
            }
            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            provider.unbindAll()
            provider.bindToLifecycle(
                lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture
            )
            onReady(capture, null)
        } catch (t: Throwable) {
            onReady(null, "Camera unavailable: ${t.message ?: t::class.java.simpleName}")
        }
    }, ContextCompat.getMainExecutor(context))
}

private fun takePhoto(
    capture: ImageCapture?,
    executor: java.util.concurrent.Executor,
    onCaptured: (Bitmap, Int) -> Unit,
    onError: (String) -> Unit,
) {
    val ic = capture ?: return onError("Camera is not ready yet")
    ic.takePicture(executor, object : ImageCapture.OnImageCapturedCallback() {
        override fun onCaptureSuccess(image: ImageProxy) {
            try {
                // rotationDegrees is the CLOCKWISE rotation needed to upright the buffer.
                // toBitmap() does not apply it, so it is passed on and applied explicitly
                // downstream rather than assumed to be zero.
                val degrees = image.imageInfo.rotationDegrees
                val bitmap = image.toBitmap()
                onCaptured(bitmap, degrees)
            } catch (t: Throwable) {
                onError("Could not read the capture: ${t.message}")
            } finally {
                image.close()
            }
        }

        override fun onError(exception: ImageCaptureException) {
            onError("Capture failed: ${exception.message ?: exception.imageCaptureError.toString()}")
        }
    })
}
