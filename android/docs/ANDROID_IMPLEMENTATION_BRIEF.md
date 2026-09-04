# IMPLEMENTATION BRIEF — Offline Tea-Leaf Health Classifier as Field Research Instrument

Target: Samsung SM-S916B (Galaxy S23+), Android 16 / API 36, arm64-v8a, Snapdragon 8 Gen 2.
Build host: Windows 11, JDK 21 Temurin, Gradle wrapper only, no Android Studio, no cmdline-tools.
Model: Keras-exported `.tflite`, input `[1,224,224,3]` float32 **raw [0,255]** (normalization is in-graph), output `[1,7]` softmax.
Date of record: 2026-07-20. Every version below was verified against a primary source on or before this date.

---

## 1. FINAL DEPENDENCY SET

### 1.1 Toolchain — pinned, and the reason is offline-capability

| Component | Version | Why this exact value |
|---|---|---|
| AGP | **8.13.0** | Min+default Gradle 8.13; JDK min 17; build-tools default 35.0.0 (installed); max API 36.1. Source: developer.android.com/build/releases/agp-8-13-0-release-notes |
| Gradle | **8.14** | **Decisive reason: `gradle-8.14-all` is the only distribution in the wrapper cache on this host.** AGP 9.0 needs Gradle ≥9.1.0, 9.2 needs 9.4.1, 9.3 needs 9.5.0 — every one of those forces a ~150 MB download. |
| JDK | **21.0.11 Temurin** (`C:\Users\<username>\tools\jdk-21.0.11+10`) | No Android doc states an AGP JDK *maximum*; JDK 17 is the documented minimum. JDK 21 support is established by Gradle's own matrix (JDK 21 supported for running Gradle since 8.5). **This is inference from Gradle's docs, not an AGP guarantee** — but it was empirically confirmed by a successful `assembleDebug` on this machine. |
| Kotlin (KGP) | **2.2.10** | Matches the version AGP 9.x would demand later; nothing in 8.13 constrains it downward. |
| compileSdk / targetSdk | **36** | `platforms/android-36` is installed. AGP 8.13 max is 36.1. |
| minSdk | **24** | Hard floors: LiteRT 2.1.6 AAR manifest declares `<uses-sdk android:minSdkVersion="23"/>`; CameraX 1.6.x floor is 23 (raised from 21 in 1.5.0-rc01). 24 clears both. |
| Build tools | leave default (35.0.0) | 36.0.0 is **not** installed. Do not upgrade AGP without pinning `buildToolsVersion = "36.1.0"`. |

**Do not use AGP 9.x on this host.** Current AGP is 9.3.0 (July 2026) — noted so nobody thinks 8.13 is the frontier — but 9.x is unreachable offline here. Correction to earlier reasoning circulating on this project: "AGP 8.13 max API is 36.1" is *not* a differentiator (AGP 9.0's max is also 36.1), and the build-tools-36.0.0 argument is weak (defeated by one `buildToolsVersion` line). The Gradle-distribution-cache argument is the real one.

### 1.2 `settings.gradle.kts`

```kotlin
pluginManagement {
    repositories { google(); mavenCentral(); gradlePluginPortal() }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()        // MANDATORY — LiteRT is NOT on Maven Central
        mavenCentral()
    }
}
rootProject.name = "tealeaf"
include(":app")
```

`google()` is non-negotiable. Verified: `https://repo1.maven.org/maven2/com/google/ai/edge/litert/litert/maven-metadata.xml` → **HTTP 404**; Maven Central solrsearch for `g:com.google.ai.edge.litert` → `numFound: 0`. `https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/maven-metadata.xml` → `<release>2.1.6</release>`, `lastUpdated 20260701221338`.

**The official page `developers.google.com/edge/litert/android/development` is wrong here** — it says the AAR is "hosted at MavenCentral" and shows `implementation 'com.google.ai.edge.litert:+'`. Both statements are false/dangerous. Ignore that page.

### 1.3 Root `build.gradle.kts`

```kotlin
plugins {
    id("com.android.application") version "8.13.0" apply false
    id("org.jetbrains.kotlin.android") version "2.2.10" apply false
    // id("org.jetbrains.kotlin.plugin.compose") version "2.2.10" apply false  // only if Compose
}
```

### 1.4 `app/build.gradle.kts` — dependencies

```kotlin
dependencies {
    // ---- Runtime: LiteRT V2 ----
    implementation("com.google.ai.edge.litert:litert:2.1.6")
    // pulls com.google.ai.edge.litert:litert-api:2.1.6 transitively (verified: the POM has
    // exactly one compile dependency, and it is litert-api:2.1.6)

    // ---- Camera ----
    implementation("androidx.camera:camera-core:1.6.1")
    implementation("androidx.camera:camera-camera2:1.6.1")
    implementation("androidx.camera:camera-lifecycle:1.6.1")
    implementation("androidx.camera:camera-view:1.6.1")     // PreviewView

    // ---- Persistence for the field log ----
    implementation("androidx.core:core-ktx:1.13.1")          // NOT re-verified this session

    // ---- Unit tests ----
    testImplementation("junit:junit:4.13.2")

    // ---- Instrumented tests ----
    androidTestImplementation("androidx.test:core:1.7.0")
    androidTestImplementation("androidx.test:runner:1.7.0")
    androidTestImplementation("androidx.test:rules:1.7.0")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test.ext:truth:1.7.0")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.7.0")

    // ---- Benchmark (separate com.android.test module, see §5) ----
    // androidTestImplementation("androidx.benchmark:benchmark-junit4:1.4.1")
    // implementation("androidx.tracing:tracing-ktx:1.3.0")
}
```

Version provenance:
- CameraX **1.6.1** — latest stable, released 2026-05-06 (1.6.0 was 2026-03-25; 1.7.0-alpha02 2026-07-01 is preview). All seven camera artifacts share one version.
- `androidx.exifinterface:exifinterface:1.4.2` (2025-12-03) — **not needed** by the recommended in-memory capture path. Add only if you switch to file-based capture.
- androidx.test coordinates verified against Google Maven `group-index.xml`, not just release notes. Note two release-note errors this corrects: the artifact is `androidx.test.services:test-services` (there is no `:services`), and `androidx.test:monitor` latest is 1.9.0 (release notes said 1.8.0). Espresso 3.7.0 came from release notes only and was **not** cross-checked against group-index — the one soft version in this table.
- Benchmark **1.4.1** stable (2025-09-10); alpha 1.5.0-alpha07 exists — use stable.
- `androidx.tracing:tracing` **1.3.0** stable (2025-04-23).

### 1.5 Compose — FLAGGED, NOT VERIFIED

**Compose versions were not researched or fact-checked in this cycle. Do not treat any Compose BOM number as verified.**

Recommendation: **build the UI in Views**, not Compose. Concrete reasons, not stylistic:
1. `androidx.camera:camera-view` `PreviewView` is a View and is the fully-documented preview surface; `camera-compose:1.6.1` was listed in release notes but its existence on Maven was never checked.
2. This app's UI is a preview surface, a shutter button, a 7-row result list and a verdict form. Compose buys nothing here and adds an unverified compiler-plugin/BOM matrix to an offline build.
3. Every fact in this brief is anchored to a fetched primary source. Introducing an unpinned Compose BOM would be the one unanchored dependency.

If Compose is mandated anyway: add `org.jetbrains.kotlin.plugin.compose` at the same version as KGP (2.2.10), `buildFeatures { compose = true }`, and resolve `androidx.compose:compose-bom` — **and verify that BOM version against `dl.google.com/dl/android/maven2/androidx/compose/compose-bom/maven-metadata.xml` before committing.** Keep `PreviewView` inside an `AndroidView { }` rather than adopting `camera-compose`.

### 1.6 `app/build.gradle.kts` — android block

```kotlin
android {
    namespace = "com.example.tealeaf"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.tealeaf"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        debug { /* AGP injects android:debuggable=true */ }
        release {
            isMinifyEnabled = false          // keep true accuracy/latency parity for the study
            isDebuggable = false             // REQUIRED for any published latency number
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    // NO androidResources { noCompress } BLOCK. See below.
}
```

**On `noCompress`: do not add it.** Verified by disassembling AGP itself, not by reading docs. `com.android.builder.packaging.PackagingUtils.getAllNoCompressExtensions(...)` executes, unconditionally:

```
builder.addAll(DEFAULT_AAPT_NO_COMPRESS_EXTENSIONS);   // .jpg .png .gif .opus .mp4 ...
builder.add(".tflite");                                 // hardcoded, no flag, no opt-in
```

Byte-identical in **AGP 8.13.0 and AGP 9.3.0** (jars pulled from `dl.google.com/dl/android/maven2/com/android/tools/build/builder/{8.13.0,9.3.0}/`). That method feeds all three packaging paths — `getNoCompressPredicate` (APK assets, via `PackageAndroidArtifact` and `CompressAssetsTask`), `getNoCompressPredicateForJavaRes`, and `getNoCompressGlobsForBundle` (the glob list handed to bundletool) — so **AAB delivery is covered too**.

The critical caveat nobody states: matching is `StringsKt.endsWith(path, ext, ignoreCase = true)` — **suffix only**. `model.tflite` and `MODEL.TFLITE` are exempt. `model.lite`, `model.bin`, `model.litertlm`, `model.tflite.gz`, or an extensionless file are **not** — `.lite` and `.litertlm` are absent from the default list entirely (grepped the whole builder jar).

**Asset path: `app/src/main/assets/model.tflite`. Do not rename it.**

### 1.7 `local.properties` (gitignored)

```properties
sdk.dir=C:/Users/<username>/AppData/Local/Android/Sdk
```

**Forward slashes, mandatory.** This is a Java `.properties` file, so `C:\Users\<username>\...` gets `\U` and `\L` eaten as escapes. The resulting failure is maximally misleading — it does *not* say "SDK location not found", it says:

```
> Could not determine the dependencies of task ':app:compileDebugJavaWithJavac'.
   > java.io.IOException: The filename, directory name, or volume label syntax is incorrect
```

### 1.8 `gradle.properties`

```properties
org.gradle.jvmargs=-Xmx4g -Dfile.encoding=UTF-8
org.gradle.parallel=true
org.gradle.caching=true
android.useAndroidX=true
android.nonTransitiveRClass=true
# android.builder.sdkDownload=false   # uncomment for hermetic offline builds
```

4 GB cap because the device-adjacent build host has 6.91 GB RAM. `android.builder.sdkDownload` defaults to `true`; AGP can self-heal missing SDK packages over the network **without** cmdline-tools because `<SDK>/licenses/android-sdk-license` exists on this host. Setting it false converts a silent download into a hard failure — which is what you want for reproducibility, and not what you want on first build.

### 1.9 `AndroidManifest.xml` — permissions

```xml
<uses-permission android:name="android.permission.CAMERA" />

<uses-feature android:name="android.hardware.camera.any" android:required="false" />
<uses-feature android:name="android.hardware.camera"     android:required="false" />
```

**Both `uses-feature` lines are required.** Per developer.android.com/guide/topics/manifest/uses-feature-element: "The CAMERA permission implies that your app also uses `android.hardware.camera`. A back camera is a required feature unless `android.hardware.camera` is declared with `android:required="false"`." Declaring only `camera.any` leaves an implied **hard** requirement and gets the app filtered on Play for cameraless devices.

**Zero storage permissions.** Not `WRITE_EXTERNAL_STORAGE`, not `READ_MEDIA_IMAGES`, not `MANAGE_EXTERNAL_STORAGE`, no `requestLegacyExternalStorage`. Confirmed from AOSP `Context.java` javadoc on `getExternalFilesDir()`: *"Starting in Build.VERSION_CODES#KITKAT, no permissions are required to read or write to the returned path; it's always accessible to the calling app."*

Location is optional and separate — request `ACCESS_COARSE_LOCATION` only if you log plot coordinates (§3).

---

## 2. INFERENCE PATH — capture to 7-class probability vector

### 2.1 The chain, named transform by transform

```
[1] CameraX ImageCapture.takePicture(executor, OnImageCapturedCallback)
        ↓  ImageProxy, format JPEG (or JPEG_R), pixels in SENSOR orientation
[2] READ imageProxy.imageInfo.rotationDegrees   ← FIRST STATEMENT, before anything else
        ↓  Int in {0, 90, 180, 270}, CLOCKWISE
[3] imageProxy.toBitmap()
        ↓  ARGB_8888 software Bitmap, NO rotation applied, NO crop applied
[4] Matrix().postRotate(degrees) → Bitmap.createBitmap(...)     ← ROTATION HAPPENS HERE, ONLY HERE
        ↓  upright Bitmap, W×H
[5] centre-crop to square, side = min(W, H)
        ↓  edge×edge Bitmap
[6] bilinear downscale to 224×224
        ↓  224×224 ARGB_8888
[7] getPixels() → IntArray(50176) → RGB float32 raw [0,255] into a direct ByteBuffer
        ↓  602112 bytes, ByteOrder.nativeOrder()
[8] Interpreter.run(input, Array(1){FloatArray(7)})
        ↓  7 softmax probabilities
```

### 2.2 Rotation is handled at step [4], and nowhere else

Three verified facts make this the only correct design:

1. **`setTargetRotation` never rotates pixels for `ImageCapture`.** `ImageCapture.java` javadoc: *"This will affect the EXIF rotation metadata in images saved by takePicture calls and the `ImageInfo#getRotationDegrees()` value of the ImageProxy returned by OnImageCapturedCallback."* Metadata only.

2. **`onCaptureSuccess` hands you unrotated pixels.** Same file: the image *"is provided as captured by the underlying ImageReader without rotation applied. The value in `image.getImageInfo().getRotationDegrees()` describes the magnitude of clockwise rotation, which if applied to the image will make it match the currently configured target rotation."*

3. **`toBitmap()` does not rotate.** `ImageProxy.toBitmap()` → `ImageUtil.createBitmapFromImageProxy()`; the JPEG/JPEG_R branch is literally `BitmapFactory.decodeByteArray(bytes, 0, bytes.length, null)`. `BitmapFactory` ignores EXIF orientation. `toBitmap()` applies neither rotation nor crop.

So: `takePicture → toBitmap → model` yields a sensor-oriented bitmap. On a phone held upright with sensor orientation 90, the model sees a landscape image. **No exception, no log, no warning** — the only symptom is degraded accuracy that looks like a bad model.

Keep `targetRotation` current with an `OrientationEventListener`, because a portrait-locked activity never sees display rotation change:

```kotlin
private val orientationListener by lazy {
    object : OrientationEventListener(this) {
        override fun onOrientationChanged(orientation: Int) {
            if (orientation == ORIENTATION_UNKNOWN) return
            // Verbatim from developer.android.com/media/camera/camerax/orientation-rotation
            // The 90/270 inversion is INTENTIONAL: display rotation is CCW, target rotation is CW.
            imageCapture.targetRotation = when (orientation) {
                in 45 until 135  -> Surface.ROTATION_270
                in 135 until 225 -> Surface.ROTATION_180
                in 225 until 315 -> Surface.ROTATION_90
                else             -> Surface.ROTATION_0
            }
        }
    }
}
override fun onStart() { super.onStart(); orientationListener.enable() }
override fun onStop()  { super.onStop();  orientationListener.disable() }
```

**Do not use `ImageAnalysis.setOutputImageRotationEnabled(true)` as the fix.** It is the one CameraX API that rotates pixels, but (a) it exists only on `ImageAnalysis`, not `ImageCapture`, and (b) the bug *"images were not correctly rotated when output image rotation is enabled and the initial relative rotation is 0 degrees"* (Id46c2, b/487160584) was fixed **only in 1.7.0-alpha01** — it is live in the recommended stable 1.6.1.

**Correction to a rationale in circulation:** the rule "rotate before cropping, or you crop the wrong axis" has a false justification. A *centred* square crop of side `min(W,H)` commutes with 90/180/270° rotation — both orders produce identical pixels. Order only matters for a non-square crop region or a non-multiple-of-90 rotation. Keep the rotate-then-crop order (it is correct and generalizes), but do not reason from the stated rule; it will mislead on other crop geometries.

### 2.3 Matching a desktop bilinear resize — read this carefully

Requirement: the 224×224 tensor must match what the model saw at training/eval time on desktop. This is **the highest-risk unverified item in the brief.**

What is certain:
- Channel order **R, G, B**; alpha dropped.
- Values are **raw 0…255 floats**. Do **not** divide by 255. Do **not** subtract a mean. Normalization is inside the graph.
- `ByteBuffer.allocateDirect(1*224*224*3*4)` = 602112 bytes, `.order(ByteOrder.nativeOrder())`.

What is **not** certain, and must be measured rather than assumed: Android's `Bitmap` bilinear scaler and a desktop resize are **not** guaranteed to produce identical pixels. `Bitmap.createScaledBitmap(src, 224, 224, /*filter=*/true)` performs bilinear filtering **without a low-pass prefilter**. `PIL.Image.resize(..., Image.BILINEAR)` in Pillow ≥2.7 applies an antialiasing support window when downscaling. `tf.image.resize(..., method='bilinear', antialias=False)` — the Keras/TF default — does **not**. So:

- If the desktop pipeline was **`tf.image.resize` with default `antialias=False`** (overwhelmingly the common case for a Keras classifier), Android's `createScaledBitmap(filter=true)` is a close match — both are plain bilinear. Residual differences come from `tf.image.resize`'s `half_pixel_centers` sampling convention versus Skia's, which produces small sub-LSB-to-1-LSB deviations, not structural differences.
- If the desktop pipeline used **PIL/`Image.resize` or `antialias=True`**, plain Android bilinear will produce visibly aliased input on a large downscale (a 4000×4000 sensor crop → 224 is a ~18× reduction; plain bilinear samples only 4 source pixels per output pixel and discards ~99.7% of the image). **This is a real accuracy risk, and it is silent.**

**Mandated mitigations, in order:**

1. **Determine the desktop resize empirically, do not guess.** Find the training/eval preprocessing code. If it is `tf.keras.utils.load_img(..., target_size=(224,224))` that is PIL under the hood → antialiased. If it is `tf.image.resize` → not antialiased by default.
2. **Two-stage downscale on Android regardless.** Do the reduction in steps rather than one 18× jump — repeatedly halve with `createScaledBitmap(filter=true)` until within 2× of 224, then do the final scale. This approximates a box prefilter and dramatically reduces aliasing. It is strictly closer to an antialiased desktop resize and only marginally different from a non-antialiased one.
3. **Prove it numerically.** Ship a debug instrumented test that loads a fixed PNG from `androidTest/assets`, runs the exact production `toModelInput()`, and asserts the resulting 602112-byte buffer matches a golden `FloatArray` exported from the desktop pipeline for the same PNG, within a stated tolerance (start at max-abs-diff ≤ 2.0 on the 0…255 scale). **This test is the only thing that actually settles the question.** Without it, "matches desktop" is an assertion, not a fact.

```kotlin
private const val SIDE = 224

/** Rotate-then-crop-then-scale. Two-stage downscale to limit aliasing. */
fun Bitmap.toModelInput(): ByteBuffer {
    val edge = minOf(width, height)
    var b = Bitmap.createBitmap(this, (width - edge) / 2, (height - edge) / 2, edge, edge)

    // Halve repeatedly until within 2x of the target, then final bilinear step.
    while (b.width / 2 >= SIDE) {
        val next = Bitmap.createScaledBitmap(b, b.width / 2, b.height / 2, true)
        if (next !== b) b.recycle()
        b = next
    }
    val scaled = Bitmap.createScaledBitmap(b, SIDE, SIDE, true)
    if (scaled !== b) b.recycle()

    val px = IntArray(SIDE * SIDE)
    scaled.getPixels(px, 0, SIDE, 0, 0, SIDE, SIDE)

    val buf = ByteBuffer.allocateDirect(SIDE * SIDE * 3 * 4).order(ByteOrder.nativeOrder())
    for (p in px) {
        buf.putFloat(((p shr 16) and 0xFF).toFloat())   // R — RAW, do NOT /255
        buf.putFloat(((p shr 8)  and 0xFF).toFloat())   // G
        buf.putFloat(( p         and 0xFF).toFloat())   // B
    }
    buf.rewind()
    return buf
}
```

`ByteOrder.nativeOrder()` is mandatory. Java's `ByteBuffer` defaults to **big-endian**; the native runtime reads little-endian. Omitting it throws nothing and logs nothing — it produces near-uniform or single-class-pinned softmax output. This is the single most common silent-wrong-answer bug in TFLite integration.

### 2.4 Runtime: LiteRT V2 Interpreter, CPU + XNNPACK

```kotlin
import org.tensorflow.lite.Interpreter        // package UNCHANGED from tensorflow-lite

private fun loadModel(ctx: Context, name: String): ByteBuffer =
    ctx.assets.openFd(name).use { afd ->
        FileInputStream(afd.fileDescriptor).use { fis ->
            fis.channel.map(FileChannel.MapMode.READ_ONLY,
                            afd.startOffset,
                            afd.declaredLength)   // NOT afd.length — that runs to end of APK
        }
    }

class Classifier(ctx: Context) : AutoCloseable {
    private val interpreter = Interpreter(
        loadModel(ctx, "model.tflite"),
        Interpreter.Options().apply {
            setNumThreads(4)      // 8 Gen 2 is 1+4+3; oversubscribing little cores costs sync time
            setUseXNNPACK(true)
        }
    )
    private val output = Array(1) { FloatArray(7) }

    fun classify(input: ByteBuffer): FloatArray {
        input.rewind()
        interpreter.run(input, output)
        return output[0].copyOf()
    }
    override fun close() = interpreter.close()
}
```

**The Maven coordinate moved; the Java package did not.** `com.google.ai.edge.litert:litert:2.1.6` ships `org/tensorflow/lite/Interpreter.class` (12 classes, all under `org.tensorflow.lite`). Your `import org.tensorflow.lite.Interpreter` lines are unchanged from the tensorflow-lite era. Only the Gradle line changes. `org.tensorflow:tensorflow-lite` is frozen at 2.17.0, `lastUpdated 20250109` — ~18 months stale.

**Verified absent from 2.1.6** (javap over the real jars, zero matching classes): `org.tensorflow.lite.Delegate`, `DelegateFactory`, `nnapi.NnApiDelegate`, `gpu.GpuDelegate`, `gpu.CompatibilityList`. `Interpreter.Options` in 2.1.6 has exactly four methods: `setNumThreads(int)`, `setUseXNNPACK(boolean)`, `setCancellable(boolean)`, `setRuntime(TfLiteRuntime)`. `addDelegate`, `setUseNNAPI` and `setAllowFp16PrecisionForFp32` are not deprecated — **they are uncompilable**. Any pre-2025 tutorial snippet will fail to build.

**Native libs shipped:** `libLiteRt.so` + `libLiteRtClGlAccelerator.so` for `arm64-v8a`, `armeabi-v7a`, `x86_64` **only**. 32-bit `x86` was dropped after 1.4.2 — an x86 AVD throws `UnsatisfiedLinkError: couldn't find "libLiteRt.so"`. Physical S23+ is arm64-v8a, unaffected; use an x86_64 image for emulator work.

**NNAPI: do not spend one minute on it.** Deprecated in Android 15 per the official NDK migration guide. The device does not declare `android.hardware.neuralnetworks` among its 213 features. The class does not exist in V2. The `nnapi_native.current_feature_level=7` system properties are leftover platform properties and are **not** a capability signal.

**Disclosed tradeoff on the Interpreter API.** The LiteRT README states verbatim: *"MUST USE: The Compiled Model API for all new kotlin and C++ native execution tasks. DO NOT USE: `tflite::Interpreter`, `InterpreterBuilder`, or manual delegate creation. The legacy Interpreter API is strictly deprecated for new features."* The DO-NOT-USE list names C++ symbols, and the Java `org.tensorflow.lite.Interpreter` still ships and works in 2.1.6 — the code above will run. But the MUST-USE clause names Kotlin. Meanwhile the Android docs call the Java Interpreter "maintained for backward compatibility." **These two official framings are in tension and cannot be reconciled from public sources.** Treat the Interpreter path as a *medium-term* choice justified by simplicity for a 224×224×3 → 7-class model, not as the permanently maintained path.

**GPU fallback (only if measurement demands it).** GPU is unreachable from `Interpreter` in V2 — it requires `CompiledModel`:

```kotlin
val env = Environment.create()
val useGpu = Accelerator.GPU in env.getAvailableAccelerators()
val options = if (useGpu) CompiledModel.Options(Accelerator.GPU)
              else CompiledModel.Options(Accelerator.CPU)
val model = CompiledModel.create(ctx.assets, "model.tflite", options, env)
```

**`CompiledModel.CpuOptions` signature correction.** javap on the real artifact gives `CompiledModel$CpuOptions(java.lang.Integer, java.lang.Integer, java.lang.String)` plus a no-arg `CpuOptions()`. There is **no Boolean parameter**. Code written against a `CpuOptions(numThreads = 4, useXnnpack = true)`-shaped signature **will not compile**. The Kotlin parameter names were not decoded — use the no-arg constructor, or verify names against the AAR before writing named arguments. The third parameter's `String` type is consistent with an XNNPACK weight-cache path; the second `Integer` with XNNPACK flags, whose valid constant values are **undocumented — pass null, do not guess.**

`CompiledModel`, `Environment` and `TensorBuffer` all extend `JniHandle implements AutoCloseable`. Use Kotlin `.use { }`; leaking them leaks off-heap memory the GC will not reclaim in time.

Do **not** mix `litert:2.1.6` with `litert-gpu:1.4.2`. `litert-gpu` never advanced past 1.4.2 (confirmed from `group-index.xml`: only `litert` and `litert-api` reached 2.1.6) and it requires the `Delegate`/`DelegateFactory` interfaces V2 deleted. `litert:1.4.2`'s POM pins `litert-api:[1.4.2]` as a **hard range**, so straddling the lines yields either a duplicate-class merge failure or a runtime `NoClassDefFoundError: org/tensorflow/lite/Delegate`.

---

## 3. FIELD LOG DESIGN

### 3.1 The governing principle

**Measured facts and human judgements are different epistemic objects and must never share a column.** The single most destructive mistake available here is writing the user's correction over the model's prediction. That operation is irreversible, throws no error, and permanently destroys the ability to compute the study's primary outcome.

This is not a stylistic preference. DECIDE-AI (Nature Medicine / BMJ 2022) item 12 — *"Report on the user agreement with the AI system. Describe any instances of and reasons for user variation from the AI system's recommendations"* — is unanswerable on an overwritten record. CrowdWorkSheets (Díaz et al., FAccT 2022) asks directly: *"How do the individual annotator responses relate to the final labels released in the dataset?"* Aggregation at write time makes that question unanswerable too.

**The log is append-only. One JSONL line per (image × annotator) event. No updates, no deletes, ever.**

### 3.2 Why this matters for what the data can support — the hard number

In the closest published analogue — PlantVillage Nuru, cassava CMD/CBSD/CGM, East Africa (Frontiers in Plant Science, 2020, doi 10.3389/fpls.2020.590889) — accuracy under the CaSRAT assessment was:

| Rater | Accuracy |
|---|---|
| Trained researchers | **86%** |
| Trained agricultural extension officers | **49%** |
| Trained farmers | **23%** |
| Nuru (the model) | **65%** |

**Untrained field users were roughly 23% accurate — materially worse than the model.** A farmer's confirmation is therefore *not* a label and *not* a reference standard. It is a datum about the human, which happens to also be weak evidence about the leaf. The schema must encode that asymmetry or every downstream analysis will silently treat 23%-accurate opinion as ground truth.

The same study also showed capture conditions dominate accuracy: single-leaf in-field CMD detection was 59%, while six leaves (3 upper + 3 lower) reached **93%**. That is why capture-condition fields are mandatory, not optional metadata.

Field naming follows **Darwin Core** (dwc.tdwg.org/terms/) wherever a ratified term exists — notably `identifiedBy`, whose documented examples include **`MegaDetector V5`**, i.e. Darwin Core explicitly contemplates a machine as the identifying agent. That is exactly the machine-vs-human split needed here.

### 3.3 JSONL record schema

One object per line, UTF-8, `\n`-terminated. Written to `.../research/log.jsonl`.

```jsonc
{
  "schemaVersion": "1.0",
  "recordId": "9f2b...",              // UUID v4, unique per line
  "observationId": "3c81...",         // UUID v4, groups all rows about ONE captured image
  "recordType": "capture",            // "capture" | "expert_review"

  // ─────────── BLOCK A: MEASURED. Machine-generated. Never edited. ───────────
  "measured": {
    "capturedAtEpochMs": 1784563987123,      // System.currentTimeMillis, WALL CLOCK, for ordering only
    "capturedAtElapsedNs": 918273645000,     // SystemClock.elapsedRealtimeNanos, monotonic
    "frameFile": "20260720T143307Z_9f2b.jpg",// relative to research/frames/

    "modelId": "tea-leaf-classifier",
    "modelVersion": "1.3.0",
    "modelSha256": "e3b0c442...",            // hash of the .tflite bytes, computed at load
    "runtime": "litert",
    "runtimeVersion": "2.1.6",
    "api": "Interpreter",                    // "Interpreter" | "CompiledModel"
    "accelerator": "CPU_XNNPACK",            // resolved at runtime, never assumed
    "numThreads": 4,

    "probs": [0.02,0.71,0.05,0.11,0.03,0.06,0.02],  // ALL 7, full precision, never just argmax
    "argmax": 1,
    "topProb": 0.71,
    "entropyNats": 1.0413,                   // -sum p log p, for uncertainty analysis

    "inferenceNs": 4183920,                  // Interpreter.run() ONLY
    "preprocessNs": 12047310,                // rotate+crop+scale+buffer fill
    "captureToBitmapNs": 208441200,          // takePicture callback → Bitmap in hand
    "nativeInferenceNs": 4102887,            // getLastNativeInferenceDurationNanoseconds()

    "thermalStatus": 0,                      // PowerManager.getCurrentThermalStatus(), API 29
    "thermalHeadroom": 0.31,                 // getThermalHeadroom(0), API 30, may be null/NaN
    "sustainedPerfModeEnabled": true,
    "batteryChargeCounterUAh": 3892000,      // BATTERY_PROPERTY_CHARGE_COUNTER, null if unsupported
    "batteryPluggedIn": false,

    "srcWidth": 4000, "srcHeight": 3000,
    "rotationDegreesApplied": 90,            // the value read from ImageInfo, for audit
    "deviceModel": "SM-S916B",
    "androidApiLevel": 36,
    "appVersionName": "1.0", "appVersionCode": 1
  },

  // ─────────── BLOCK B: HUMAN JUDGEMENT. Opinion. Never a label. ───────────
  "judgement": {
    "annotatorId": "a41f...",                // device-generated UUID, user-resettable
    "annotatorExpertise": "FARMER",          // FARMER|EXTENSION_OFFICER|RESEARCHER|PLANT_PATHOLOGIST|UNKNOWN
    "elicitationMode": "BLIND_FIRST",        // BLIND_FIRST | ASSISTED   ← LOAD-BEARING, see §3.4
    "verdict": "DISAGREE",                   // AGREE | DISAGREE | UNSURE | SKIPPED
    "assertedClass": 3,                      // 0..6, or null iff confidence == CANNOT_TELL
    "assertedConfidence": "FAIRLY_SURE",     // SURE | FAIRLY_SURE | GUESSING | CANNOT_TELL
    "identificationQualifier": null,         // dwc: hedge, e.g. "cf."
    "identificationRemarks": null,           // dwc: free text
    "respondedAtEpochMs": 1784563995004,
    "timeToRespondMs": 7881,                 // long latency ≈ genuine deliberation; near-zero ≈ reflexive tap
    "consentVersion": "2026-07-01",
    "consentTimestamp": 1784500000000
  },

  // ─────────── BLOCK C: CAPTURE CONDITIONS. Semi-measured, semi-declared. ───────────
  "conditions": {
    "leavesInFrame": 3,                      // Nuru: 1 leaf 59% → 6 leaves 93% for CMD
    "leafSurface": "UPPER",                  // UPPER | LOWER | MIXED
    "blurScore": 0.87,                       // Laplacian variance, normalized; app-computed
    "lightingSelfReport": "OVERCAST",        // DIRECT_SUN | OVERCAST | SHADE | INDOOR | UNKNOWN
    "decimalLatitude": null,                 // full precision LOCAL ONLY; coarsen on export
    "decimalLongitude": null,
    "coordinateUncertaintyInMeters": null    // dwc: 0 is INVALID — use null when unknown
  },

  // ─────────── BLOCK D: EXPERT VERIFICATION. Separate rows, appended later. ───────────
  // Written as recordType:"expert_review" lines referencing the same observationId.
  // NEVER merged into the capture row.
  "expertReview": null
}
```

An expert review line:

```jsonc
{
  "schemaVersion": "1.0",
  "recordId": "7d10...",
  "observationId": "3c81...",         // links back to the capture
  "recordType": "expert_review",
  "expertReview": {
    "identifiedBy": "Dr A. Nugroho",
    "identifiedByID": "https://orcid.org/0000-0002-...",
    "expertise": "PLANT_PATHOLOGIST",
    "determinedClass": 1,
    "dateIdentified": "2026-07-25",        // dwc, ISO 8601-1:2019
    "method": "VISUAL",                    // VISUAL | LAB_CULTURE | PCR
    "identificationQualifier": null,
    "identificationRemarks": "confirmed on high-res crop",
    "selectedBy": "RANDOM_SAMPLE",         // RANDOM_SAMPLE | DISAGREEMENT | LOW_CONFIDENCE
    "verificationSamplingFraction": 0.10   // ← MANDATORY, see pitfall on partial verification
  }
}
```

### 3.4 Four schema decisions that carry the whole design

**(a) `elicitationMode` — without it the agreement rate is meaningless.** Ship two capture flows and record which produced each row:
- `BLIND_FIRST` — user picks a class **before** the prediction is revealed.
- `ASSISTED` — prediction shown first, user confirms or corrects.

Only `BLIND_FIRST` rows measure user skill. `ASSISTED` rows measure user skill *plus anchoring and automation bias*. Assign `BLIND_FIRST` on a fixed deterministic schedule (e.g. every 5th capture), **never by user choice** — self-selection reintroduces the bias you are trying to isolate. Additionally: make "confirm" and "correct" cost the same number of taps. A large "Correct" button with the correction picker buried two screens deep manufactures agreement.

**(b) `CANNOT_TELL` is a required option, not a courtesy.** Forcing a choice from 7 classes when the user cannot tell produces a fat spurious diagonal on the most visually generic class. This is the analogue of documented "stop points" in confidence-tiering protocols and of Darwin Core `identificationQualifier`. Use a coarse ordinal (`SURE`/`FAIRLY_SURE`/`GUESSING`/`CANNOT_TELL`), not a 0–100 slider — humans are poorly calibrated on fine numeric self-assessment and forced-choice formats yield higher inter-rater reliability.

**(c) `annotatorExpertise` is a closed enum matching the Nuru strata.** `FARMER | EXTENSION_OFFICER | RESEARCHER | PLANT_PATHOLOGIST | UNKNOWN`. Free text ("farmer", "Farmer", "grower", "I have a tea garden") is unstratifiable, which destroys the only published calibration you can anchor to.

**(d) `annotatorId` is a device-generated UUID, user-resettable, stored in encrypted prefs.** Never `ANDROID_ID`, never IMEI, never an account email. On API 36 most hardware identifiers are unreadable anyway, but the real reason is CrowdWorkSheets' question: *"Is there a process by which annotators can later choose to withdraw their data?"* Withdrawal is impossible without a stable, revocable pseudonymous key.

### 3.5 Verification status is DERIVED, never stored

Following iNaturalist's Data Quality Assessment model — where the grade is recomputed from independently-stored atomic votes rather than set as a boolean:

```kotlin
enum class VerificationStatus { UNVERIFIED, COMMUNITY, EXPERT, LAB }

fun statusOf(rows: List<Record>): VerificationStatus {
    if (rows.any { it.expertReview?.method == "PCR" ||
                   it.expertReview?.method == "LAB_CULTURE" }) return LAB
    if (rows.any { it.expertReview?.expertise == "PLANT_PATHOLOGIST" ||
                   it.expertReview?.expertise == "RESEARCHER" }) return EXPERT
    val votes = rows.mapNotNull { it.judgement?.assertedClass }
    if (votes.size >= 2) {
        val top = votes.groupingBy { it }.eachCount().maxBy { it.value }
        if (top.value.toDouble() / votes.size > 2.0/3.0) return COMMUNITY  // iNat threshold
    }
    return UNVERIFIED
}

fun usableAsReferenceStandard(s: VerificationStatus) = s == EXPERT || s == LAB
```

**Only `EXPERT` and `LAB` may serve as a reference standard.** `COMMUNITY` consensus among non-experts who share a misconception converges on the same error — a 2023 field study of iNaturalist lichen records found fewer than half of user-logged species matched specialists' determinations *(reached via search summary; primary paper not fetched — verify before citing)*. Nuru's own reference standard was two researchers with ≥3 years' experience, cross-confirmed on a 75-plant subset by **conventional PCR/qPCR** with CTAB extraction. Note that the Nuru paper reports "percentage matches" between its two experts but describes **no conflict-resolution procedure** — a documented weakness in the closest analogue, not a template to copy. Use ≥2 experts, store both determinations as separate rows, and report disagreement.

### 3.6 What this data can and cannot support

**Can support:**
- User–model agreement rate, per class, per expertise stratum, split by `elicitationMode` (DECIDE-AI item 12).
- Correction *patterns* — which classes users systematically move predictions away from. A hypothesis generator, not a conclusion.
- Model accuracy against the `EXPERT`/`LAB` subset, **provided** `verificationSamplingFraction` is recorded and corrected for.
- Distribution-shift evidence: field image statistics versus the training set.
- Latency, thermal behaviour, memory, and usability/learning-curve claims (DECIDE-AI item 14b).

**Cannot support, and must not be claimed:**
- **"Field accuracy is X%"** derived from non-expert confirmations. Nuru measured trained farmers at 23%.
- **"The model was validated in the field."** CLAIM 2024 explicitly discourages "validation" because it reads as clinical validation — say **"evaluated."** *(CLAIM 2024 item numbers could not be verified: `pubs.rsna.org/doi/full/10.1148/ryai.240300` returns HTTP 403. Items 14–18 quoted from the 2020 version, PMC8017414. The "reference standard over ground truth" and "evaluation over validation" terminology changes were confirmed via the journal's own summary page.)*
- Per-class sensitivity/specificity from a partially-verified subset without an explicit sampling-fraction correction.
- **Any retraining claim.** User corrections at ~23% expected accuracy are a label-noise injection, not a training signal.

**Also record once, at dataset level:** the preprocessing contract — *input 224×224×3 float32, raw [0,255], normalization embedded in the graph, RGB, two-stage bilinear downscale* — as free text under the Croissant-RAI property `rai:dataAnnotationProtocol`. A future maintainer who re-normalizes to [0,1] will silently halve accuracy with no error. *(Croissant-RAI property names — `rai:dataAnnotationProtocol`, `rai:dataAnnotationPlatform`, `rai:annotationsPerItem`, `rai:annotatorDemographics`, `rai:dataCollectionType`, `rai:dataCollectionTimeframe`, `rai:personalSensitiveInformation` — were read from docs.mlcommons.org/croissant/docs/croissant-rai-spec.html, but the page showed no version banner and Croissant 1.1 shipped Feb 2026. Re-check property names against the live spec before writing an exporter.)*

**Statistics:** report **Krippendorff's α**, not Cohen's κ, alongside the full 7×7 confusion matrix and per-class recall. κ assumes two fixed raters with equal marginals; you will have variable raters per item and missing data, which is α's design case. Also beware the kappa paradox — κ is deflated under skewed class distributions and can read ~0.2 while raters visibly agree. Consider Gwet's AC1 as a stability check. *(κ paradox and AC1 come from search summaries, not fetched primary sources — a lead to verify, not a settled recommendation.)*

**No agricultural ML annotation standard exists.** This was searched for explicitly and not found; the literature contains only review papers calling for standardization. This schema is an adaptation of medical (CLAIM, DECIDE-AI), crowdsourcing (CrowdWorkSheets), and biodiversity (Darwin Core, iNaturalist) practice. **State that in writing.** Do not imply conformance to an agricultural standard.

---

## 4. RETRIEVAL — pulling the log and images

### 4.1 Where the app must write

```kotlin
private val root: File = File(
    requireNotNull(context.getExternalFilesDir(null)) {
        "External storage unavailable: ${Environment.getExternalStorageState()}"
    },
    "research"
).apply { mkdirs() }

private val framesDir = File(root, "frames").apply { mkdirs() }
private val jsonl     = File(root, "log.jsonl")
```

Resolves at runtime to `/storage/emulated/0/Android/data/com.example.tealeaf/files/research/`, reachable as `/sdcard/Android/data/com.example.tealeaf/files/research/`.

**Never construct that path by string concatenation.** Android 11+ forbids apps creating their own directory on external storage; `mkdirs()` on a hand-built path returns `false` and every write throws `ENOENT`/`EACCES`. Always go through `getExternalFilesDir(null)` and `mkdirs()` only on the subdirectory beneath it. `getExternalFilesDir()` is `@Nullable` — null-check it.

**Log the resolved absolute path at startup** so the pull command can be copied verbatim rather than assumed:

```kotlin
Log.i("ResearchLog", "adb pull ${root.absolutePath} .")
```

### 4.2 Why `adb pull` works on a non-rooted Android 16 device

The Android 11+ `Android/data` lockdown is real but is scoped to **apps**, not the shell. No developer.android.com page states the shell exemption — this is established from AOSP source, and is **inference from code, not a documented API contract**:

`packages/providers/MediaProvider/jni/FuseDaemon.cpp`:

```c
static bool is_app_accessible_path(struct fuse* fuse, const string& path, uid_t uid) {
    MediaProviderWrapper* mp = fuse->mp;
    if (uid < AID_APP_START || uid == MY_UID) {
        return true;
    }
    ...
```

`AID_APP_START` is 10000. `adb shell` runs as uid 2000 (`AID_SHELL`), so the function short-circuits to `return true` **before** the `Android/data` ownership regex is ever evaluated. Independently, `packages/modules/adb/daemon/main.cpp` gives adbd the supplementary GID `AID_EXT_DATA_RW (1078)` — "GID for app-private data directories on external storage."

This is exactly why file-manager apps (uid ≥ 10000, including on Samsung One UI) *cannot* browse `Android/data` while adb can. Same kernel, different uid, different outcome. Do not let XDA threads about file managers talk you into requesting `MANAGE_EXTERNAL_STORAGE` — it is unnecessary here and a Play policy problem.

**Nothing changed in Android 16.** Both `developer.android.com/about/versions/16/behavior-changes-all` and `.../behavior-changes-16` were read end to end: zero external-storage, scoped-storage, `Android/data`, `getExternalFilesDir`, or adb/shell file-access items.

**Not empirically confirmed on the SM-S916B.** `adb devices -l` returned an empty list during research — the device was not attached. The verdict above rests on AOSP source. Samsung has historically added OEM storage restrictions; per the AOSP code those target app UIDs via SAF/FUSE and cannot reach uid 2000, but **run the 30-second check below before building anything on this assumption.**

### 4.3 Exact commands (PowerShell, Windows 11)

```powershell
$ADB = "C:/Users/<username>/AppData/Local/Android/Sdk/platform-tools/adb.exe"
$PKG = "com.example.tealeaf"
$REM = "/sdcard/Android/data/$PKG/files/research"

# 0. Device present and authorized (must print "device", not "unauthorized")
& $ADB devices -l
& $ADB wait-for-device

# 1. THE 30-SECOND PROOF — run this once before trusting §4.2
& $ADB shell ls -la $REM

# 2. Primary pull: whole tree, preserving device timestamps and modes
New-Item -ItemType Directory -Force "D:\tea\pulled" | Out-Null
& $ADB pull -a $REM "D:\tea\pulled"
#   -> D:\tea\pulled\research\log.jsonl
#   -> D:\tea\pulled\research\frames\20260720T143307Z_9f2b.jpg

# 3. JSONL only
& $ADB pull "$REM/log.jsonl" "D:\tea\pulled\log.jsonl"

# 4. Tail the log live without pulling
& $ADB shell cat "$REM/log.jsonl"

# 5. Clear between runs, no uninstall needed
& $ADB shell rm -rf $REM
```

`adb pull` flags verified by running `adb help` locally (adb 1.0.41 / 37.0.0-14910828): `pull [-a] [-z ALGORITHM] [-Z] REMOTE... LOCAL`, where `-a` = "preserve file timestamp and mode". `-q` suppresses progress, `-Z` disables compression. Directory pulls are recursive.

**Deleted from earlier guidance:** `adb shell echo /sdcard/Android/data/<pkg>/files/research`, presented as a way to "discover the real path." `echo` prints its literal argument back. It verifies nothing. `ls -la` is the command that actually checks existence.

### 4.4 Debug-only fallback for internal storage

Only needed if you ever also want `/data/user/0/<pkg>/`:

```powershell
& $ADB exec-out run-as $PKG tar -c -C files research > "D:\tea\pulled\research.tar"
& $ADB exec-out run-as $PKG cat files/research/log.jsonl > "D:\tea\pulled\log.jsonl"
```

**`exec-out`, never `shell`, for binary redirects on Windows** — the `shell` transport allocates a PTY that performs LF→CRLF translation, silently corrupting JPEGs and tar archives.

Both commands fail on a release APK with `package not debuggable: <pkg>`. `system/core/run-as/run-as.cpp` gates on exactly two things: caller must be shell or root, and `if (!info.debuggable) error(1, 0, "package not debuggable: %s", pkgname);`. There is no `profileable` fallback. **This is the whole argument for `getExternalFilesDir()`: it behaves identically in debug and release.**

### 4.5 The path not taken, and why

`developer.android.com/training/data-storage/use-cases` says verbatim: *"For test output, it's better to instead write to app-scoped storage that's readable by the shell... To determine which directory to pull from, call `getExternalMediaDirs()`."* That is `Android/media/<pkg>/`, and it is Google's only explicit guidance for this exact use case.

**Do not follow it.** `getExternalMediaDirs()` is `@Deprecated` in AOSP `Context.java`: *"These directories still exist and are scanned, but developers are encouraged to migrate to inserting content into a MediaStore collection directly."* The use-cases page is stale relative to the API reference. Additionally, `Android/media` **is** MediaStore-scanned — research JPEGs would appear in the user's gallery, violating the privacy constraint. `Android/data` is not scanned.

Net: `getExternalFilesDir(null)` is non-deprecated, permission-free, gallery-invisible, and adb-reachable. Its only weakness is that reachability rests on a UID bypass rather than a documented promise — which §4.3 step 1 closes out empirically.

### 4.6 Durability and filename rules

```kotlin
fun append(json: String) {
    FileOutputStream(jsonl, /* append = */ true).use { fos ->
        fos.write((json + "\n").toByteArray(Charsets.UTF_8))
        fos.flush()
        fos.fd.sync()          // durability: adb pull never sees a torn tail
    }
}
```

`adb pull` reads whatever is on disk *right now*. A `BufferedWriter` holding the last N records yields a file ending mid-line, and the JSONL parser throws on the final record. `flush()` + `fd.sync()` after each record. Parse defensively anyway — skip a trailing partial line.

**Filenames: `[A-Za-z0-9._-]` only.** ISO-8601 extended format (`2026-07-20T14:33:07Z`) is legal on Android's FUSE volume but `:` is **illegal in Windows filenames**. Symptom: `adb pull` reports directory success while individual files are silently missing, or `couldn't create file: Invalid argument`. Use basic format: `20260720T143307Z_9f2b.jpg`. Timestamp via `SimpleDateFormat("yyyyMMdd'T'HHmmss'Z'", Locale.US)`.

**Uninstall and "Clear storage" wipe the log** (docs: "When the user uninstalls your app, the files saved in app-specific storage are removed"). `adb install -r` does **not** wipe it; `adb uninstall` does. Pull before every uninstall; put a pull step in the iteration script.

Note the directory does not exist until the app has run once. A pull before first launch fails with "does not exist" — a benign error, not the restriction biting. Do not misread it.

---

## 5. BENCHMARK HARNESS

### 5.1 Clocks

- **Inner per-inference timer: `System.nanoTime()`.** Monotonic, uptime-based ("This is the basis for most interval timing such as Thread.sleep(millis), Object.wait(millis), and System.nanoTime()").
- **Outer session timeline and thermal-sample timestamps: `SystemClock.elapsedRealtimeNanos()`.** "guaranteed to be monotonic," includes deep sleep, "the recommended basis for general purpose interval timing."
- **Never `System.currentTimeMillis()` in a timing path.** "may jump backwards or forwards unpredictably." An NTP sync mid-run produces negative or wildly inflated durations that survive outlier filters because they look like plausible tail latency.

A screen-on foreground inference loop cannot deep-sleep, so `nanoTime()` and `elapsedRealtimeNanos()` are numerically equivalent for the inner timer. *(That equivalence is my inference from the two clocks' documented definitions, not a doc statement.)*

### 5.2 Three separately-timed stages, never conflated

| Stage | What is timed | Reported as |
|---|---|---|
| **A** | `takePicture` callback → `Bitmap` in hand (incl. JPEG decode + Matrix rotate) | camera acquisition |
| **B** | Bitmap → 224×224×3 float32 `ByteBuffer` (crop, two-stage scale, `getPixels`, float writes) | preprocessing |
| **C** | `interpreter.run(input, output)` **alone** | **model inference latency** |
| A+B+C | | **end-to-end pipeline latency** |

**Report C as "model inference latency." Report A+B+C as "end-to-end." Never let one masquerade as the other.** For C, drive the interpreter from **one pre-filled static buffer with no camera running at all** — otherwise camera thread contention contaminates the number you publish.

Cross-check C against `interpreter.getLastNativeInferenceDurationNanoseconds()`; a large gap between wall-clock C and native duration indicates JNI/marshalling overhead worth reporting separately.

### 5.3 Warm-up and repetition

**Warm-up 50, measure 500–1000.** Google's own `benchmark_model` defaults are `warmup_runs=1` and `num_runs=50` — those exist for quick triage, **not publication**. The tool itself reports four separate quantities (`Init`, `First inference`, `Warmup (avg)`, `Inference (avg)`), which is the strongest available evidence that first-inference and steady-state are distinct and must be reported separately. Timing the first inference and calling it "latency" produces a headline number 5–50× the steady state.

Jetpack Microbenchmark's docs state warm-up and iteration counts are handled automatically because "complex microbenchmarks can take a long time to stabilize." With a hand-rolled harness, watch for a descending latency curve over the first tens-to-hundreds of iterations — that is warm-up, not signal.

**Report median, p95, and IQR or MAD. Not mean ± SD.** Latency distributions are right-skewed.

### 5.4 Device stabilization — `lockClocks` is unavailable

`gradlew lockClocks` "requires a rooted Android device" and is "not supported on most devices." Off the table on a non-rooted S23+.

The available lever is **Sustained Performance Mode**:

```kotlin
fun Activity.enableSustainedPerformance(): Boolean {
    val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
    val supported = pm.isSustainedPerformanceModeSupported          // API 24
    if (supported) window.setSustainedPerformanceMode(true)
    return supported          // LOG THIS BOOLEAN INTO EVERY RECORD
}
```

Per AOSP's performance-management page: *"the power hint caps the maximum frequencies of the CPU and GPU at the highest sustainable levels."* So your absolute latencies **will be higher** than an unconstrained burst run. Both numbers are defensible; silently mixing them across conditions is not. State in the paper that you used it.

### 5.5 Thermal instrumentation

```kotlin
class ThermalMonitor(ctx: Context) {
    private val pm = ctx.getSystemService(Context.POWER_SERVICE) as PowerManager
    private var lastHeadroomNs = 0L
    private var cached = Float.NaN

    fun status(): Int = pm.currentThermalStatus        // API 29, cheap, safe per-iteration

    fun headroom(): Float {                            // API 30 — MAX ONCE PER 10 SECONDS
        val now = SystemClock.elapsedRealtimeNanos()
        if (now - lastHeadroomNs >= 10_000_000_000L) {
            cached = pm.getThermalHeadroom(0)
            lastHeadroomNs = now
        }
        return cached
    }

    fun thermalApiSupported(): Boolean {
        val v = pm.getThermalHeadroom(0)
        return !(v.isNaN() || v == 0f)
    }
}
```

Constants: `THERMAL_STATUS_NONE`=0, `LIGHT`=1, `MODERATE`=2, `SEVERE`=3, `CRITICAL`=4, `EMERGENCY`=5, `SHUTDOWN`=6. `getThermalHeadroom(int forecastSeconds)` returns 0.0f–1.0f (0.0 = no throttling, 1.0 = `THERMAL_STATUS_SEVERE`), **NaN if unsupported or called too frequently**. Documented heuristics: headroom > 1.0 → possibly severely throttled; > 0.95 → moderately; > 0.85 → light.

**Critical caveat, verbatim from the ADPF page: *"Some devices return `THERMAL_STATUS_NONE` regardless of actual thermal state. Validate with `getThermalHeadroom()` instead."*** The hardware probe on this SM-S916B reported thermal status 0. **That reading is not by itself evidence the device is unthrottled.** Run `thermalApiSupported()` before writing any sentence claiming absence of throttling; if it returns false, say the API is unsupported on this unit rather than claiming no throttling was observed.

Log `getCurrentThermalStatus()` per iteration and headroom once per 10 s. Discard, or separately report, any block where status ≠ 0 or headroom > 0.85.

### 5.6 Memory

```kotlin
class PeakPssSampler(private val intervalMs: Long = 100) : Thread() {
    @Volatile var running = true
    @Volatile var peakKb = 0
    override fun run() {
        val mi = Debug.MemoryInfo()
        while (running) {
            Debug.getMemoryInfo(mi)
            peakKb = maxOf(peakKb, mi.totalPss)   // getTotalPss(), API 5, kB
            sleep(intervalMs)
        }
    }
}
```

**Never sample PSS inside the timed loop.** `Debug.getMemoryInfo()` walks `/proc/<pid>/smaps`; the cost scales with process memory size and directly contaminates the latency number you publish. Separate thread, or between conditions only. Record the sampling interval in the paper.

PSS is the officially blessed footprint metric: *"This (PSS) total is what the system considers to be your physical memory footprint."* `getMemoryStat(String)` (API 23) accepts exactly nine keys — `summary.java-heap`, `summary.native-heap`, `summary.code`, `summary.stack`, `summary.graphics`, `summary.private-other`, `summary.system`, `summary.total-pss`, `summary.total-swap` — and the javadoc warns they are **approximate**, that "individual allocations may not be immediately reflected," and that keys "may be added or removed in a future API level."

Host cross-check: `adb shell dumpsys meminfo -d com.example.tealeaf` (`-d` = "Prints more info related to Dalvik and ART memory usage").

*Do not cite an API level for `getTotalRss()` — it exists in AOSP `Debug.java` but its API level was not verifiable.*

### 5.7 Energy — the honest answer

**You cannot make a per-app energy claim on this device.** Three official statements close it off:

- **Power Profiler / ODPM:** *"Power Profiler reads power consumption data from the ODPM, which is only available on Pixel 6 and subsequent Pixel devices."* And: *"ODPM measures power consumption at the device level—not specific to any app."*
- **Macrobenchmark `PowerMetric`** (marked *Experimental*): *"These metrics measure system-wide consumption, not the consumption on a per-app basis, and are limited to Pixel 6, Pixel 6 Pro, and later devices."*
- **BatteryStats** is a model, not a measurement: it *"doesn't track battery current draw directly, but instead collects timing information that can be used to approximate battery consumption,"* estimating from OEM-supplied `power_profile.xml` coefficients. The page explicitly lists what it does not provide: *"Quantitative battery consumption amounts."* Battery Historian is *"no longer actively maintained."*

The only defensible route is the whole-device coulomb counter:

```kotlin
fun chargeCounterMicroAh(ctx: Context): Long? {
    val bm = ctx.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    val v = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)  // µAh
    return if (v == Long.MIN_VALUE) null else v      // MIN_VALUE = property unsupported
}
```

Protocol: **unplugged** (the counter is meaningless while charging, and `CURRENT_NOW` goes positive), airplane mode, fixed screen brightness, screen on. Run ≥10–30 min of continuous inference. Measure ΔµAh. Run a **matched-duration idle baseline with identical screen state**. Report `(Δ_inference − Δ_idle)` mAh per N inferences, n ≥ 5 repetitions, with a dispersion measure. Convert to energy only by multiplying by nominal pack voltage, and say so explicitly. **Label it device-level, not per-app.**

*(`BATTERY_PROPERTY_CHARGE_COUNTER` = µAh, `CURRENT_NOW` = µA, and the `Long.MIN_VALUE` sentinel come from a search-result summary quoting the official reference page, not a page fetched directly — one notch below the rest of this brief. No authoritative statement was found for `BATTERY_PROPERTY_ENERGY_COUNTER` units; do not cite it.)*

### 5.8 Jetpack Benchmark module (optional but recommended)

Root `build.gradle`:
```groovy
plugins { id 'androidx.benchmark' version '1.4.1' apply false }
```
Benchmark module:
```groovy
plugins { id 'androidx.benchmark' }
android {
  defaultConfig {
    testInstrumentationRunner "androidx.benchmark.junit4.AndroidBenchmarkRunner"
  }
}
dependencies { androidTestImplementation "androidx.benchmark:benchmark-junit4:1.4.1" }
```

The plugin AOT-compiles the benchmark APK by default (needs Benchmark ≥1.3.0-beta01 and AGP ≥8.4.0 — 8.13.0 qualifies); disable with `androidx.benchmark.forceaotcompilation=false`. Use `runWithTimingDisabled { }` (Kotlin) or `state.pauseTiming()/resumeTiming()` (Java) to exclude the buffer-fill setup from the measured region — this is exactly the A/B/C separation of §5.2.

*Note a doc inconsistency: the microbenchmark-write guide shows `benchmark-junit4:1.2.4` while the authoritative releases page gives 1.4.1. Trust the releases page.*

### 5.9 Reference cross-check against Google's own tool

```powershell
& $ADB install -r -d -g android_aarch64_benchmark_model.apk
& $ADB push model.tflite /data/local/tmp
& $ADB shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity `
    --es args '"--graph=/data/local/tmp/model.tflite --num_threads=4 --warmup_runs=50 --num_runs=500 --use_xnnpack=true --report_peak_memory_footprint=true"'
```
APK: `https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk`

Verified flags/defaults: `num_runs`(50), `warmup_runs`(1), `max_secs`(150.0), `num_threads`(-1), `use_gpu`(false), `use_nnapi`(false), `use_xnnpack`(false), `enable_op_profiling`(false), `report_peak_memory_footprint`(false), `memory_footprint_check_interval_ms`(50).

**Never pass `--use_nnapi=true`.** It silently falls back to CPU on this device, and you would publish "NNAPI accelerated" numbers that are plain CPU numbers.

---

## 6. PITFALL CHECKLIST — silent failure modes, each with its symptom

### Build / dependencies

| # | Pitfall | Observable symptom |
|---|---|---|
| 1 | `mavenCentral()` without `google()` | `Could not find com.google.ai.edge.litert:litert:2.1.6`. Artifacts are on Google Maven despite the official page saying MavenCentral. |
| 2 | Backslashes in `local.properties` `sdk.dir` | `java.io.IOException: The filename, directory name, or volume label syntax is incorrect` on `:app:compileDebugJavaWithJavac`. Does **not** say "SDK not found." |
| 3 | `gradle wrapper` in a directory with no settings file | `Directory ... does not contain a Gradle build`. Create `settings.gradle.kts` first. |
| 4 | `implementation("com.google.ai.edge.litert:+")` (from the official dev-tools page) | Nondeterministic resolution; one day silently jumps the V1→V2 breaking boundary. Always pin `2.1.6`. |
| 5 | Mixing `litert:2.1.6` with `litert-gpu:1.4.2` | `NoClassDefFoundError: org/tensorflow/lite/Delegate` at runtime, or duplicate-class errors at merge. `litert:1.4.2`'s POM pins `litert-api:[1.4.2]` as a hard range. |
| 6 | Copy-pasting `addDelegate(GpuDelegate())` / `setUseNNAPI(...)` from a pre-2025 tutorial | Compile error `unresolved reference: addDelegate`. Reads as a bad import; is actually "the method and its parameter type no longer exist." |
| 7 | `CompiledModel.CpuOptions(numThreads = 4, useXnnpack = true)` | Does not compile. Real signature is `(Integer, Integer, String)` — no Boolean. |
| 8 | Moving to AGP 9.x without `buildToolsVersion` | Network auto-download of build-tools 36.0.0, or hard failure offline. Also needs Gradle ≥9.1.0 which is not cached. |
| 9 | Renaming the model away from a `.tflite` suffix | `FileNotFoundException: This file can not be opened as a file descriptor; it is probably compressed` at model load. AGP's no-compress match is `.tflite` suffix only, case-insensitive. |
| 10 | x86 (32-bit) emulator image | `UnsatisfiedLinkError: couldn't find "libLiteRt.so"`. 2.1.6 ships arm64-v8a / armeabi-v7a / x86_64 only. |
| 11 | Only `<uses-feature android:name="android.hardware.camera.any" required="false"/>` | App filtered on Google Play for cameraless devices — the CAMERA permission implies a *hard* `android.hardware.camera` requirement unless separately declared `required="false"`. |
| 12 | `minSdk` 21 or 22 with CameraX 1.6.1 | Build failure. CameraX floor rose to 23 in 1.5.0-rc01. |

### Inference correctness — the silent ones

| # | Pitfall | Observable symptom |
|---|---|---|
| 13 | **Missing `ByteOrder.nativeOrder()`** | No crash, no log. Wildly wrong softmax — often near-uniform or pinned to one class. Java defaults big-endian; the native runtime reads little-endian. **The single most common silent-wrong-answer bug.** |
| 14 | **Normalizing twice** (dividing by 255 in Kotlin when the graph already does it) | Uniform/near-uniform softmax; every image returns the same class. Looks exactly like a broken model rather than a broken pipeline. |
| 15 | **Not rotating after `toBitmap()`** | Model is near-random or systematically confuses classes; works perfectly on the same images from the desktop harness. No exception, no log, no warning. |
| 16 | `afd.length` instead of `afd.declaredLength` when mmapping | `IllegalArgumentException`, or a garbage oversized mapping — for an uncompressed asset `length` runs to the end of the APK. |
| 17 | Reading `rotationDegrees` **after** `imageProxy.close()` | `IllegalStateException`, or a garbage rotation value. Capture it as the first statement in `onCaptureSuccess`. |
| 18 | Double rotation (save JPEG → reload with an EXIF-aware loader → *also* apply `rotationDegrees`) | Image is 180° off rather than 90° — deceptively close to correct, survives a quick glance. Pick exactly one source of truth. |
| 19 | Activity locked to `screenOrientation="portrait"` without an `OrientationEventListener` | Works in portrait, wrong by 90° in landscape. Display rotation never changes so the default target rotation is frozen. |
| 20 | Not calling `imageProxy.close()` | Capture works 2–3 times then the camera silently stops delivering frames. Use `try/finally`. Never call `Media.Image.close()` on the wrapped image. |
| 21 | `Bitmap.Config.HARDWARE` reaching `getPixels()` | `IllegalStateException: unable to getPixels(), pixel access is not supported on Config#HARDWARE bitmaps`. `BitmapFactory.decodeByteArray` gives software ARGB_8888 so the recommended path is safe; if you route through `ImageDecoder`, call `setAllocator(ALLOCATOR_SOFTWARE)`. |
| 22 | **Resize-method mismatch with the desktop pipeline** | Accuracy degrades by several points with no error. Plain Android bilinear on an ~18× downscale aliases badly versus an antialiased PIL resize. **Mitigated by the two-stage downscale and settled only by the golden-tensor test (§2.3).** |
| 23 | Front camera enabled | Mirroring. `isReversedHorizontal`'s javadoc describes only the save-to-file path; whether the in-memory `OnImageCapturedCallback` path signals mirroring at all is **unconfirmed**. Rear camera only avoids this entirely. |
| 24 | `ImageAnalysis.setOutputImageRotationEnabled(true)` on 1.6.1 with initial relative rotation 0 | Incorrectly rotated images. Fixed only in 1.7.0-alpha01 (Id46c2, b/487160584) — live in stable 1.6.1. |
| 25 | Assuming NNAPI helps | Delegate creation returns null / `IllegalArgumentException: Internal error: Error applying delegate`, **or silent CPU fallback that looks like it worked**. Device lacks `android.hardware.neuralnetworks`; the class does not exist in V2 anyway. |
| 26 | Not calling `.close()` on `CompiledModel`/`Environment`/`TensorBuffer` | Off-heap memory leak the GC will not reclaim in time. Use `.use { }`. |

### Storage / retrieval

| # | Pitfall | Observable symptom |
|---|---|---|
| 27 | Colons in filenames (ISO-8601 extended timestamps) | `adb pull` reports directory success while individual files are silently missing, or `couldn't create file: Invalid argument`. Windows host. |
| 28 | Buffered writes without `fd.sync()` | Pulled JSONL ends mid-line; the parser throws on the final record. |
| 29 | Hardcoding `/sdcard/Android/data/...` and calling `mkdirs()` | `mkdirs()` returns `false`; every write throws `ENOENT`/`EACCES`. Android 11+ forbids apps creating their own external dir. |
| 30 | Not null-checking `getExternalFilesDir()` | Kotlin NPE, or a silent no-op log. |
| 31 | `adb shell run-as ... cat` for binary on Windows | JPEGs subtly larger than on-device and won't decode; tar fails with "unexpected EOF". PTY does LF→CRLF. Use `exec-out`. |
| 32 | Prototyping the pull workflow via `run-as` on internal storage | Works in debug, then `package not debuggable` the first time you ship release. |
| 33 | `adb uninstall` before pulling | Research data destroyed. `adb install -r` is safe; `adb uninstall` and "Clear storage" are not. |
| 34 | `adb pull <remotedir> <localdir>` when `<localdir>` doesn't exist | adb renames the tree to `<localdir>` instead of creating `<localdir>\research`. Files land a level higher than the script expects. Create the destination first. |
| 35 | Believing XDA/file-manager reports that `Android/data` needs root | Leads to requesting `MANAGE_EXTERNAL_STORAGE` — unnecessary and a Play policy problem. Those reports concern apps at uid ≥ 10000; adb is uid 2000. |

### Measurement

| # | Pitfall | Observable symptom |
|---|---|---|
| 36 | `getThermalHeadroom()` more than once per 10 s | Thermal log full of NaN; you wrongly conclude the API is unsupported. |
| 37 | Trusting `getCurrentThermalStatus() == 0` | A paper claim of "no thermal throttling observed" that is actually an unimplemented HAL. *"Some devices return THERMAL_STATUS_NONE regardless of actual thermal state."* |
| 38 | Benchmarking a `debuggable = true` build | Latencies inflated by a large, non-constant factor from JVMTI hooks and disabled optimizations. |
| 39 | Insufficient warm-up (e.g. `benchmark_model`'s default `warmup_runs=1`) | A descending latency curve over the first tens-to-hundreds of iterations, mistaken for signal. |
| 40 | Sampling PSS inside the timed loop | Measured inference latency inflates by milliseconds, scaling with process memory — you contaminate exactly the number you publish. |
| 41 | `System.currentTimeMillis()` in the timing path | Rare unexplained outliers that survive outlier filtering because they look like plausible tail latency. NTP sync mid-run. |
| 42 | Reporting first inference as "latency" | A headline number 5–50× the steady state. First call includes delegate setup and lazy allocation. |
| 43 | `getLongProperty` without checking `Long.MIN_VALUE` | `Long.MIN_VALUE` averaged into results → absurd energy numbers. |
| 44 | Coulomb counter measured while plugged in | Negative or nonsensical "energy consumed"; `CURRENT_NOW` is positive when charging. |
| 45 | Citing ODPM / `PowerMetric` on a Galaxy S23+ | APIs return nothing; or worse, a Pixel-derived methodology cited in a paper whose measurements came from a Samsung device. |
| 46 | Assuming `lockClocks` worked | Requires root, "not supported on most devices." CPU frequency is actually floating and your variance is governor noise. |
| 47 | Mixing sustained-performance-mode and burst runs across conditions | Systematically higher latencies in one arm with no recorded cause. Log `sustainedPerfModeEnabled` in every record. |

### Study design / data

| # | Pitfall | Observable symptom |
|---|---|---|
| 48 | **Showing the prediction before the user answers** | Agreement rate climbs toward 90%+ and looks like validation. It is anchoring/automation bias. **The biggest threat to this study.** Mitigate with `BLIND_FIRST` and symmetric confirm/correct tap cost. |
| 49 | **Overwriting the model prediction with the user's correction** | Silent and terminal. Dataset looks fine; agreement rate, correction patterns, and DECIDE-AI item 12 are permanently uncomputable. Enforce append-only at the storage layer. |
| 50 | Aggregating multiple responses to a majority label at write time | Inter-annotator agreement uncomputable; the disagreement signal destroyed. Aggregate in analysis only. |
| 51 | **Verifying only cases the model flagged uncertain** (partial verification bias) | Sensitivity biased **up**, specificity **down**. Every accuracy number inflated. Verify all disagreements **plus a random sample of agreements**, and store `verificationSamplingFraction`. |
| 52 | Reporting a bare Cohen's κ on 7 imbalanced classes | κ ≈ 0.2 while raters visibly agree; you wrongly conclude annotation is unreliable. Report Krippendorff's α + confusion matrix + per-class recall. |
| 53 | No `modelVersion` / `accelerator` on the record | You ship a retrained `.tflite`, accuracy shifts, and you cannot tell whether the model changed or the season did. |
| 54 | No "cannot tell" option | A fat spurious diagonal on the most visually generic class. Users who can't tell pick the model's suggestion or the most familiar label. |
| 55 | Free-text expertise field | Unstratifiable data; the Nuru-style stratified comparison — the entire point of recording expertise — becomes impossible. |
| 56 | Hardware identifier as `annotatorId` | Consent withdrawal impossible; permission failure or null on API 36; privacy liability. |
| 57 | Treating one expert's opinion as the reference standard | Unmeasurable error floor. CLAIM item 18 requires methods to measure inter- and intra-rater variability. |
| 58 | Treating non-expert consensus as truth | Confident, wrong reference standard — shared misconceptions converge on the same error. |
| 59 | Publishing photos with full-precision GPS | A grower's diseased plot publicly geolocated; commercially sensitive. Store full precision locally, coarsen on export, use `coordinateUncertaintyInMeters`. |
| 60 | Claiming "field accuracy is X%" from user confirmations | Reviewer rejection. Nuru measured trained farmers at 23%. |

---

## 7. BUILD AND TEST COMMANDS (Windows, Gradle wrapper)

### 7.1 One-time wrapper bootstrap

The `wrapper` task fails without an existing settings file, so create it first:

```powershell
Set-Location "D:\tea\tea-leaf-android-feia\android"
Set-Content -Path "settings.gradle.kts" -Value 'rootProject.name = "tealeaf"' -Encoding utf8
& "C:/Users/<username>/.gradle/wrapper/dists/gradle-8.14-all/<cache-id>/gradle-8.14/bin/gradle.bat" `
    wrapper --gradle-version 8.14 --distribution-type all
```

This produces `gradlew`, `gradlew.bat`, `gradle/wrapper/gradle-wrapper.jar`, `gradle/wrapper/gradle-wrapper.properties`. The cached distribution path was verified to resolve on this host; it is the only wrapper distribution present, which is why nothing needs the network.

Expected `gradle-wrapper.properties`:
```
distributionUrl=https\://services.gradle.org/distributions/gradle-8.14-all.zip
```

### 7.2 Build

```powershell
Set-Location "D:\tea\tea-leaf-android-feia\android"

.\gradlew.bat :app:assembleDebug
#   -> app\build\outputs\apk\debug\app-debug.apk  (signed with the debug key, zipaligned)

.\gradlew.bat :app:assembleRelease
.\gradlew.bat clean
.\gradlew.bat :app:tasks --group=verification      # confirm task names
```

`--no-daemon` still forks a single-use daemon (*"To honour the JVM settings for this build a single-use Daemon process will be forked"*) — normal, just slower. Leave the daemon on for iterative work.

### 7.3 Unit tests (JVM)

```powershell
.\gradlew.bat :app:testDebugUnitTest
.\gradlew.bat test                                            # all variants

.\gradlew.bat :app:testDebugUnitTest --tests '*.PreprocessTest'
.\gradlew.bat :app:testDebugUnitTest --tests '*.matchesDesktopGoldenTensor'
```

Reports: HTML `app\build\reports\tests\` · XML `app\build\test-results\`

### 7.4 Instrumented tests (on the S23+)

```powershell
.\gradlew.bat :app:connectedDebugAndroidTest
.\gradlew.bat cAT                                             # abbreviation

.\gradlew.bat :app:connectedDebugAndroidTest `
    -Pandroid.testInstrumentationRunnerArguments.class=com.example.tealeaf.OrientationTest

.\gradlew.bat :app:connectedCheck `
    -Pandroid.testInstrumentationRunnerArguments.size=medium

.\gradlew.bat :benchmark:connectedCheck                       # benchmark module
```

Reports: HTML `app\build\reports\androidTests\connected\` · XML `app\build\outputs\androidTest-results\connected\`

### 7.5 Device, install, capture

```powershell
$ADB = "C:/Users/<username>/AppData/Local/Android/Sdk/platform-tools/adb.exe"

& $ADB devices -l                          # must show "device", not "unauthorized"
& $ADB wait-for-device

& $ADB install -r -t -g app\build\outputs\apk\debug\app-debug.apk
#   -g auto-grants manifest permissions (CAMERA) — removes a manual dialog from every test cycle

& $ADB exec-out screencap -p > shot.png    # exec-out, NOT shell — raw binary
& $ADB shell screenrecord --size 1280x720 --bit-rate 6000000 /sdcard/demo.mp4
& $ADB pull /sdcard/demo.mp4

# Raw instrumentation, bypassing Gradle
& $ADB shell am instrument -w com.example.tealeaf.test/androidx.test.runner.AndroidJUnitRunner
& $ADB shell am instrument -w -e class com.example.tealeaf.OrientationTest `
    com.example.tealeaf.test/androidx.test.runner.AndroidJUnitRunner
```

Disambiguate multiple devices with `& $ADB -s <serial>` or `$env:ANDROID_SERIAL = "<serial>"`.

**No device was attached during research** (`adb devices -l` empty). The build path — wrapper bootstrap, `assembleDebug`, compileSdk 36, JDK 21 — was genuinely executed and succeeded. The instrumented-test task names were confirmed by listing the verification task group, but the connected-test runs, report paths, and adb screenshot commands are **documented-but-not-executed** in this environment.

### 7.6 The two verification steps to run before trusting anything

**(a) Orientation — verify visually, never by watching accuracy numbers.**

```kotlin
fun Bitmap.dumpForInspection(ctx: Context, name: String) {
    if (!BuildConfig.DEBUG) return
    File(ctx.getExternalFilesDir(null), name).outputStream().use {
        compress(Bitmap.CompressFormat.PNG, 100, it)
    }
}
```

Dump the **exact 224×224 bitmap handed to the interpreter**, once per device orientation — portrait, landscape-left, landscape-right, upside-down:

```powershell
& $ADB pull /sdcard/Android/data/com.example.tealeaf/files/ "D:\tea\pulled"
```

Any orientation bug is visible in two seconds. Accuracy numbers will not tell you.

**(b) Preprocessing — assert against a golden tensor.** The instrumented test described in §2.3. This is the only thing that settles whether the Android input tensor matches the desktop pipeline. Without it, "matches desktop" is an assertion.

---

## APPENDIX — Open items, ranked by how much they could hurt

1. **Desktop resize method is unknown.** Determine whether the training pipeline used PIL (antialiased) or `tf.image.resize` (not antialiased by default), then close it with the golden-tensor test. **Highest risk in the brief** — it degrades accuracy silently.
2. **`adb pull` from `Android/data` unconfirmed on this physical device.** The AOSP argument is strong and two-legged, but Samsung One UI was never tested. `adb shell ls -la /sdcard/Android/data/<pkg>/files/` closes it in 30 seconds.
3. **Thermal API may be unimplemented on this unit.** `getCurrentThermalStatus()` returning 0 is not evidence of no throttling. Run `thermalApiSupported()` before writing any throttling claim.
4. **`CompiledModel.CpuOptions` Kotlin parameter names not decoded.** Types are `(Integer, Integer, String)`. If you need Path B, verify names against the AAR. Pass `null` for the XNNPACK flags integer — valid constants are undocumented and will not be guessed here.
5. **Interpreter-vs-CompiledModel longevity is genuinely unresolved.** The LiteRT README and the Android docs contradict each other. Treat Path A as medium-term.
6. **Compose versions entirely unverified.** Recommendation is to avoid Compose; if mandated, verify the BOM against Google Maven before committing.
7. **`androidx.core:core-ktx:1.13.1` and `espresso-core:3.7.0`** were not cross-checked against Google Maven `group-index.xml` this cycle — the two softest coordinates in §1.4.
8. **CLAIM 2024 item numbers unverifiable** (RSNA returns 403). Items 14–18 are quoted from CLAIM 2020. Obtain the 2024 PDF before citing item numbers.
9. **GPU delegatability of this specific graph is unknown.** The GPU accelerator ships in 2.1.6 and `Accelerator.GPU` is the API, but whether a graph with fused normalization is fully delegatable was never tested — partial delegation causes partition thrash that can make GPU *slower*. Measure Path A first; a 224×224×3 → 7-class model on an 8 Gen 2 should run in single-digit milliseconds on CPU+XNNPACK, and GPU setup cost likely exceeds the saving.
10. **Croissant-RAI property names may have changed** in the 1.1 release (Feb 2026). Re-check before writing an exporter.
11. **Consent and data-protection specifics for smallholder field data collection are out of scope here.** Croissant-RAI has no explicit consent property; CrowdWorkSheets asks about withdrawal but prescribes no mechanism. Get local legal/ethics review — the `consentVersion` field in §3.3 is a hook, not compliance.
