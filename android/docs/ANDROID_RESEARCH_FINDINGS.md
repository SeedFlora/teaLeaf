# Android Technical Research — Verified Findings

Each section records what official documentation actually states, the concrete
decision taken, and the silent-failure modes found.


---

## litert-runtime

### Findings

## 1. The coordinate HAS moved, but the Java package has NOT

`org.tensorflow:tensorflow-lite` is frozen. Maven Central `maven-metadata.xml` shows `<release>2.17.0</release>` with `<lastUpdated>20250109190757</lastUpdated>` — no release in ~18 months. It still resolves; it is simply not maintained.

The live coordinate is `com.google.ai.edge.litert:*`.

**Critical correction to a widespread misreading of the migration guide:** the migration page says migration is "only a package name update." That means the *Maven* package. I unzipped both AARs and ran `javap` — the **Java/Kotlin package is still `org.tensorflow.lite`** in every version, V1 and V2 alike. `com.google.ai.edge.litert:litert:2.1.6` ships `org/tensorflow/lite/Interpreter.class`. Your `import org.tensorflow.lite.Interpreter` lines do not change. Only the Gradle line changes.

## 2. Artifacts are on Google Maven, NOT Maven Central

The official dev-tools page says the AAR is "hosted at MavenCentral". **That is wrong.** Verified:
- `https://repo1.maven.org/maven2/com/google/ai/edge/litert/litert/maven-metadata.xml` → empty/404 for all five artifacts I tried.
- `https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/maven-metadata.xml` → full metadata.

You need `google()` in `repositories`. `mavenCentral()` alone will fail to resolve.

## 3. Two divergent version lines (the central fact)

Verified from `maven-metadata.xml` on Google Maven:

| Artifact | Latest release |
|---|---|
| `litert` | **2.1.6** |
| `litert-api` | **2.1.6** |
| `litert-gpu` | **1.4.2** (V1 only) |
| `litert-gpu-api` | **1.4.2** (V1 only) |
| `litert-support` | **1.4.2** (V1 only) |
| `litert-metadata` | **1.4.2** (V1 only) |

`litert` jumped 1.4.2 → 2.0.0-alpha → 2.1.6; the GPU/support/metadata siblings **never followed**. Docs state 2.1.6 was released 2026-07-07 and that minSdk is 23.

Docs state plainly: *"GPU acceleration with the Interpreter API is exclusive to LiteRT Maven V1 packages. If you are using LiteRT Maven V2, you must transition to the CompiledModel API for GPU inference."*

## 4. V2 physically deleted the delegate machinery — verified by javap, not inferred

`Interpreter$Options` in **1.4.2**:
`setNumThreads`, `setUseXNNPACK`, `setUseNNAPI`, `setAllowFp16PrecisionForFp32`, `addDelegate(Delegate)`, `addDelegateFactory(DelegateFactory)`, `setCancellable`, `setRuntime`, `setAllowBufferHandleOutput`

`Interpreter$Options` in **2.1.6** — the complete list:
`setUseXNNPACK(boolean)`, `setNumThreads(int)`, `setCancellable(boolean)`, `setRuntime(TfLiteRuntime)`

Classes present in `litert-api:1.4.2` and **absent from `litert-api:2.1.6`**: `org.tensorflow.lite.Delegate`, `org.tensorflow.lite.DelegateFactory`, `org.tensorflow.lite.nnapi.NnApiDelegate`, `org.tensorflow.lite.acceleration.ValidatedAccelerationConfig`. And `org/tensorflow/lite/nnapi/NnApiDelegateImpl.class` is in `litert:1.4.2` but gone from `litert:2.1.6`.

So on V2 the `Delegate` *interface itself* no longer exists. `addDelegate` is not deprecated — it is uncompilable. In V2 the Interpreter is CPU/XNNPACK and nothing else.

## 5. NNAPI is a triple dead end for your device

- Official NDK migration guide (last updated 2026-03-06) states NNAPI was *"introduced in Android 8.1 ... and deprecated in Android 15."* Recommended paths: TFLite in Google Play Services, AICore, and optionally the GPU delegate.
- Your SM-S916B does not declare `android.hardware.neuralnetworks`, so even on V1 `CompatibilityList`/NNAPI init would fail or silently fall back.
- On V2 the class does not exist at all.

Do not spend any effort on NNAPI. The `nnapi_native.current_feature_level=7` properties on your device are leftover platform properties and are not a usable signal.

## 6. Asset compression — the premise is outdated

I verified this rather than trusting the docs. I extracted `com/android/builder/packaging/PackagingUtils.class` from `com.android.tools.build:builder` and dumped its constant pool. The default no-compress extension list in **both AGP 8.13.1 and AGP 9.3.0** begins:

`.tflite, .so, .dex, .jpg, .jpeg, .png, .gif, .opus, ...`

`.tflite` is the *first* entry in `DEFAULT_NO_COMPRESS_FILE_NAMES` / `DEFAULT_AAPT_NO_COMPRESS_EXTENSIONS`. **You do not need a `noCompress` block** on any modern AGP. This has been true since AGP 4.1. Adding one is harmless but is cargo-cult. The real risk is the *filename*, not the setting (see pitfalls).

## 7. GPU on V2 is bundled

`litert-2.1.6.aar` ships `jni/{arm64-v8a,armeabi-v7a,x86_64}/libLiteRt.so` **and `libLiteRtClGlAccelerator.so`**. Docs confirm: *"For Kotlin users, the GPU accelerator is built-in and does not require additional steps."* No extra GPU artifact on V2 — which is why `litert-gpu` stopped at 1.4.2.

V1 by contrast ships `libtensorflowlite_jni.so`, and `litert-gpu:1.4.2` separately ships `libtensorflowlite_gpu_jni.so` + `org.tensorflow.lite.gpu.GpuDelegate` / `CompatibilityList` / `GpuDelegateFactory`.

### Decisions

## Recommendation for your case

Your model is a plain 224x224x3 float32 → 7-class softmax classifier. That is small. **Use V2 (`litert:2.1.6`) with the Interpreter API on CPU+XNNPACK.** On a Snapdragon 8 Gen 2 this will run in single-digit milliseconds; GPU delegate setup cost would likely exceed the inference saving for a model this size. This also keeps you on the maintained line and requires zero delegate code.

Only if you measure and need GPU: stay on V2 and use `CompiledModel` with `Accelerator.GPU`. Do **not** drop back to the V1 1.4.2 line — it is a dead branch.

### Gradle (settings.gradle.kts)
```kotlin
dependencyResolutionManagement {
    repositories {
        google()          // REQUIRED — litert is NOT on Maven Central
        mavenCentral()
    }
}
```

### Gradle (app/build.gradle.kts)
```kotlin
android {
    compileSdk = 36
    defaultConfig {
        minSdk = 24      // litert 2.x requires >= 23
        targetSdk = 36
    }
    // NO noCompress block needed — AGP already excludes .tflite
}

dependencies {
    implementation("com.google.ai.edge.litert:litert:2.1.6")
    // pulls com.google.ai.edge.litert:litert-api:2.1.6 transitively
}
```

Put the model at `app/src/main/assets/model.tflite`.

### Thread count
- **Interpreter (V2):** `Interpreter.Options().setNumThreads(4)`
- **CompiledModel (V2):** `CompiledModel.CpuOptions(numThreads = 4, ...)`

Use 4, not 8. The 8 Gen 2 is 1+4+3 heterogeneous; oversubscribing onto little cores costs more in sync than it gains.

### Delegate switches (V2)
- **XNNPACK:** `setUseXNNPACK(true)` — on by default for float models; call it explicitly to be sure.
- **GPU:** not reachable from Interpreter. Use `CompiledModel.Options(Accelerator.GPU)`.
- **NNAPI:** does not exist. No API, no coordinate.

### Runtime accelerator probe
`Environment.create().getAvailableAccelerators()` returns `Set<Accelerator>` — use this instead of guessing, and instead of the removed `CompatibilityList`.

### Exact verified V2 API surface
```
org.tensorflow.lite.Interpreter(File | ByteBuffer [, Interpreter.Options])
  .run(Object input, Object output)
  .allocateTensors() / .getInputTensor(int) / .getOutputTensor(int)
  .getLastNativeInferenceDurationNanoseconds()
  .close()
Interpreter.Options: setNumThreads(int), setUseXNNPACK(boolean),
                     setCancellable(boolean), setRuntime(TfLiteRuntime)

com.google.ai.edge.litert.CompiledModel
  static create(AssetManager, String)
  static create(AssetManager, String, Options)
  static create(AssetManager, String, Options, Environment)
  static create(String filePath [, Options [, Environment]])
  createInputBuffers() / createOutputBuffers() : List<TensorBuffer>
  run(List<TensorBuffer> in, List<TensorBuffer> out)
CompiledModel.Options(vararg Accelerator) ; .Options.CPU
  var cpuOptions: CpuOptions ; var gpuOptions: GpuOptions ; var qualcommOptions
CompiledModel.CpuOptions(numThreads: Int?, xnnPackFlags: Int?, xnnPackWeightCachePath: String?)
Accelerator: NONE | CPU | GPU | NPU
TensorBuffer: writeFloat(FloatArray) / readFloat(): FloatArray (also Int/Int8/Long/Boolean)
Environment.create() ; .getAvailableAccelerators(): Set<Accelerator>
```
`CompiledModel`, `Environment`, `TensorBuffer` all extend `JniHandle implements AutoCloseable` → usable with Kotlin `.use { }`.

### Build commands (no cmdline-tools / no gradle on PATH)
```
cd d:\tea\tea-leaf-android-feia
.\gradlew.bat :app:assembleDebug
.\gradlew.bat :app:installDebug
```

### Pitfalls

**1. `mavenCentral()` only → build fails to resolve litert.**
Symptom: `Could not find com.google.ai.edge.litert:litert:2.1.6`. Cause: artifacts live on Google Maven despite the official page claiming MavenCentral. Fix: add `google()`.

**2. Mixing `litert:2.1.6` with `litert-gpu:1.4.2`.**
Symptom: `NoClassDefFoundError: org/tensorflow/lite/Delegate` at runtime, or duplicate-class errors at merge time. Cause: `litert-gpu:1.4.2` needs the `Delegate`/`DelegateFactory` interfaces that V2 deleted, and `litert:1.4.2`'s POM pins `litert-api:[1.4.2]` as a *hard range* — Gradle will either conflict or silently give you a `litert-api` that lacks classes the other AAR needs. Never straddle the two lines.

**3. Copy-pasting a pre-2025 tutorial's `addDelegate(GpuDelegate())` onto 2.1.6.**
Symptom: compile error `unresolved reference: addDelegate`. This reads as "wrong import" but is actually "the method and its parameter type no longer exist." Same for `setUseNNAPI` and `setAllowFp16PrecisionForFp32`.

**4. `afd.length` instead of `afd.declaredLength` when mmapping.**
Symptom: `IllegalArgumentException` or a garbage/oversized mapping — for an uncompressed asset `length` can report to the end of the APK. Always `startOffset` + `declaredLength`.

**5. Renaming the model to a non-`.tflite` extension.**
This is the *real* version of the "compressed assets" problem. AGP's no-compress list is keyed on the literal extension `.tflite`. Name it `model.lite`, `model.bin`, or `model.tflite.enc` and it gets deflated; `assets.openFd()` then throws `FileNotFoundException: This file can not be opened as a file descriptor; it is probably compressed`. Fix: keep the `.tflite` extension, or add `androidResources { noCompress += "yourext" }`.

**6. Forgetting `ByteOrder.nativeOrder()` on the input ByteBuffer.**
Symptom: no crash, no error — just wildly wrong softmax outputs, often near-uniform or pinned to one class. Java defaults to big-endian; the native runtime reads little-endian. This is the single most common silent-wrong-answer bug.

**7. Assuming NNAPI would help on this device.**
Symptom: delegate creation returns null / `IllegalArgumentException: Internal error: Error applying delegate`, or silent CPU fallback that looks like the delegate "worked." Your device lacks `android.hardware.neuralnetworks` entirely. Ignore the `nnapi_native.current_feature_level=7` property — it is not a capability signal.

**8. x86 emulator images will not run litert 2.1.6.**
Verified: `litert-2.1.6.aar` ships `jni/` for `arm64-v8a`, `armeabi-v7a`, `x86_64` only — the 32-bit `x86` ABI that 1.4.2 shipped was dropped. Symptom on an x86 AVD: `UnsatisfiedLinkError: couldn't find "libLiteRt.so"`. Use an x86_64 or arm64 image. (Your physical S23+ is arm64-v8a, so device testing is unaffected.)

**9. Not calling `.close()`.**
`CompiledModel`, `Environment`, and `TensorBuffer` all hold native handles via `JniHandle`. Leaking them leaks off-heap memory that the GC will not reclaim in time. Use Kotlin `.use { }`.

**10. Trusting the `/edge/litert/android/development` page.**
It still recommends `implementation 'com.google.ai.edge.litert:+'` — an unqualified version wildcard with no artifact name. That resolves nondeterministically and will one day silently jump you across the V1→V2 breaking boundary. Always pin `litert:2.1.6`.

### Reference code

// ============================================================
// PATH A (RECOMMENDED for your 224x224x3 -> 7-class model):
// V2 Interpreter, CPU + XNNPACK. Package is STILL org.tensorflow.lite.
// ============================================================
import android.content.Context
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/** Memory-maps the asset. Requires the asset be stored uncompressed —
 *  AGP already guarantees this for the ".tflite" extension. */
private fun loadModel(context: Context, assetName: String): ByteBuffer {
    context.assets.openFd(assetName).use { afd ->
        java.io.FileInputStream(afd.fileDescriptor).use { fis ->
            return fis.channel.map(
                FileChannel.MapMode.READ_ONLY,
                afd.startOffset,
                afd.declaredLength      // NOT afd.length — that is the whole APK
            )
        }
    }
}

class Classifier(context: Context) : AutoCloseable {
    private val interpreter = Interpreter(
        loadModel(context, "model.tflite"),
        Interpreter.Options().apply {
            setNumThreads(4)        // 8 Gen 2: 4 big cores, not all 8
            setUseXNNPACK(true)
            // addDelegate(...)   <-- DOES NOT COMPILE on 2.1.6; Delegate class removed
            // setUseNNAPI(...)   <-- DOES NOT COMPILE on 2.1.6; removed
        }
    )

    // Model takes RAW [0,255] floats; normalization is baked into the graph.
    private val input = ByteBuffer
        .allocateDirect(1 * 224 * 224 * 3 * 4)
        .order(ByteOrder.nativeOrder())          // MUST be nativeOrder

    private val output = Array(1) { FloatArray(7) }

    fun classify(pixels: IntArray): FloatArray {   // pixels = 224*224 ARGB_8888
        input.rewind()
        for (p in pixels) {
            input.putFloat(((p shr 16) and 0xFF).toFloat())  // R, raw 0..255
            input.putFloat(((p shr 8)  and 0xFF).toFloat())  // G
            input.putFloat((p          and 0xFF).toFloat())  // B
        }
        input.rewind()
        interpreter.run(input, output)
        return output[0]
    }

    override fun close() = interpreter.close()
}


// ============================================================
// PATH B: V2 CompiledModel with GPU + graceful CPU fallback.
// Only adopt this if Path A is measurably too slow.
// ============================================================
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment

fun buildCompiled(context: Context): CompiledModel {
    val env = Environment.create()
    val useGpu = Accelerator.GPU in env.getAvailableAccelerators()

    val options =
        if (useGpu) CompiledModel.Options(Accelerator.GPU)
        else CompiledModel.Options(Accelerator.CPU).apply {
            cpuOptions = CompiledModel.CpuOptions(numThreads = 4)
        }

    // Reads straight from AssetManager — no manual mmap needed.
    return CompiledModel.create(context.assets, "model.tflite", options, env)
}

fun runCompiled(model: CompiledModel, raw: FloatArray /* 224*224*3 */): FloatArray {
    val inputs  = model.createInputBuffers()
    val outputs = model.createOutputBuffers()
    inputs[0].writeFloat(raw)
    model.run(inputs, outputs)
    return outputs[0].readFloat()   // 7 softmax probabilities
}

### Not confirmed

**Could not confirm:**

1. **No formal EOL date for the V1 (1.4.x) line or for `org.tensorflow:tensorflow-lite`.** I found no page stating a removal date or a support-window end. The evidence for "V1 is dead" is circumstantial but strong: `litert-gpu`/`litert-support`/`litert-metadata` frozen at 1.4.2 while `litert` advanced to 2.1.6, plus tensorflow-lite's last Maven Central publish being 2025-01-09. That is my inference, not a documented statement.

2. **Whether the V2 Interpreter path is itself slated for removal.** The GitHub README's "AI Coding Directives" say *"The legacy Interpreter API is strictly deprecated for new features"* and *"MUST USE: The Compiled Model API"* — but that text is aimed at C++ (`tflite::Interpreter`, `InterpreterBuilder`) and at AI coding agents. The Android docs simultaneously describe the Java Interpreter as "maintained for backward compatibility." These two framings are in tension and I could not reconcile them from official sources. If Path A's longevity matters to you, treat it as a medium-term choice, not a permanent one.

3. **GPU delegate correctness for your specific model.** I verified the GPU accelerator *ships* in 2.1.6 and that `Accelerator.GPU` is the API. I did **not** verify that your particular graph (with normalization fused in) is fully GPU-delegatable — ops that fall back to CPU cause partition thrash that can make GPU *slower*. This needs measurement on the actual device; I had no way to run it.

4. **The exact `xnnPackFlags` integer values** for `CompiledModel.CpuOptions`. The parameter exists and is `Integer?`, but I found no documentation of the valid flag constants and will not guess them. Pass `null` unless you find an authoritative list.

5. **minSdk for 2.1.6 read from the AAR manifest.** The manifests are binary AXML and I could not decode them without `aapt2` wired up. The "minSdk 23" figure is from the docs page, not from the artifact. I recommended `minSdk = 24` which is safely above it either way.

6. **The 2.1.6 release date.** The docs page states 2026-07-07; Google Maven's `<lastUpdated>` for the `litert` artifact is `20260701221338` (2026-07-01). These disagree by six days — likely metadata-refresh vs. announcement timing. Immaterial, but I did not resolve it.

7. **I did not verify the Play Services route end-to-end.** `com.google.android.gms:play-services-tflite-java:16.5.0` / `-gpu:16.5.0` exist on Google Maven (confirmed), and the NNAPI migration guide recommends them, but I did not inspect those AARs or confirm their API shape. If you ever want an updatable runtime with a smaller APK, that path needs its own investigation.

### Sources

- https://developers.google.com/edge/litert/android
- https://developers.google.com/edge/litert/migration
- https://developers.google.com/edge/litert/android/development
- https://developers.google.com/edge/litert/next/android_kotlin
- https://developers.google.com/edge/litert/next/gpu
- https://developers.google.com/edge/litert/android/gpu
- https://developer.android.com/ndk/guides/neuralnetworks/migration-guide
- https://github.com/google-ai-edge/LiteRT
- https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/maven-metadata.xml
- https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert-gpu/maven-metadata.xml
- https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert-api/maven-metadata.xml
- https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/2.1.6/litert-2.1.6.pom
- https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/1.4.2/litert-1.4.2.pom
- https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/2.1.6/litert-2.1.6.aar (unzipped + javap)
- https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/1.4.2/litert-1.4.2.aar (unzipped + javap)
- https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert-api/2.1.6/litert-api-2.1.6.aar (unzipped + javap)
- https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert-api/1.4.2/litert-api-1.4.2.aar (unzipped + javap)
- https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert-gpu/1.4.2/litert-gpu-1.4.2.aar (unzipped + javap)
- https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert-gpu-api/1.4.2/litert-gpu-api-1.4.2.aar (unzipped + javap)
- https://repo1.maven.org/maven2/org/tensorflow/tensorflow-lite/maven-metadata.xml
- https://dl.google.com/dl/android/maven2/com/android/tools/build/gradle/maven-metadata.xml
- https://dl.google.com/dl/android/maven2/com/android/tools/build/builder/9.3.0/builder-9.3.0.jar (PackagingUtils.class constant pool)
- local gradle cache: com.android.tools.build/builder/8.13.1 builder-8.13.1.jar (PackagingUtils.class constant pool)
- https://dl.google.com/dl/android/maven2/com/google/android/gms/play-services-tflite-java/maven-metadata.xml

---

## camerax-orientation

### Findings

## 1. Current CameraX versions (fetched from the release-notes page, 2026-07-20)

From https://developer.android.com/jetpack/androidx/releases/camera:
- **Stable: 1.6.1** (released **May 06, 2026**)
- **RC: none**, **Beta: none**
- **Alpha: 1.7.0-alpha02** (July 01, 2026)

The page lists these artifacts under a single shared version: `camera-core`, `camera-camera2`, `camera-lifecycle`, `camera-view`, `camera-extensions`, `camera-video`, `camera-compose`.

Related: `androidx.exifinterface:exifinterface` latest is **1.4.2** (per https://developer.android.com/jetpack/androidx/releases/exifinterface).

Recent changes worth flagging:
- **CameraX 1.6.0 migrated to "CameraPipe"**, a unified camera stack shared with the Pixel Camera team. This is a large under-the-hood rewrite shipped only ~8 months ago; it is the single biggest "recently changed" risk in this area. Pin to 1.6.1 (not 1.6.0) — the .1 exists because of post-CameraPipe fixes.
- CameraX 1.5 added `OUTPUT_IMAGE_FORMAT_NV21` for ImageAnalysis and expanded `ImageProxy.toBitmap()`.
- A release note records a workaround for **incorrect JPEG image metadata on Samsung S10e/S10+** so that `ImageProxy.toBitmap()` returns correct Bitmaps. Directly relevant precedent: Samsung JPEG metadata has needed device-specific quirk handling in CameraX. Your target is a Samsung SM-S916B.
- A release note records a fix for "images not correctly rotated when output image rotation is enabled and the initial relative rotation is 0 degrees" — i.e. `setOutputImageRotationEnabled` has had rotation bugs.

## 2. ImageCapture vs ImageAnalysis for a one-shot classifier

Both are viable; they differ in what you get back.

**ImageCapture** — two `takePicture` overloads (per https://developer.android.com/media/camera/camerax/take-photo):
- `takePicture(OutputFileOptions, Executor, OnImageSavedCallback)` — writes to disk
- `takePicture(Executor, OnImageCapturedCallback)` — in-memory `ImageProxy`

From the `ImageCapture.java` source javadoc for `OnImageCapturedCallback.onCaptureSuccess`:
> "The image is of format ImageFormat#JPEG or ImageFormat#JPEG_R, queryable via ImageProxy#getFormat()."
> "The application is responsible for calling ImageProxy#close() to close the image."
> "The value in image.getImageInfo().getRotationDegrees() describes the magnitude of clockwise rotation, which if applied to the image will make it match the currently configured target rotation."

**ImageAnalysis** (https://developer.android.com/media/camera/camerax/analyze) delivers `YUV_420_888` by default, or `RGBA_8888`/`NV21` via `setOutputImageFormat`.

For a single-shot 224x224 classifier, **ImageCapture + `takePicture(Executor, OnImageCapturedCallback)` is the right choice**: you get one JPEG ImageProxy, no file I/O, no EXIF re-parsing, and no per-frame budget. Use ImageAnalysis only if you want continuous live classification.

## 3. setTargetRotation — what it actually does

From https://developer.android.com/media/camera/camerax/orientation-rotation:
- Set via `imageCapture.targetRotation = rotation` / `imageAnalysis.targetRotation = rotation`, values are `Surface.ROTATION_0/_90/_180/_270`.
- "By default, the use cases set their target rotation to match the display's rotation."

From `ImageCapture.Builder.setTargetRotation` javadoc (source):
> "This will affect the EXIF rotation metadata in images saved by takePicture calls and the ImageInfo#getRotationDegrees() value of the ImageProxy returned by OnImageCapturedCallback."

**Critical:** `setTargetRotation` never rotates pixels for ImageCapture. It only changes *metadata* — the EXIF tag on disk, and the `rotationDegrees` number on the in-memory ImageProxy.

The docs give two ways to keep it current: an `OrientationEventListener` (tracks physical device orientation, works even when the activity is locked to portrait) and a `DisplayManager.DisplayListener` (catches 180-degree flips and multi-window where the activity isn't recreated).

Note the deliberate inversion in the doc's `OrientationEventListener` sample — sensor degrees `45..134` map to `Surface.ROTATION_270`, `225..314` map to `Surface.ROTATION_90`. Display rotation is counter-clockwise, target rotation is clockwise. Copy the mapping verbatim; do not re-derive it.

## 4. EXIF: is rotation applied to the file?

From https://developer.android.com/media/camera/camerax/transform-output, verbatim:
> "For the `ImageCapture` use case, the crop rect buffer is applied before saving to disk and the rotation is saved in the Exif data. There is no additional action needed from the app."

So: **crop is baked into pixels, rotation is NOT.** The saved JPEG's pixel buffer is in sensor orientation; only the EXIF `Orientation` tag records the needed rotation. "No additional action needed from the app" is true for a gallery viewer (which honors EXIF) and **false for an ML pipeline** (a raw decode does not).

The orientation doc gives the read-back paths: `Exif.createFromFile(file).rotation`, `Exif.createFromInputStream(...)`, etc.

## 5. The root cause of the "rotated 90 degrees" bug — confirmed in source

I read `camera/camera-core/src/main/java/androidx/camera/core/internal/utils/ImageUtil.java` on androidx-main. `ImageProxy.toBitmap()` (declared in `ImageProxy.java` as `default @NonNull Bitmap toBitmap() { return createBitmapFromImageProxy(this); }`) dispatches:

```java
switch (imageProxy.getFormat()) {
  case ImageFormat.YUV_420_888: return ImageProcessingUtil.convertYUVToBitmap(imageProxy);
  case ImageFormat.JPEG:
  case ImageFormat.JPEG_R:      return createBitmapFromJpegImage(imageProxy);
  case PixelFormat.RGBA_8888:   return createBitmapFromRgbaImage(imageProxy);
```

The JPEG branch decodes with **`BitmapFactory.decodeByteArray(bytes, 0, bytes.length, null)`**. `BitmapFactory` does **not** apply EXIF orientation. `toBitmap()` applies **neither rotation nor crop**.

That is the whole bug. `takePicture` -> `toBitmap()` -> feed to TFLite gives you a sensor-oriented bitmap. On a phone held upright in portrait with a typical sensor orientation of 90, the model sees a landscape-rotated image. It never shows up in desktop tests because your test JPEGs were either already upright or were loaded by a tool that honored EXIF.

Supported formats for `toBitmap()`: "YUV_420_888, JPEG or RGBA_8888". It throws `IllegalArgumentException` for invalid formats and `UnsupportedOperationException` if conversion fails. `PRIVATE` is not supported.

## 6. ImageAnalysis path details

From the analyze guide, verbatim on RGBA output:
> "CameraX internally converts images from YUV to RGBA color space and packs image bits into the `ByteBuffer` of the ImageProxy's first plane (the other two planes are not used)"

with byte order `[0]=alpha, [1]=red, [2]=green, [3]=blue`.

Constants confirmed from `ImageAnalysis.java` source: `OUTPUT_IMAGE_FORMAT_YUV_420_888 = 1`, `OUTPUT_IMAGE_FORMAT_RGBA_8888 = 2`, `STRATEGY_KEEP_ONLY_LATEST = 0`, `STRATEGY_BLOCK_PRODUCER = 1`.

`setOutputImageFormat` javadoc warns:
> "Requesting OUTPUT_IMAGE_FORMAT_RGBA_8888 or OUTPUT_IMAGE_FORMAT_NV21 will have extra overhead because format conversion takes time."

`setOutputImageRotationEnabled` signature and javadoc from source:
```java
@RequiresApi(23)
public @NonNull Builder setOutputImageRotationEnabled(boolean outputImageRotationEnabled)
```
> "Once this is enabled, user doesn't need to handle the rotation, the output image will be a rotated ImageProxy and ImageInfo#getRotationDegrees() will return 0."

This is the one API in CameraX that will rotate pixels for you — and it exists **only on ImageAnalysis, not on ImageCapture**.

Lifecycle rule, verbatim:
> "Applications can use the wrapped Media.Image inside ImageProxy directly. Just do not call Media.Image.close() on the wrapped image as this would break the image sharing mechanism inside CameraX; instead, use ImageProxy.close()"

### Decisions

**Gradle (libs.versions.toml or direct):**
```
androidx.camera:camera-core:1.6.1
androidx.camera:camera-camera2:1.6.1
androidx.camera:camera-lifecycle:1.6.1
androidx.camera:camera-view:1.6.1
androidx.exifinterface:exifinterface:1.4.2   // only if you go the file-based route
```
Skip `camera-extensions`, `camera-video`, `camera-compose` unless you need them — they pull weight you don't use for a classifier.

**Manifest:** `<uses-permission android:name="android.permission.CAMERA" />` plus `<uses-feature android:name="android.hardware.camera.any" android:required="false" />`. You do NOT need `WRITE_EXTERNAL_STORAGE` for the in-memory path — that is a further reason to prefer it.

**Architecture decision — use ImageCapture, in-memory, and rotate the bitmap yourself:**
1. Build `ImageCapture` with `.setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)` — a classifier wants quality, not shutter latency, and the user is taking one deliberate photo.
2. Call `takePicture(executor, OnImageCapturedCallback)` — NOT the file overload. No file I/O, no EXIF round-trip, no storage permission, no temp-file cleanup.
3. In `onCaptureSuccess`, read `imageProxy.imageInfo.rotationDegrees` FIRST, then `imageProxy.toBitmap()`, then apply the rotation with a `Matrix`, then close in a `finally`.
4. Keep an `OrientationEventListener` updating `imageCapture.targetRotation` so `rotationDegrees` is correct when the activity is locked to portrait.

**Do NOT** rely on `setTargetRotation` alone to give you upright pixels from ImageCapture — it only writes metadata.

**Do NOT** use `ImageAnalysis.setOutputImageRotationEnabled(true)` as your primary fix here. It works only on ImageAnalysis, it is API 23+ (fine for your API 36 device), and it has a recorded rotation bug in the "initial relative rotation is 0" case. For a single-shot classifier the explicit Matrix rotation is fewer moving parts and is verifiable in a unit test.

**Preprocessing to match your model** (224x224x3 float32, RAW [0,255], normalization already in the graph):
- Rotate first, then center-crop to square, then scale to 224x224. Cropping before rotating crops the wrong axis.
- Write floats as raw 0..255 — do NOT divide by 255 and do NOT subtract a mean. Your graph does it.
- Channel order R,G,B. Drop alpha.
- `ByteBuffer.allocateDirect(1*224*224*3*4).order(ByteOrder.nativeOrder())`.

**Build command (no Android Studio, no gradle on PATH):**
```
cd d:\tea && .\gradlew.bat :app:assembleDebug
```
Set `compileSdk = 36`, `targetSdk = 36`, `minSdk = 24`. JDK 21 is on the host; set `kotlin.jvmToolchain(21)` or `compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }` — AGP does not require the toolchain to match the daemon JDK.

**Verification step that actually catches the bug:** have the debug build write the exact 224x224 bitmap you hand to TFLite out to `context.getExternalFilesDir(null)` as a PNG, then `adb pull` it and look at it. Do this once per device orientation (portrait, landscape-left, landscape-right, upside-down). Any orientation bug is visible in two seconds. Do not verify by watching accuracy numbers.

### Pitfalls

**1. `ImageProxy.toBitmap()` silently drops rotation.** Symptom: model is ~random or systematically confuses classes; works perfectly on the same images fed from your desktop test harness. Confirmed in `ImageUtil.createBitmapFromImageProxy` — the JPEG branch is `BitmapFactory.decodeByteArray`, which never applies EXIF. This is the headline bug and it produces no exception, no log, no warning.

**2. Reading `rotationDegrees` after calling `close()`.** Symptom: `IllegalStateException`, or worse, a garbage rotation value. Capture `image.imageInfo.rotationDegrees` into a local as the first statement in `onCaptureSuccess`.

**3. Double rotation.** Symptom: image is upside-down (180 off) instead of 90 off — deceptively close to correct, so it survives a quick glance. Happens if you save the ImageProxy JPEG bytes to a file, reload it with something that DOES honor EXIF (`ImageDecoder`, Glide, Coil, `ExifInterface`-aware loaders), and then also apply `rotationDegrees`. Pick exactly one source of truth: either in-memory + `rotationDegrees`, or file + EXIF. Never both.

**4. Activity locked to portrait in the manifest kills `targetRotation` updates.** Symptom: works in portrait, wrong by 90 in landscape. If `screenOrientation="portrait"`, the display rotation never changes and the default target rotation is frozen — the physical device rotation is invisible to CameraX. The `OrientationEventListener` in the sketch is what fixes this, and it is precisely why the doc recommends it over `DisplayListener`.

**5. Cropping before rotating.** Symptom: subject is off-center or sliced, accuracy degrades subtly rather than catastrophically — the hardest failure mode to notice. Rotate first, then center-crop.

**6. `Bitmap.Config.HARDWARE` bitmaps throw on `getPixels()`.** Symptom: `IllegalStateException: unable to getPixels(), pixel access is not supported on Config#HARDWARE bitmaps`. `BitmapFactory.decodeByteArray` gives you software ARGB_8888 so the sketch is safe, but if you ever route through `ImageDecoder` you must call `setAllocator(ALLOCATOR_SOFTWARE)`.

**7. Normalizing twice.** Symptom: uniform/near-uniform softmax, every image gets the same class. Your graph embeds normalization and takes RAW [0,255]. Dividing by 255.0f in Kotlin as well is the single most common TFLite integration error and it looks exactly like a broken model rather than a broken pipeline.

**8. Front camera mirroring.** `ImageCapture.Metadata.isReversedHorizontal` javadoc: "The reflection is meant to be applied to the upright image (after rotation to the target orientation). When saving the image to file, it is combined with the rotation degrees, to generate the corresponding EXIF orientation value." Note that sentence says "when saving the image to file" — I could not confirm the in-memory ImageProxy path exposes mirroring at all. If you ever enable the front camera, verify mirroring separately with the PNG dump. Rear camera only: not an issue.

**9. Not closing the ImageProxy.** Symptom: capture works 2-3 times then the camera silently stops delivering frames. Use `try/finally`. The docs are explicit that you must call `ImageProxy.close()` and must NOT call `Media.Image.close()`.

**10. CameraPipe (1.6.0) is young.** If you hit inexplicable device-specific behavior on the S23+, test against 1.5.1 as a control before assuming your code is wrong. There is precedent for Samsung JPEG-metadata quirks needing CameraX-side workarounds (the S10e/S10+ `toBitmap()` fix in the release notes).

**11. Unrelated but adjacent to your NNAPI note:** `android.hardware.neuralnetworks` being absent from the feature list is consistent with NNAPI being deprecated for new work — do not build the TFLite delegate selection around NNAPI on this device. That is outside this research area; flagging only because it will bite the same integration.

### Reference code

// build.gradle.kts (app)
dependencies {
    val cameraxVersion = "1.6.1"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")
}

// ---- Setup -------------------------------------------------------------

private lateinit var imageCapture: ImageCapture

private val orientationEventListener by lazy {
    object : OrientationEventListener(this) {
        override fun onOrientationChanged(orientation: Int) {
            if (orientation == ORIENTATION_UNKNOWN) return
            // Mapping copied verbatim from the CameraX orientation-rotation doc.
            // The 90/270 inversion is intentional: display rotation is CCW,
            // target rotation is CW. Do not "fix" this.
            val rotation = when (orientation) {
                in 45 until 135  -> Surface.ROTATION_270
                in 135 until 225 -> Surface.ROTATION_180
                in 225 until 315 -> Surface.ROTATION_90
                else             -> Surface.ROTATION_0
            }
            imageCapture.targetRotation = rotation
        }
    }
}

override fun onStart() { super.onStart(); orientationEventListener.enable() }
override fun onStop()  { super.onStop();  orientationEventListener.disable() }

private fun bindCamera(previewView: PreviewView) {
    val future = ProcessCameraProvider.getInstance(this)
    future.addListener({
        val provider = future.get()
        val preview = Preview.Builder().build().also {
            it.surfaceProvider = previewView.surfaceProvider
        }
        imageCapture = ImageCapture.Builder()
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .setTargetRotation(previewView.display.rotation)
            .build()

        provider.unbindAll()
        provider.bindToLifecycle(
            this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
    }, ContextCompat.getMainExecutor(this))
}

// ---- Capture -----------------------------------------------------------

fun capture(onReady: (Bitmap) -> Unit) {
    imageCapture.takePicture(
        ContextCompat.getMainExecutor(this),
        object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                try {
                    // Read rotation BEFORE toBitmap(); toBitmap() discards it.
                    val degrees = image.imageInfo.rotationDegrees
                    // JPEG branch of toBitmap() uses BitmapFactory.decodeByteArray,
                    // which ignores EXIF. Bitmap is in SENSOR orientation here.
                    val raw = image.toBitmap()
                    onReady(raw.rotated(degrees))
                } finally {
                    image.close()   // required; never call Media.Image.close()
                }
            }
            override fun onError(exc: ImageCaptureException) { /* handle */ }
        }
    )
}

private fun Bitmap.rotated(degrees: Int): Bitmap {
    if (degrees == 0) return this
    val m = Matrix().apply { postRotate(degrees.toFloat()) }
    val out = Bitmap.createBitmap(this, 0, 0, width, height, m, true)
    if (out !== this) recycle()
    return out
}

// ---- Preprocess: rotate -> center-crop -> scale -> raw [0,255] floats ----

private const val SIDE = 224

fun Bitmap.toModelInput(): ByteBuffer {
    // 1. center-crop to square (AFTER rotation, or you crop the wrong axis)
    val edge = minOf(width, height)
    val square = Bitmap.createBitmap(this, (width - edge) / 2, (height - edge) / 2, edge, edge)
    // 2. scale to 224x224, ARGB_8888 software config (not HARDWARE)
    val scaled = square.scale(SIDE, SIDE)   // androidx.core.graphics.scale

    val pixels = IntArray(SIDE * SIDE)
    scaled.getPixels(pixels, 0, SIDE, 0, 0, SIDE, SIDE)

    val buf = ByteBuffer.allocateDirect(SIDE * SIDE * 3 * 4)
        .order(ByteOrder.nativeOrder())
    for (p in pixels) {
        // RAW 0..255 — the .tflite graph does its own normalization.
        buf.putFloat(((p shr 16) and 0xFF).toFloat())  // R
        buf.putFloat(((p shr 8)  and 0xFF).toFloat())  // G
        buf.putFloat((p          and 0xFF).toFloat())  // B
    }
    buf.rewind()
    return buf
}

// ---- Debug hook: prove the orientation is right, don't infer it ----------

fun Bitmap.dumpForInspection(ctx: Context, name: String) {
    if (!BuildConfig.DEBUG) return
    File(ctx.getExternalFilesDir(null), name).outputStream().use {
        compress(Bitmap.CompressFormat.PNG, 100, it)
    }
    // adb pull /sdcard/Android/data/<pkg>/files/<name>
}

### Not confirmed

**Method signatures came from androidx-main (tip of tree), not the 1.6.1 release tag.** The Android Developers reference pages for `ImageProxy`, `ImageAnalysis.Builder`, and `ImageCapture.Builder` are JS-rendered and returned only navigation chrome through WebFetch — I tried both the Java and Kotlin reference URLs and neither yielded method bodies. So I fell back to `android.googlesource.com/platform/frameworks/support/+/androidx-main/...`, which is the development branch. The `toBitmap()` / `setOutputImageRotationEnabled` / `setOutputImageFormat` signatures and the `ImageUtil.createBitmapFromImageProxy` switch statement are quoted accurately from what I read, but I did NOT verify they are byte-identical in the 1.6.1 artifact. Verify against the actual AAR before treating any signature as final.

**Things I could not confirm:**

- **`OUTPUT_IMAGE_FORMAT_NV21`'s int value.** I saw the constant referenced in the `setOutputImageFormat` javadoc and in a release-notes search result, but never saw its declaration. I have not stated a value for it.

- **`ImageProcessingUtil.convertYUVToBitmap` behavior.** The YUV branch of `toBitmap()` delegates here and I did not read that file. Whether it applies the crop rect, and its exact color-conversion coefficients, are unverified. My claim that `toBitmap()` applies neither rotation nor crop is solidly established for the JPEG path (which is the path the recommended design uses) and is only *inferred* for the YUV path.

- **The `Exif` class in the orientation-rotation doc.** The doc's samples call `Exif.createFromFile(file)` / `Exif.createFromInputStream(...)` without an import. I believe this is `androidx.camera.core.internal.utils.Exif`, whose package name signals it is internal/`@RestrictTo` and not part of the public API — but I did not verify its visibility annotation. This matters: if you follow that doc sample literally you may not be able to resolve the symbol from app code. Use `androidx.exifinterface.media.ExifInterface` (artifact 1.4.2) instead. Treat the doc sample as pseudocode.

- **CameraX 1.6.1's actual minSdk.** I asserted `minSdk = 24` in the recommendations based on general AndroidX baselines, not on anything I read. CameraX historically shipped minSdk 21 and has been raising it. Confirm before committing the build file — this is a guess, not a finding.

- **Whether `camera-compose:1.6.1` exists as a published artifact.** It appears in the release notes' dependency block, which lists all artifacts at the shared version, but I did not check Maven. Irrelevant if you skip it as recommended.

- **Front-camera mirroring on the in-memory `OnImageCapturedCallback` path.** The `isReversedHorizontal` javadoc I quoted describes only the save-to-file behavior. I found no statement about whether or how mirroring is signaled for in-memory capture. Unresolved.

- **Nothing device-specific to the SM-S916B / Galaxy S23+.** I found the Samsung S10e/S10+ `toBitmap()` metadata workaround in the release notes and cite it as *precedent for Samsung JPEG quirks*, but I found no S23-specific CameraX issue. Do not read my mention of it as evidence that your device is affected.

**Deprecation check:** I saw no deprecation notice on `setTargetRotation`, `takePicture`, `ImageProxy.toBitmap`, or the ImageAnalysis output-format APIs. `setTargetResolution` appears in the analyze guide alongside `setResolutionSelector`; `setTargetResolution` is widely understood to be superseded by `ResolutionSelector`, but I did not fetch a page stating that explicitly, so I have not marked it deprecated.

### Sources

- https://developer.android.com/media/camera/camerax/orientation-rotation
- https://developer.android.com/media/camera/camerax/take-photo
- https://developer.android.com/media/camera/camerax/take-photo/options
- https://developer.android.com/media/camera/camerax/analyze
- https://developer.android.com/media/camera/camerax/transform-output
- https://developer.android.com/jetpack/androidx/releases/camera
- https://developer.android.com/jetpack/androidx/releases/exifinterface
- https://android.googlesource.com/platform/frameworks/support/+/androidx-main/camera/camera-core/src/main/java/androidx/camera/core/ImageAnalysis.java
- https://android.googlesource.com/platform/frameworks/support/+/androidx-main/camera/camera-core/src/main/java/androidx/camera/core/ImageProxy.java
- https://android.googlesource.com/platform/frameworks/support/+/androidx-main/camera/camera-core/src/main/java/androidx/camera/core/ImageCapture.java
- https://android.googlesource.com/platform/frameworks/support/+/androidx-main/camera/camera-core/src/main/java/androidx/camera/core/internal/utils/ImageUtil.java

---

## storage-adb-pull

### Findings

## 1. The directory: `getExternalFilesDir()` — writable with ZERO permissions

**Docs state** (https://developer.android.com/training/data-storage/app-specific):
> "On Android 4.4 (API level 19) or higher, your app doesn't need to request any storage-related permissions to access app-specific directories within external storage."

> "If none of the pre-defined sub-directory names suit your files, you can instead pass `null` into `getExternalFilesDir()`. This returns the root app-specific directory within external storage."

> "When the user uninstalls your app, the files saved in app-specific storage are removed."

Signature is `getExternalFilesDir(String type)` (nullable `type`). Resolves at runtime to `/storage/emulated/0/Android/data/<pkg>/files`, which is the same inode reachable as `/sdcard/Android/data/<pkg>/files` (`/sdcard` → `/storage/self/primary` → the FUSE mount). NOTE: the app-specific docs page does **not** print that literal path string — I am inferring it from the `PATTERN_OWNED_PATH` regex in AOSP FuseDaemon.cpp, `^/storage/[^/]+/(?:[0-9]+/)?Android/(?:data|obb)/([^/]+)(/?.*)?`, plus universally observed behavior. Treat the path as a runtime fact to log, not a constant to hardcode (see pitfalls).

Requires **no** `<uses-permission>` at all. Not `WRITE_EXTERNAL_STORAGE`, not `READ_MEDIA_IMAGES`, not `MANAGE_EXTERNAL_STORAGE`, not `requestLegacyExternalStorage`. This satisfies "no broad storage permissions."

Also satisfies "don't touch the user's personal files": this tree is outside the MediaStore-indexed collections, so the captured JPEGs will not surface in Gallery/Photos and nothing the user owns is read or written.

## 2. Do Android 11+ / Android 16 restrictions block adb here? NO — and I can show exactly why

The Android 11 restriction is real but is scoped to **apps**, not to the adb shell.

**Docs state** (https://developer.android.com/about/versions/11/privacy/storage):
> "On Android 11, apps can no longer access files in *any* other app's dedicated, app-specific directory within external storage."

> "You can no longer use the `ACTION_OPEN_DOCUMENT_TREE` or the `ACTION_OPEN_DOCUMENT` intent action to request that the user select individual files from the following directories: The `Android/data/` directory and all subdirectories. The `Android/obb/` directory and all subdirectories."

That page contains **no mention of adb or shell access**. Neither restriction applies to adb. Two independent AOSP mechanisms confirm the shell is exempt:

**(a) DAC layer — adbd deliberately keeps the external-storage app-data GID.**
`packages/modules/adb/daemon/main.cpp` sets adbd's supplementary groups (inherited by every `adb shell` child) via `minijail_set_supplementary_gids()`:
> `gid_t groups[] = {AID_ADB, AID_LOG, AID_INPUT, AID_INET, AID_NET_BT, AID_NET_BT_ADMIN, AID_SDCARD_R, AID_SDCARD_RW, AID_NET_BW_STATS, AID_READPROC, AID_UHID, AID_EXT_DATA_RW, AID_EXT_OBB_RW, AID_READTRACEFS}`

And `system/core/libcutils/include/private/android_filesystem_config.h`:
> `AID_EXT_DATA_RW (1078)` — "GID for app-private data directories on external storage"
> `AID_EXT_OBB_RW (1079)` — "GID for OBB directories on external storage"
> `AID_SHELL (2000)` — "adb and debug shell user"

`AID_EXT_DATA_RW` is precisely the GID that gates `Android/data`. adbd holds it.

**(b) FUSE layer — the access check exempts every uid below 10000.**
`packages/providers/MediaProvider/jni/FuseDaemon.cpp`, `is_app_accessible_path()`:
```c
static bool is_app_accessible_path(struct fuse* fuse, const string& path, uid_t uid) {
  MediaProviderWrapper* mp = fuse->mp;
  if (uid < AID_APP_START || uid == MY_UID) {
    return true;
  }
  ...
  if (std::regex_match(path, match, PATTERN_OWNED_PATH)) {
    ...
    if (!mp->isUidAllowedAccessToDataOrObbPath(uid, path)) {
      PLOG(WARNING) << "Invalid other package file access from " << uid << "(: " << path;
      return false;
    }
  }
  return true;
}
```
`AID_APP_START` is `10000 /* first app user */`. Shell is uid **2000 < 10000**, so the function short-circuits to `return true` before the `Android/data` regex check is ever reached.

This also explains the widely-reported "file managers can't open Android/data anymore" complaints (including on Samsung One UI): those are installed apps at uid ≥ 10000, which DO hit the regex branch. Your adb session does not. Same kernel, different uid, different outcome.

**Android 16 (API 36) specifically:** I fetched both official pages. https://developer.android.com/about/versions/16/behavior-changes-all and https://developer.android.com/about/versions/16/behavior-changes-16 contain **no** storage / external-storage / scoped-storage / `Android/data` / adb changes whatsoever. Android 16's only storage-adjacent items are `MediaStore#getVersion()` being per-app and photo-picker pre-selection — neither touches app-specific dirs. Nothing in Android 12–16 has changed this path.

## 3. `run-as` on internal storage: debug builds ONLY

`system/core/run-as/run-as.cpp` gates on exactly two things:
> `"only 'shell' or 'root' users can run this program"`
```c
if (!info.debuggable) {
    error(1, 0, "package not debuggable: %s", pkgname);
}
```
Confirmed: "The debuggable flag is the sole application-level security gate after UID validation." There is no `profileable` fallback for run-as.

`info.debuggable` reflects `android:debuggable` in the **merged manifest of the installed APK**. AGP injects it for the `debug` build type and omits it for `release`. So:
- `assembleDebug` APK → `run-as` works → `/data/user/0/<pkg>/files` reachable.
- `assembleRelease` APK → `run-as` fails with `package not debuggable: <pkg>` → internal storage is **completely unreachable** without root.

Corroborated by https://developer.android.com/studio/debug/device-file-explorer:
> "Not all files on a hardware device are visible in the Device Explorer. For example, in the `data/data/` directory, entries corresponding to apps on the device that are not debuggable can't be expanded in the Device Explorer."

**Conclusion: do not put the research log in internal storage.** It would silently become unpullable the moment you build a release APK. `getExternalFilesDir()` works identically for debug and release — that's the property you want.

## 4. adb pull

https://developer.android.com/tools/adb:
> "To copy a file or directory and its sub-directories *from* the device, do the following: `adb pull remote local`"

Directory pulls are recursive — one command gets the whole log tree.

### Decisions

**Write target (decided):** `context.getExternalFilesDir(null)` → subdir `research/`.
- On-device path: `/sdcard/Android/data/<applicationId>/files/research/`
- JSONL: `.../research/log.jsonl`
- JPEGs: `.../research/frames/<name>.jpg`

**Manifest:** add NOTHING. Zero `<uses-permission>` lines for this feature. No `requestLegacyExternalStorage`, no `MANAGE_EXTERNAL_STORAGE`.

**Gradle:** no dependency, no coordinate, no `minSdk` bump needed. Works on the project's existing `compileSdk 36` / `targetSdk 36` setup unchanged. Do **not** rely on `debuggable` for the log path.

**API calls:**
- `android.content.Context.getExternalFilesDir(null)` → `java.io.File?` (null-check it)
- `android.os.Environment.getExternalStorageState()` → compare to `Environment.MEDIA_MOUNTED`
- `java.io.FileOutputStream.getFD().sync()` to force durability before pulling
- `android.graphics.Bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)`

**Exact adb commands (Windows 11 host, Gradle-wrapper workflow, PowerShell):**

Discover the real path rather than assuming (works on any device/user profile):
```
adb shell echo /sdcard/Android/data/<applicationId>/files/research
adb shell ls -la /sdcard/Android/data/<applicationId>/files/research
```

Pull the whole log tree (this is the primary command):
```
adb pull /sdcard/Android/data/<applicationId>/files/research "D:\tea\pulled"
```
Result: `D:\tea\pulled\research\log.jsonl` and `D:\tea\pulled\research\frames\*.jpg`.

Preserve device mtimes:
```
adb pull -a /sdcard/Android/data/<applicationId>/files/research "D:\tea\pulled"
```

Pull just the JSONL:
```
adb pull /sdcard/Android/data/<applicationId>/files/research/log.jsonl "D:\tea\pulled\log.jsonl"
```

Tail the log live without pulling:
```
adb shell cat /sdcard/Android/data/<applicationId>/files/research/log.jsonl
```

Clear between runs (no uninstall needed):
```
adb shell rm -rf /sdcard/Android/data/<applicationId>/files/research
```

**Debug-only fallback for internal storage** (only if you ever also need `/data/user/0/<pkg>/`). `exec-out` is mandatory here on Windows — see pitfalls:
```
adb exec-out run-as <applicationId> tar -c -C files research > "D:\tea\pulled\research.tar"
```
Single file variant:
```
adb exec-out run-as <applicationId> cat files/research/log.jsonl > "D:\tea\pulled\log.jsonl"
```
Both fail on a release APK with `package not debuggable: <applicationId>`.

### Pitfalls

**1. Colons in filenames — will bite you specifically, host is Windows 11.** ISO-8601 timestamps (`2026-07-20T14:33:07Z`) are legal on Android's FUSE volume but `:` is illegal in Windows filenames. *Symptom:* `adb pull` reports success for the directory but individual files are silently missing, or you get `couldn't create file: Invalid argument`. *Fix:* name files `20260720T143307Z.jpg` (basic-format, no colons). Restrict filenames to `[A-Za-z0-9._-]`.

**2. Buffered writes not flushed → truncated JSONL.** `adb pull` reads whatever is on disk right now. A `BufferedWriter` holding the last N records means the pulled file ends mid-line and your JSONL parser throws on the final record. *Fix:* `writer.flush()` then `fos.fd.sync()` after each record (or each batch), and always parse defensively by skipping a trailing partial line.

**3. Do NOT hardcode `/sdcard/Android/data/...` in the app.** The Android 11 docs state: "On Android 11 (API level 30) and higher, apps cannot create their own app-specific directory on external storage." Constructing the path via string concatenation and calling `mkdirs()` on it. *Symptom:* `mkdirs()` returns `false`, every write throws `ENOENT`/`EACCES`. *Fix:* always go through `getExternalFilesDir(null)` and `mkdirs()` only on the subdirectory beneath it.

**4. `getExternalFilesDir()` returns nullable.** Returns `null` if external storage is unmounted/unavailable. *Symptom:* Kotlin NPE or a silent no-op log. *Fix:* null-check and fall back to `context.filesDir` (accepting that the fallback is only pullable from a debug build).

**5. `adb shell run-as ... cat` corrupts binary on Windows.** The `shell` transport allocates a PTY that performs LF→CRLF translation. *Symptom:* pulled JPEGs are subtly larger than on-device and won't decode; tar archives fail with "unexpected EOF". *Fix:* use `adb exec-out`, never `adb shell`, for any binary redirect. (This is why the docs' own screencap example says: "use 'exec-out' instead of 'shell' to get raw data".)

**6. Release build silently kills `run-as`.** If you prototype the pull workflow against `assembleDebug` using `run-as` on internal storage, it works — then breaks with `package not debuggable` the first time you ship `assembleRelease`. *Fix:* this is the whole argument for choosing `getExternalFilesDir()`; it behaves identically in both build types.

**7. Uninstall / "Clear storage" wipes the log.** Docs: "When the user uninstalls your app, the files saved in app-specific storage are removed." *Symptom:* reinstalling the APK to fix a bug destroys the research data you hadn't pulled yet. Note `adb install -r` does NOT wipe it, but `adb uninstall` does. *Fix:* pull before every uninstall; consider a pull step in your iteration script.

**8. Trailing-slash ambiguity in `adb pull` destinations.** `adb pull <remotedir> <localdir>` creates `<localdir>\<basename>`; if `<localdir>` doesn't exist yet, adb instead renames the tree to `<localdir>`. *Symptom:* files land one directory level higher or lower than your script expects. *Fix:* create the destination directory first and always assert on the final resolved path.

**9. Third-party file-manager reports are not evidence about adb.** Much of the web (XDA threads, One UI 6.1 complaints) says "Android/data is blocked, you need root." That is true for apps at uid ≥ 10000 and false for adb at uid 2000, per `is_app_accessible_path()`. Don't let those reports talk you into requesting `MANAGE_EXTERNAL_STORAGE` — that permission is both unnecessary here and a Play policy problem.

**10. Files won't appear in Gallery.** Not a bug — `Android/data` is not MediaStore-indexed. This is desirable for your privacy constraint, but means you cannot preview the JPEGs on-device via Photos; use `adb pull` or an in-app viewer.

### Reference code

// AndroidManifest.xml — deliberately NO storage permissions of any kind.
// <manifest ...>
//     <!-- (CAMERA only if you actually capture; nothing storage-related) -->
//     <application ...>

// ResearchLog.kt
package com.example.tea

import android.content.Context
import android.graphics.Bitmap
import android.os.Environment
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ResearchLog(context: Context) {

    // Windows-safe: basic-format timestamp, no ':' characters.
    private val stamp = SimpleDateFormat("yyyyMMdd'T'HHmmss'Z'", Locale.US)

    // /sdcard/Android/data/<applicationId>/files/research
    // Never build this path by hand — Android 11+ forbids creating it yourself.
    private val root: File = File(
        requireNotNull(context.getExternalFilesDir(null)) {
            "External storage unavailable: ${Environment.getExternalStorageState()}"
        },
        "research"
    ).apply { mkdirs() }

    private val framesDir = File(root, "frames").apply { mkdirs() }
    private val jsonl = File(root, "log.jsonl")

    /** Log this once at startup so the pull command can be copied verbatim. */
    fun logPullCommand(pkg: String) {
        android.util.Log.i("ResearchLog", "adb pull ${'$'}{root.absolutePath} .")
        android.util.Log.i("ResearchLog", "or: adb pull /sdcard/Android/data/${'$'}pkg/files/research .")
    }

    /** Appends one JSON object per line and fsyncs, so `adb pull` never sees a torn tail. */
    fun append(fields: Map<String, Any?>) {
        val line = JSONObject(fields).toString() + "\n"
        FileOutputStream(jsonl, /* append = */ true).use { fos ->
            fos.write(line.toByteArray(Charsets.UTF_8))
            fos.flush()
            fos.fd.sync()          // durability: pull-safe immediately
        }
    }

    /** Returns the frame filename to reference from the JSONL record. */
    fun saveFrame(bitmap: Bitmap, label: String): String {
        val safeLabel = label.replace(Regex("[^A-Za-z0-9._-]"), "_")
        val name = "${'$'}{stamp.format(Date())}_${'$'}safeLabel.jpg"
        FileOutputStream(File(framesDir, name)).use { fos ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 90, fos)
            fos.flush()
            fos.fd.sync()
        }
        return name
    }
}

// Usage at an inference site:
// val frame = log.saveFrame(inputBitmap, "cls")
// log.append(mapOf(
//     "ts"     to System.currentTimeMillis(),
//     "frame"  to frame,
//     "probs"  to org.json.JSONArray(probs.map { it.toDouble() }),
//     "argmax" to probs.indices.maxBy { probs[it] },
//     "ms"     to elapsedMs
// ))

// ---- Pull from the Windows host (PowerShell) ----
// New-Item -ItemType Directory -Force "D:\tea\pulled"
// adb pull -a /sdcard/Android/data/com.example.tea/files/research "D:\tea\pulled"
// -> D:\tea\pulled\research\log.jsonl
// -> D:\tea\pulled\research\frames\20260720T143307Z_cls.jpg

// ---- Debug-build-only fallback for INTERNAL storage (fails on release) ----
// adb exec-out run-as com.example.tea tar -c -C files research > "D:\tea\pulled\research.tar"

### Not confirmed

**Not verified on the actual SM-S916B.** Every claim below the docs level comes from AOSP `main`-branch source, not from running `adb pull` against this Galaxy S23+ on One UI / Android 16. Samsung ships a modified MediaProvider/vold in principle. I rate the risk low because the mechanism that exempts shell (`uid < AID_APP_START`) is upstream and load-bearing for `adb`'s own test infrastructure, and because the One UI 6.1 lockdown reports I found describe *file-manager apps*, which is exactly the uid ≥ 10000 branch. **Recommended 30-second empirical check before you build anything around this:**
`adb shell touch /sdcard/Android/data/<pkg>/files/probe && adb pull /sdcard/Android/data/<pkg>/files/probe .`

**No official developer.android.com statement says "adb shell is exempt from the Android/data restriction."** I looked for one on the Android 11 storage page, the Android 16 behavior-change pages, the adb tools page, the Device File Explorer page, and source.android.com/docs/core/storage/scoped. None states it. The exemption is my **inference from AOSP source** (two independent mechanisms, quoted above) — a strong inference, but flagged as inference rather than documentation.

**Could not confirm the literal path string in official docs.** The app-specific storage page never prints `/storage/emulated/0/Android/data/<pkg>/files`. I derived it from the FuseDaemon regex plus common knowledge. This is why the code sketch logs the runtime `absolutePath` instead of trusting the constant.

**Did not verify the exact mode bits / st_gid on the `Android/data` node.** I confirmed adbd *holds* `AID_EXT_DATA_RW` and confirmed what that GID is *for*, but did not locate the code that stamps that gid onto the directory (it wasn't in FuseDaemon.cpp; likely in vold or MediaProvider's node layer). The DAC argument is therefore one link short of complete — however the FUSE-layer argument (`uid < AID_APP_START`) is complete and independently sufficient.

**Did not verify `getExternalFilesDir()` auto-creates the directory.** I did not fetch the `android.content.Context` reference page. The sketch calls `mkdirs()` unconditionally, which is correct either way.

**Did not verify AGP's exact `debuggable` injection mechanism** for the `debug` build type against current AGP release notes — stated from standard behavior. The *consequence* (release APK → `package not debuggable`) is solid, since it comes straight from run-as.cpp.

**source.android.com/docs/core/storage/scoped was a dead end** — fetched it, and it does not document mount views, uids, gids, or shell/adb behavior at all. Noting this so nobody re-fetches it.

### Sources

- https://developer.android.com/training/data-storage/app-specific
- https://developer.android.com/about/versions/11/privacy/storage
- https://developer.android.com/about/versions/16/behavior-changes-all
- https://developer.android.com/about/versions/16/behavior-changes-16
- https://developer.android.com/tools/adb
- https://developer.android.com/studio/debug/device-file-explorer
- https://android.googlesource.com/platform/packages/modules/adb/+/refs/heads/main/daemon/main.cpp
- https://android.googlesource.com/platform/system/core/+/refs/heads/main/libcutils/include/private/android_filesystem_config.h
- https://android.googlesource.com/platform/system/core/+/refs/heads/main/run-as/run-as.cpp
- https://android.googlesource.com/platform/packages/providers/MediaProvider/+/refs/heads/main/jni/FuseDaemon.cpp
- https://source.android.com/docs/core/storage/scoped

---

## measurement-methodology

### Findings

## 1. Clock choice — what the docs actually say

Fetched the SystemClock class overview (developer.android.com/reference/* renders as a JS shell to WebFetch and returned only navigation on 4 attempts; I used the MIT AOSP doc mirror, whose class-overview text is verbatim-stable). Quotes:

- `uptimeMillis()` — "counted in milliseconds since the system was booted. This clock stops when the system enters deep sleep ... but is not affected by clock scaling, idle, or other power saving mechanisms. **This is the basis for most interval timing such as Thread.sleep(millis), Object.wait(millis), and System.nanoTime().**"
- `elapsedRealtime()` / `elapsedRealtimeNanos()` — "return the time since the system was booted, **and include deep sleep**. This clock is **guaranteed to be monotonic**, and continues to tick even when the CPU is in power saving modes, so is the **recommend[ed] basis for general purpose interval timing**."
- `System.currentTimeMillis()` — wall clock, "may jump backwards or forwards unpredictably." Never use it for latency.

Inference I draw (labelled as inference, not doc text): both `System.nanoTime()` and `elapsedRealtimeNanos()` are monotonic and nanosecond-resolution. They differ only in whether suspend counts. A screen-on foreground inference loop cannot enter deep sleep, so for the *inner* per-inference timer the two are numerically equivalent, and `System.nanoTime()` is the conventional, portable choice (it is what JMH and the Jetpack Benchmark library are built on). `elapsedRealtimeNanos()` is the correct choice for the *outer* session timeline that you correlate against thermal samples, because it survives any suspension. The docs do NOT say nanoTime is inaccurate — they say it is based on the uptime clock.

## 2. Warm-up — the official defaults are far too low for a paper

LiteRT/TFLite `benchmark_model` defaults (from the tool README, verbatim): `warmup_runs` **default = 1** ("The number of warmup runs to do before starting the benchmark"), `num_runs` **default = 50** ("The number of runs. Increase this to reduce variance."). The LiteRT measurement page shows the tool reports four separate numbers: `Init`, `First inference`, `Warmup (avg)`, `Inference (avg)` — i.e. Google's own tool treats first-inference and steady-state as distinct quantities you must report separately.

Jetpack Microbenchmark "handles warmup, measures your code performance and allocation counts, and outputs benchmarking results" and the docs state warmup/iterations are **handled automatically; no manual configuration needed**. The `androidx.benchmark` Gradle plugin "fully compiles your microbenchmark APK by default" (AOT, equivalent to `CompilationMode.Full`), requiring Benchmark 1.3.0-beta01+ and AGP 8.4.0+; disable with `androidx.benchmark.forceaotcompilation=false`. Rationale quoted: "Complex microbenchmarks can take a long time to stabilize."

## 3. Isolating pure inference from camera + bitmap preprocessing

Microbenchmark's mechanism is explicit and is the pattern to copy even if you hand-roll:
- Kotlin: `benchmarkRule.measureRepeated { listToSort = runWithTimingDisabled { unsorted.copyOf() }; SortingAlgorithms.quickSort(listToSort) }`
- Java: `state.pauseTiming(); ...setup...; state.resumeTiming();`

Docs guidance: minimize work inside `measureRepeated`; assert once outside the loop, not per iteration; set `debuggable = false`.

For pipeline decomposition, `androidx.tracing` `Trace.beginSection()`/`endSection()` emits Perfetto slices; async slices take a cookie int to match begin/end across threads. Latest **stable androidx.tracing is 1.3.0** (2025-04-23); 2.0.0-alpha09 (2026-06-23) adds a low-overhead in-process API.

## 4. CPU frequency scaling and thermal — the non-root levers

Microbenchmark docs list three approaches in priority order:
1. **`gradlew lockClocks`** — "Requires **rooted Android device**", "Not supported on most devices". **Unavailable to you.**
2. **Sustained Performance Mode (default)** — via `Window.setSustainedPerformanceMode()`, "Enabled by default by the `testInstrumentationRunner` set by the Android Gradle plugin". AOSP performance-management page (verbatim): "the power hint **caps the maximum frequencies of the CPU and GPU at the highest sustainable levels**"; "lowering the MAX bar in CPU/GPU frequency will lower the frame rate, but this lower rate is preferred in this mode due to its sustainability." Available **Android 7.0+ (API 24)**. Guard with `PowerManager.isSustainedPerformanceModeSupported()`. This is the correct lever on a non-rooted S23+.
3. **Automatic thermal throttling detection** — fallback; "periodically detects CPU performance degradation" and "pauses execution to allow device cooling."

## 5. Thermal status programmatically (ADPF Thermal API page — fetched in full)

- `getCurrentThermalStatus()` → int, **API 29**. Constants and values: `THERMAL_STATUS_NONE`=0, `LIGHT`=1, `MODERATE`=2, `SEVERE`=3, `CRITICAL`=4, `EMERGENCY`=5, `SHUTDOWN`=6.
- `addThermalStatusListener(PowerManager.OnThermalStatusChangedListener)` / `removeThermalStatusListener(listener)`; callback is `onThermalStatusChanged(int status)`.
- `getThermalHeadroom(int forecastSeconds)` → float, **API 30 (Android 11)**. Range 0.0f–1.0f, where 0.0f = no throttling (`THERMAL_STATUS_NONE`) and 1.0f = heavy throttling (`THERMAL_STATUS_SEVERE`). Returns **NaN** if unsupported or called too frequently.
- `getThermalHeadroomThresholds()` listed as Android 15 DP1 (Preview) on that page.

**Critical caveat, verbatim from the page:** "Some devices return `THERMAL_STATUS_NONE` regardless of actual thermal state. Validate with `getThermalHeadroom()` instead." Your hardware probe reported `thermal status 0` — that reading is therefore **not by itself evidence the device is unthrottled**, and must be cross-checked against headroom before you assert it in a paper.

Documented heuristic: headroom > 1.0 → could be severely throttled; > 0.95 → moderately; > 0.85 → light. Support check: if the first headroom call returns 0 or NaN, the API is unsupported on that device.

## 6. Memory

**Confirmed API levels and units** (from an AOSP-doc mirror; "All results are in kB"):
- `getTotalPss()` — API 5, kB
- `getTotalPrivateDirty()` — API 5, kB
- `getTotalSharedDirty()` — API 5, kB
- `getMemoryStat(String)` — API 23, returns String or null

**`getMemoryStat` keys — confirmed by two independent sources** (AOSP `Debug.java` on android.googlesource.com, and the reference page via search): exactly nine — `summary.java-heap`, `summary.native-heap`, `summary.code`, `summary.stack`, `summary.graphics`, `summary.private-other`, `summary.system`, `summary.total-pss`, `summary.total-swap`. Javadoc note: these are **approximate**; "individual allocations may not be immediately reflected"; and "memory statistics may be added or removed in a future API level."

**PSS is the officially blessed footprint metric.** memory-overview page, verbatim: "Android computes a value called the Proportional Set Size (PSS), which accounts for both dirty and clean pages that are shared with other processes—but only in an amount that's proportional to how many apps share that RAM. **This (PSS) total is what the system considers to be your physical memory footprint.**"

**dumpsys** (tools/dumpsys page, verbatim syntax): `adb shell dumpsys meminfo [-d] package_name|pid`; `-d` = "Prints more info related to Dalvik and ART memory usage"; `-h` lists flags. Example given: `adb shell dumpsys meminfo -d com.google.android.apps.maps`.

**`benchmark_model` peak-memory flags:** `--report_peak_memory_footprint` (bool, default false, "Whether to report the peak memory footprint by periodically checking") and `--memory_footprint_check_interval_ms` (int, default 50).

## 7. Energy on a non-rooted Galaxy S23+ — the honest answer is "you cannot make a per-app energy claim"

Three official statements, all fetched, that jointly close this off:

- **Power Profiler / ODPM**, verbatim: "Power Profiler reads power consumption data from the ODPM, which is **only available on Pixel 6 and subsequent Pixel devices** running Android 10 (API level 29) and higher." And: "**ODPM measures power consumption at the device level—not specific to any app.** ... you can expect noise in power consumption data based on how many apps are active." Devices without ODPM "can offer power consumption data through Coulomb counters and the battery gauge."
- **Macrobenchmark `PowerMetric`** (marked *Experimental*), verbatim: "**These metrics measure system-wide consumption, not the consumption on a per-app basis, and are limited to Pixel 6, Pixel 6 Pro, and later devices.**" Categories: `CPU`, `DISPLAY`, `GPU`, `GPS`, `MEMORY`, `MACHINE_LEARNING`, `NETWORK`, `UNCATEGORIZED`. Outputs `power<category>Uw` and `energy<category>Uws`. Also: "With some categories, like CPU, it might be difficult to separate work done by other processes from work done by your own app."
- **BatteryStats** is a *model*, not a measurement: it "doesn't track battery current draw directly, but instead collects timing information that can be used to approximate battery consumption by different components," estimating from OEM-supplied coefficients. The page explicitly lists what it does NOT show: "**Quantitative battery consumption amounts** — the chart only displays whether a component is active, not how much current it draws." And a standing warning: "**Battery Historian is no longer actively maintained**; if possible, consider using system tracing, the Macrobenchmark power metric, or the Power Profiler."

Your S23+ is not a Pixel 6+, so ODPM, Power Profiler, and `PowerMetric` energy/power values are all unavailable. What remains legitimate is the coulomb counter via `BatteryManager.getLongProperty()`.

## 8. NNAPI — matches your hardware probe

NNAPI Migration Guide, verbatim: "The Neural Networks API (NNAPI) is deprecated. It was introduced in Android 8.1 to provide a unified interface for hardware accelerated inference for on-device machine learning, and **deprecated in Android 15**." Replacement path: TensorFlow Lite in Google Play Services, plus optionally the TFLite GPU delegate. This corroborates the absent `android.hardware.neuralnetworks` feature on your device — do not benchmark or report an NNAPI path; report CPU/XNNPACK and GPU delegate only.

### Decisions

## Versions and coordinates (verified against the official releases pages, 2026-07-20)

**Jetpack Benchmark — latest stable 1.4.1 (released 2025-09-10); latest alpha 1.5.0-alpha07 (2026-07-01).** Use 1.4.1.

Top-level `build.gradle`:
```groovy
plugins { id 'androidx.benchmark' version '1.4.1' apply false }
```
Benchmark module `build.gradle`:
```groovy
plugins { id 'androidx.benchmark' }
android {
  defaultConfig {
    testInstrumentationRunner "androidx.benchmark.junit4.AndroidBenchmarkRunner"
  }
}
dependencies { androidTestImplementation "androidx.benchmark:benchmark-junit4:1.4.1" }
```

**androidx.tracing — stable 1.3.0** (`androidx.tracing:tracing:1.3.0`, `tracing-ktx:1.3.0`). 2.0.0-alpha09 exists; stay on 1.3.0 for a paper.

Command-line build (no Studio, no cmdline-tools): `.\gradlew.bat :benchmark:connectedCheck` — the wrapper is the only supported path in your environment. AGP must be 8.4.0+ for the plugin's default AOT compilation; JDK 21 Temurin is fine.

## Timing API decisions

- Inner per-inference timer: `System.nanoTime()`, delta in ns, converted to ms only at report time.
- Outer session/epoch timeline and thermal-sample timestamps: `SystemClock.elapsedRealtimeNanos()`.
- Never `System.currentTimeMillis()`.
- State in the paper which clock you used and that both are monotonic.

## Thermal APIs (all available on API 36)

- `PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);`
- `int s = pm.getCurrentThermalStatus();` — API 29, 0..6.
- `pm.addThermalStatusListener(listener)` / `pm.removeThermalStatusListener(listener)` with `PowerManager.OnThermalStatusChangedListener { onThermalStatusChanged(int) }`.
- `float h = pm.getThermalHeadroom(0);` — API 30. **Poll at most once per 10 s.**
- `pm.isSustainedPerformanceModeSupported()` then `getWindow().setSustainedPerformanceMode(true)` — API 24. Enable this for the whole measurement Activity.

## Memory APIs

- Own process: `Debug.MemoryInfo mi = new Debug.MemoryInfo(); Debug.getMemoryInfo(mi);` then `mi.getTotalPss()` (kB) and `mi.getMemoryStat("summary.total-pss")`.
- Cross-check from host: `adb shell dumpsys meminfo -d <your.package.name>`
- Peak: sample PSS on a dedicated thread at a fixed interval (e.g. 100 ms) and take the max; record the sampling interval in the paper.

## Energy: the only defensible route on this device

`BatteryManager.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)` → charge in **µAh**; `BATTERY_PROPERTY_CURRENT_NOW` → instantaneous current in **µA** (negative = discharging). `getLongProperty` returns `Long.MIN_VALUE` if the platform does not provide the property — check for that sentinel.

Protocol: device **unplugged** (charge counter is meaningless while charging), airplane mode, fixed screen brightness, screen-on. Run ≥10–30 minutes of continuous inference. Measure ΔµAh. Run a matched-duration idle baseline with the same screen state. Report `(Δ_inference − Δ_idle)` mAh per N inferences, with n≥5 repetitions and a dispersion measure. Convert to energy only if you multiply by nominal pack voltage and say so explicitly.

## Reference tooling for a cross-check against your in-app numbers

Prebuilt LiteRT benchmark APK (arm64):
`https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model.apk`

```
adb install -r -d -g android_aarch64_benchmark_model.apk
adb push model.tflite /data/local/tmp
adb shell am start -S \
  -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity \
  --es args '"--graph=/data/local/tmp/model.tflite --num_threads=4 --warmup_runs=50 --num_runs=500 --use_xnnpack=true --report_peak_memory_footprint=true"'
```
Output reports `Init`, `First inference`, `Warmup (avg)`, `Inference (avg)`.

Native binary variant:
```
adb push benchmark_model /data/local/tmp
adb shell chmod +x /data/local/tmp/benchmark_model
adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/model.tflite --num_threads=4
```

Verified `benchmark_model` flags and defaults: `num_runs` (int, 50), `warmup_runs` (int, 1), `max_secs` (float, 150.0), `run_delay` (float, -1.0), `num_threads` (int, -1), `use_gpu` (bool, false), `use_nnapi` (bool, false), `use_xnnpack` (bool, false), `enable_op_profiling` (bool, false), `profiling_output_csv_file` (string, "", deprecated), `report_peak_memory_footprint` (bool, false), `memory_footprint_check_interval_ms` (int, 50).

Do **not** pass `--use_nnapi=true` — NNAPI is deprecated as of Android 15 and your device does not declare `android.hardware.neuralnetworks`.

## Recommended reported protocol (my synthesis, not a doc quote)

Three separately-timed stages, never conflated: (A) camera frame acquisition → Bitmap; (B) Bitmap → 224×224×3 float32 ByteBuffer fill (raw [0,255], no normalization since it's in-graph); (C) `interpreter.run()` alone. Report C as "model inference latency" and A+B+C as "end-to-end pipeline latency". For C, drive the interpreter from one pre-filled static buffer with no camera running at all.

Warm-up 50, measure 500–1000. Report **median, p95, IQR or MAD** — not mean±SD, since latency distributions are right-skewed. Log `getCurrentThermalStatus()` per iteration and `getThermalHeadroom(0)` once per 10 s; discard or separately report any block where status ≠ 0 or headroom > 0.85.

### Pitfalls

**`getThermalHeadroom()` called more often than once per 10 seconds returns NaN.** Symptom: your thermal log is full of NaN and you conclude the device doesn't support the API. Documented explicitly on the ADPF page, along with "avoid calling from multiple threads."

**`getCurrentThermalStatus()` returning 0 may be a lie.** Verbatim doc caveat: "Some devices return THERMAL_STATUS_NONE regardless of actual thermal state." Your hardware probe already recorded status 0. Symptom: a paper claim of "no thermal throttling observed" that is actually an unimplemented HAL. Cross-validate with `getThermalHeadroom()`; if the first headroom call returns 0 or NaN, the whole thermal API is unsupported on that unit and you must say so rather than claim absence of throttling.

**`gradlew lockClocks` silently isn't an option.** It "requires a rooted Android device" and is "not supported on most devices." Symptom: you assume clocks are locked, but CPU frequency is actually floating and your variance is governor noise. On your non-rooted S23+ the real control is `Window.setSustainedPerformanceMode(true)`, and you must report that you used it — it *caps max CPU/GPU frequency*, so your absolute latencies will be higher than an unconstrained burst run. Both numbers are defensible; silently mixing them across conditions is not.

**Benchmarking a `debuggable = true` build.** Docs say set `debuggable = false`. Symptom: latencies inflated by a large, non-constant factor from JVMTI hooks and disabled optimizations, invalidating any cross-device comparison.

**Not AOT-compiling, or not warming up enough.** The `androidx.benchmark` plugin AOT-compiles by default precisely because "complex microbenchmarks can take a long time to stabilize." Symptom with a hand-rolled harness: a descending latency curve over the first tens-to-hundreds of iterations that you mistake for signal. `warmup_runs=1` (the `benchmark_model` default) is nowhere near enough — that default exists for quick triage, not publication.

**Sampling PSS inside the timed loop.** `Debug.getMemoryInfo()` walks `/proc/<pid>/smaps` and is expensive. Symptom: your measured inference latency inflates by milliseconds and the inflation scales with process memory size, i.e. you contaminate the exact number you're publishing. Sample on a separate thread or between conditions only.

**Treating `getMemoryStat` values as exact.** AOSP javadoc says they are approximate and "individual allocations may not be immediately reflected in the results," and that keys "may be added or removed in a future API level." Symptom: irreproducible memory numbers across Android versions.

**Claiming per-app energy from BatteryStats.** It "doesn't track battery current draw directly" — it multiplies component *on-time* by OEM-supplied coefficients from `power_profile.xml`. Symptom: a reviewer asks how the coefficients were calibrated and there is no answer, because the OEM never published them. This is the single most likely place for a rigorous reviewer to reject an energy claim.

**Claiming ODPM/`PowerMetric` numbers on a Galaxy S23+.** Both are limited to Pixel 6+ (and `PowerMetric` is additionally marked *Experimental* and system-wide). Symptom: the API returns nothing, or worse, you cite a Pixel-derived methodology in a paper whose measurements came from a Samsung device.

**Measuring the coulomb counter while plugged in.** `BATTERY_PROPERTY_CURRENT_NOW` is positive when charging. Symptom: negative or nonsensical "energy consumed." Also, `getLongProperty` returns `Long.MIN_VALUE` for unsupported properties — if you don't check that sentinel you will average `Long.MIN_VALUE` into your results and get absurd numbers.

**Using `System.currentTimeMillis()` anywhere in the timing path.** "may jump backwards or forwards unpredictably" — NTP sync mid-run produces negative or wildly inflated durations. Symptom: rare, unexplained outliers that survive your outlier filter because they look like plausible tail latency.

**Timing the first inference and calling it latency.** Google's own tool reports `First inference` as a separate line from `Inference (avg)` for exactly this reason — first call includes delegate setup and lazy allocation. Symptom: a headline latency number 5–50× the steady-state one.

**Enabling `--use_nnapi=true`.** NNAPI is deprecated as of Android 15, and your device does not declare `android.hardware.neuralnetworks` despite having `nnapi_native.current_feature_level=7` properties. Symptom: silent fallback to CPU, so you publish "NNAPI accelerated" numbers that are plain CPU numbers.

**Battery Historian.** Officially "no longer actively maintained." Symptom: you build a methodology section around a tool Google has stopped supporting.

### Reference code

// ─────────────────────────────────────────────────────────────────────────────
// MeasurementHarness.kt — only APIs confirmed above are used.
// minSdk must be >= 30 for getThermalHeadroom; target device is API 36.
// ─────────────────────────────────────────────────────────────────────────────

import android.content.Context
import android.os.BatteryManager
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock

// --- 1. Stabilize the device before measuring -------------------------------
// Call from your Activity.onCreate(). Caps max CPU/GPU freq at sustainable
// levels (AOSP: "the power hint caps the maximum frequencies of the CPU and
// GPU at the highest sustainable levels"). API 24+.
fun Activity.enableSustainedPerformance(): Boolean {
    val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
    val supported = pm.isSustainedPerformanceModeSupported
    if (supported) window.setSustainedPerformanceMode(true)
    return supported   // REPORT this boolean in the paper
}

// --- 2. Thermal sampling ----------------------------------------------------
class ThermalMonitor(context: Context) {
    private val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
    private var lastHeadroomNs = 0L
    private var cachedHeadroom = Float.NaN

    // API 29. Cheap - safe to call per iteration.
    fun status(): Int = pm.currentThermalStatus     // 0=NONE .. 6=SHUTDOWN

    // API 30. MUST NOT be called more than once per 10 s or it returns NaN.
    fun headroom(): Float {
        val now = SystemClock.elapsedRealtimeNanos()
        if (now - lastHeadroomNs >= 10_000_000_000L) {
            cachedHeadroom = pm.getThermalHeadroom(0)   // 0 = "right now"
            lastHeadroomNs = now
        }
        return cachedHeadroom
    }

    // One-shot support probe: doc says if first value is 0 or NaN, unsupported.
    fun thermalApiSupported(): Boolean {
        val v = pm.getThermalHeadroom(0)
        return !(v.isNaN() || v == 0f)
    }

    private val listener = PowerManager.OnThermalStatusChangedListener { s ->
        // s is one of PowerManager.THERMAL_STATUS_*
    }
    fun start() = pm.addThermalStatusListener(listener)
    fun stop()  = pm.removeThermalStatusListener(listener)
}

// --- 3. The measurement loop: pure inference only ---------------------------
// input is filled ONCE, outside the loop. No camera, no Bitmap, no resize.
// Normalization is in-graph, so the buffer holds raw [0,255] float32.
data class LatencySample(val nanos: Long, val thermalStatus: Int)

fun benchmarkPureInference(
    interpreter: org.tensorflow.lite.Interpreter,
    input: java.nio.ByteBuffer,            // 1*224*224*3*4 bytes, direct, nativeOrder
    output: Array<FloatArray>,             // [1][7]
    thermal: ThermalMonitor,
    warmup: Int = 50,
    measured: Int = 500
): List<LatencySample> {
    repeat(warmup) { input.rewind(); interpreter.run(input, output) }

    val samples = ArrayList<LatencySample>(measured)
    repeat(measured) {
        input.rewind()
        val t0 = System.nanoTime()          // monotonic; uptime-based
        interpreter.run(input, output)      // <-- THE ONLY THING TIMED
        val t1 = System.nanoTime()
        samples += LatencySample(t1 - t0, thermal.status())
    }
    return samples
}

// Report median / p95 / IQR, not mean +/- SD (right-skewed distribution).
fun percentile(sorted: LongArray, p: Double): Long =
    sorted[((sorted.size - 1) * p).toInt().coerceIn(0, sorted.size - 1)]

// --- 4. Peak memory: sampled OFF the measurement thread ---------------------
// Debug.getMemoryInfo() walks /proc/<pid>/smaps and is expensive - never call
// it inside the timed loop.
class PeakPssSampler(private val intervalMs: Long = 100) : Thread() {
    @Volatile var running = true
    @Volatile var peakKb = 0
    override fun run() {
        val mi = Debug.MemoryInfo()
        while (running) {
            Debug.getMemoryInfo(mi)
            peakKb = maxOf(peakKb, mi.totalPss)     // getTotalPss(), API 5, kB
            // or: mi.getMemoryStat("summary.total-pss")   // API 23, String kB
            sleep(intervalMs)
        }
    }
}

// --- 5. Energy: the ONLY defensible route on a non-Pixel, non-rooted device --
// Whole-device coulomb counter delta. NOT a per-app measurement - say so.
fun chargeCounterMicroAh(context: Context): Long? {
    val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
    val v = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
    return if (v == Long.MIN_VALUE) null else v     // MIN_VALUE = unsupported
}
// Protocol: unplugged, airplane mode, fixed brightness, >=10 min run,
// matched-duration idle baseline, subtract, repeat n>=5.

/* ── Host-side cross-checks ────────────────────────────────────────────────
adb shell dumpsys meminfo -d com.example.tealeaf
adb shell dumpsys batterystats --reset
adb shell dumpsys batterystats > batterystats.txt
adb shell am start -S -n org.tensorflow.lite.benchmark/.BenchmarkModelActivity \
  --es args '"--graph=/data/local/tmp/model.tflite --num_threads=4 \
              --warmup_runs=50 --num_runs=500 --use_xnnpack=true \
              --report_peak_memory_footprint=true"'
.\gradlew.bat :benchmark:connectedCheck
────────────────────────────────────────────────────────────────────────── */

### Not confirmed

**Could not confirm (developer.android.com `/reference/*` pages are JS-rendered and returned only navigation to WebFetch across ~5 attempts):**

1. **`Debug.MemoryInfo.getTotalRss()` API level.** It exists in AOSP `Debug.java` (confirmed via android.googlesource.com) but is **absent** from the older doc mirror I could read, and my targeted search did not confirm "API 31". Do not cite an API level for it. Use `getTotalPss()` (confirmed API 5) and `getMemoryStat("summary.total-pss")` (confirmed API 23) instead — both are confirmed and PSS is the officially-designated footprint metric anyway.

2. **`@UnsupportedAppUsage` on getTotalPss/getTotalRss.** The AOSP source fetch reported these annotations, but `getTotalPss()` is unambiguously public SDK per the doc mirror (API 5). I suspect the fetch conflated overloads or internal variants. I did **not** verify this and would not repeat the `@UnsupportedAppUsage` claim.

3. **`BatteryManager` property units.** `BATTERY_PROPERTY_CHARGE_COUNTER` = µAh and `BATTERY_PROPERTY_CURRENT_NOW` = µA, and `getLongProperty` returning `Long.MIN_VALUE` when unsupported, come from a **search-result summary quoting** the official reference page, not from a page I fetched myself. High confidence but one notch below the rest. I found **no** authoritative statement on `BATTERY_PROPERTY_ENERGY_COUNTER` units — do not cite it.

4. **`ActivityManager.getProcessMemoryInfo(int[])`.** I did not fetch its reference page. I have not verified whether it is rate-limited or restricted for other processes on API 36. The sketch avoids it and uses `Debug.getMemoryInfo()` for the own-process case, which is safe.

5. **Microbenchmark JSON output path.** The overview page states results go "to both the Android Studio console and a JSON file," but I did not confirm the on-device path or the `adb pull` command. Do not cite a path.

6. **`MemoryUsageMetric` in Macrobenchmark.** The macrobenchmark-metrics page I fetched documents only StartupTimingMetric, FrameTimingMetric, TraceSectionMetric, and PowerMetric. It did **not** list MemoryUsageMetric. It may exist as a `Metric` subclass; unverified.

7. **Doc/version inconsistency to be aware of:** the `microbenchmark-write` page returned `implementation("androidx.benchmark:benchmark-junit4:1.2.4")` while the authoritative releases page gives `androidTestImplementation "androidx.benchmark:benchmark-junit4:1.4.1"`. This is likely a stale WebFetch cache of the guide page. Trust the **releases page (1.4.1)**. Note the `implementation` vs `androidTestImplementation` distinction is real in one case: in a dedicated `com.android.test` benchmark module the benchmark code lives in the main sourceset. Verify against your actual module type.

8. **`getThermalHeadroomThresholds()`** was documented on the ADPF page as "Android 15 (DP1) — Preview". I did not confirm whether it shipped as stable API or what its final signature is on API 36. Do not use it without checking.

9. **`PowerMetric.deviceSupportsHighPrecisionTracking` / `deviceBatteryHasMinimumCharge`** appeared only in a search summary, not on the page I fetched. Since PowerMetric is Pixel-6+-only and therefore unusable on your S23+ anyway, this does not affect the recommendation.

10. **Whether Samsung's S23+ HAL genuinely implements the thermal status API.** This is empirically determinable on your hardware and is exactly what the `thermalApiSupported()` probe in the sketch is for. Run it before writing any "no throttling observed" sentence.

### Sources

- https://stuff.mit.edu/afs/sipb/project/android/docs/reference/android/os/SystemClock.html (AOSP doc mirror; developer.android.com/reference/android/os/SystemClock returned only navigation to fetch)
- https://developer.android.com/games/optimize/adpf/thermal
- https://source.android.com/docs/core/power/performance
- https://developer.android.com/topic/performance/benchmarking/microbenchmark-overview
- https://developer.android.com/topic/performance/benchmarking/microbenchmark-write
- https://developer.android.com/jetpack/androidx/releases/benchmark
- https://developer.android.com/jetpack/androidx/releases/tracing
- https://developer.android.com/topic/performance/benchmarking/macrobenchmark-metrics
- https://developer.android.com/studio/profile/power-profiler
- https://developer.android.com/topic/performance/power/setup-battery-historian
- https://developer.android.com/topic/performance/memory-overview
- https://developer.android.com/tools/dumpsys
- https://developer.android.com/studio/profile/investigate-ram (fetched; contained no dumpsys meminfo syntax - negative result)
- https://android.googlesource.com/platform/frameworks/base/+/refs/heads/main/core/java/android/os/Debug.java (AOSP source; confirmed the nine getMemoryStat summary.* keys)
- https://iut-fbleau.fr/docs/android/reference/android/os/Debug.MemoryInfo.html (AOSP doc mirror; API levels and kB units)
- https://developer.android.com/ndk/guides/neuralnetworks/migration-guide
- https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/README.md
- https://developers.google.com/edge/litert/models/measurement (ai.google.dev/edge/litert/models/measurement now 301-redirects here)

---

## testing-cli

### Findings

## Verified by actually building on this machine (not just docs)

I bootstrapped a wrapper, wrote a minimal AGP project, and ran `assembleDebug` to completion. **AGP 8.13.0 + Gradle 8.14 + JDK 21 + compileSdk 36 = BUILD SUCCESSFUL**, APK produced, zero SDK downloads required.

### Machine state I confirmed (via filesystem inspection)
- JDK: `openjdk 21.0.11 Temurin` at `C:\Users\<username>\tools\jdk-21.0.11+10`
- SDK root: `C:\Users\<username>\AppData\Local\Android\Sdk`
- `platforms/`: `android-35`, `android-36`, `android-36.1` → **compileSdk 36 is present, no download needed**
- `build-tools/`: `35.0.0`, `36.1.0` — **36.0.0 is NOT installed** (this matters, see pitfalls)
- `ndk/`: `27.0.12077973`, `28.2.13676358`
- `licenses/android-sdk-license` **exists** → AGP auto-download is unblocked
- `cmdline-tools/` **is empty** → no `sdkmanager`
- `platform-tools/adb.exe` present
- `gradle` NOT on PATH
- **A full Gradle 8.14 distribution is already cached** at `C:\Users\<username>\.gradle\wrapper\dists\gradle-8.14-all\<cache-id>\gradle-8.14\bin\gradle.bat` — this solves the "no gradle to generate the wrapper" chicken-and-egg problem with no network access.
- `adb devices` currently returns **an empty list** — no device attached right now.

### Official compatibility data (developer.android.com)

From the AGP release notes, per-version minimums:

| AGP | Min Gradle | Min/Default JDK | Min/Default Build Tools | Default NDK | Max API |
|---|---|---|---|---|---|
| 8.13.0 | 8.13 | 17 | 35.0.0 | 27.0.12077973 | 36.1 |
| 9.0 | 9.1.0 | 17 | 36.0.0 | 28.2.13676358 | 36.1 |
| 9.2.0 | 9.4.1 | 17 | 36.0.0 | 28.2.13676358 | 37.0 |
| 9.3 | 9.5.0 | — | — | — | — |

Minimum AGP per API level (from `about-agp`): **API 36 → AGP 8.9.1**; API 36.1 → AGP 8.13.0; API 37.0 → AGP 9.1.1.

The docs state JDK **17 is the minimum** for both AGP 8.x and 9.x — they never state a maximum. Gradle's own compatibility page states JDK 21 is supported for running Gradle **since Gradle 8.5**, and Gradle 9.x supports Java 17–26. So JDK 21 is comfortably inside both windows; I confirmed empirically that the Gradle 8.14 daemon launches under Temurin 21.0.11 and drives AGP 8.13 correctly.

### local.properties
`developer.android.com/build` documents `sdk.dir` (path to SDK), `cmake.dir`, and `ndk.dir` (**deprecated**). Verbatim caution: *"The `local.properties` file is reserved for properties specific to the Android Gradle plugin. Putting your own values in this file can cause problems."* It is machine-specific and should be gitignored. `ANDROID_HOME` env var works as an alternative and is currently **unset** on this machine.

### Missing SDK platform without sdkmanager
Officially: *"Gradle can automatically download missing SDK packages that a project depends on, as long as the corresponding SDK license agreements have already been accepted in the SDK Manager... This licenses directory is necessary for Gradle to auto-download missing packages."* Controlled by `android.builder.sdkDownload` (default true; set `=false` to disable). Since `licenses/android-sdk-license` exists here, AGP can self-heal missing packages over the network **without cmdline-tools** — AGP does the downloading itself, it does not shell out to `sdkmanager`.

### Test tasks (confirmed present by running `gradlew :app:tasks --group=verification`)
`test`, `testDebugUnitTest`, `testReleaseUnitTest`, `connectedAndroidTest`, `connectedDebugAndroidTest`, `connectedCheck`, `deviceAndroidTest`, `deviceCheck`.

### AndroidX Test versions (verified against Google's Maven group-index XML, not just release notes)
`androidx.test:core:1.7.0`, `runner:1.7.0`, `rules:1.7.0`, `monitor:1.9.0`, `orchestrator:1.6.1`; `androidx.test.ext:junit:1.3.0`, `truth:1.7.0`; `androidx.test.services:test-services:1.6.0`, `storage:1.6.0`. Espresso `3.7.0` comes from the release-notes page only (I did not cross-check its group-index).

### Decisions

## Use AGP 8.13.0 + Gradle 8.14. Do NOT use AGP 9.x.

Rationale, specific to this machine:
- Gradle 8.14 is **already in the wrapper cache** → offline-capable, no 150MB download.
- AGP 8.13's default build-tools is **35.0.0, which is installed**. AGP 9.x defaults to **36.0.0, which is NOT installed** → would trigger a network auto-download or a hard failure if offline.
- AGP 8.13 max API is 36.1, so compileSdk 36 is fully supported.
- AGP 9.0 flips many defaults at once (`android.builtInKotlin=true` requiring KGP ≥ 2.2.10, `android.newDsl=true` hiding `BaseExtension`/`applicationVariants`, `android.uniquePackageNames=true`, `targetSdk` defaulting to `compileSdk`, `android.proguard.failOnMissingFiles=true`). Not worth it for a fresh classifier app.

## Exact paths for this machine
- SDK: `C:/Users/<username>/AppData/Local/Android/Sdk`
- adb: `C:/Users/<username>/AppData/Local/Android/Sdk/platform-tools/adb.exe`
- Bootstrap gradle: `C:/Users/<username>/.gradle/wrapper/dists/gradle-8.14-all/<cache-id>/gradle-8.14/bin/gradle.bat`

## Step 1 — bootstrap the wrapper (VERIFIED WORKING)
The `wrapper` task requires a settings file to exist first, otherwise it fails with *"Directory ... does not contain a Gradle build"*. So:

```bash
cd D:/tea/tea-leaf-android-feia/android
printf 'rootProject.name = "tealeaf"\n' > settings.gradle.kts
"C:/Users/<username>/.gradle/wrapper/dists/gradle-8.14-all/<cache-id>/gradle-8.14/bin/gradle.bat" \
  wrapper --gradle-version 8.14 --distribution-type all
```
Produces `gradlew`, `gradlew.bat`, `gradle/wrapper/gradle-wrapper.jar`, `gradle/wrapper/gradle-wrapper.properties`.

## Step 2 — local.properties (MUST use forward slashes)
```properties
sdk.dir=C:/Users/<username>/AppData/Local/Android/Sdk
```

## Step 3 — build and test commands
```bash
# Build debug APK -> app/build/outputs/apk/debug/app-debug.apk
./gradlew.bat :app:assembleDebug

# JVM unit tests
./gradlew.bat :app:testDebugUnitTest
./gradlew.bat test                      # all variants

# Single unit test class/method
./gradlew.bat :app:testDebugUnitTest --tests '*.ClassifierTest'
./gradlew.bat :app:testDebugUnitTest --tests '*.sampleTestMethod'

# Instrumented tests on the attached S23+
./gradlew.bat :app:connectedDebugAndroidTest
./gradlew.bat cAT                       # abbreviation for connectedAndroidTest

# Filter instrumented tests (-P prefixed runner args)
./gradlew.bat :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=com.example.tealeaf.ClassifierTest
./gradlew.bat :app:connectedCheck \
  -Pandroid.testInstrumentationRunnerArguments.size=medium
```

## Report output paths (official)
- Unit HTML: `app/build/reports/tests/` · Unit XML: `app/build/test-results/`
- Instrumented HTML: `app/build/reports/androidTests/connected/`
- Instrumented XML: `app/build/outputs/androidTest-results/connected/`

## adb: device check, install, screenshots, logs
```bash
ADB="C:/Users/<username>/AppData/Local/Android/Sdk/platform-tools/adb.exe"

$ADB devices -l                                   # currently EMPTY - plug in + authorize first
$ADB install -r -t app/build/outputs/apk/debug/app-debug.apk
$ADB install -g app/build/outputs/apk/debug/app-debug.apk   # auto-grant manifest permissions (CAMERA)

# Screenshot straight to the host (no temp file on device)
$ADB exec-out screencap -p > shot.png

# Or two-step
$ADB shell screencap /sdcard/screen.png && $ADB pull /sdcard/screen.png

# Screen recording (180s max, no audio)
$ADB shell screenrecord --size 1280x720 --bit-rate 6000000 /sdcard/demo.mp4
$ADB pull /sdcard/demo.mp4

# Raw instrumentation run, bypassing Gradle
$ADB shell am instrument -w com.example.tealeaf.test/androidx.test.runner.AndroidJUnitRunner
$ADB shell am instrument -w -e class com.example.tealeaf.ClassifierTest \
  com.example.tealeaf.test/androidx.test.runner.AndroidJUnitRunner
```
Use `$ADB -s <serial>` or `export ANDROID_SERIAL=<serial>` to disambiguate devices.

## Test instrumentation runner
Exact FQCN: `androidx.test.runner.AndroidJUnitRunner`. Orchestrator (optional) is `testOptions { execution = "ANDROIDX_TEST_ORCHESTRATOR" }` plus `androidTestUtil "androidx.test:orchestrator:1.6.1"` and runner arg `clearPackageData: 'true'`.

### Pitfalls

## 1. local.properties backslashes silently corrupt the SDK path — I HIT THIS FOR REAL
`local.properties` is a Java `.properties` file, so single backslashes are **escape sequences**. Writing `sdk.dir=C\:\Users\<username>\AppData\Local\Android\Sdk` makes `\U`, `\L`, `\A` get eaten.

**Symptom (verbatim from my failed build):**
```
> Could not determine the dependencies of task ':app:compileDebugJavaWithJavac'.
   > java.io.IOException: The filename, directory name, or volume label syntax is incorrect
```
Note it does NOT say "SDK location not found" — the error is maximally unhelpful and points at javac, not at the SDK path. **Fix:** forward slashes (`C:/Users/...`) or doubled backslashes (`C:\\Users\\...`). Beware: shell heredocs/printf on Windows readily produce the broken single-backslash form.

## 2. `gradle wrapper` fails if no settings file exists yet
Running the wrapper task in an empty dir dies with *"Directory ... does not contain a Gradle build"*. Create `settings.gradle.kts` **before** invoking the wrapper task. (I hit this too.)

## 3. build-tools 36.0.0 is absent — only bites if you move to AGP 9.x
AGP 9.0/9.2 default to build-tools `36.0.0`, which is not installed here. AGP would attempt an auto-download (works, licenses dir exists — but needs network) or fail offline. If you ever move to AGP 9.x, pin explicitly: `android { buildToolsVersion = "36.1.0" }`. AGP 8.13 defaults to 35.0.0 and is unaffected — I confirmed my successful build did **not** install 36.0.0.

## 4. No cmdline-tools means no `sdkmanager`, but this is mostly a non-issue
There is no `sdkmanager.bat` to install packages or run `sdkmanager --licenses`. Mitigations, in order of preference: (a) nothing needed — android-36 and build-tools 35.0.0 are already present; (b) AGP auto-download covers gaps because `licenses/android-sdk-license` exists; (c) last resort, download `commandlinetools-win-*.zip` from developer.android.com/studio and unzip to `<SDK>/cmdline-tools/latest/`. Do **not** hand-copy platform directories between machines.

## 5. `android.builder.sdkDownload` will reach out to the network mid-build
Default is `true`. If you want reproducible/offline builds, set `android.builder.sdkDownload=false` in `gradle.properties` — but then a missing package becomes a hard build failure rather than a silent download.

## 6. No device is currently attached
`adb devices -l` returned an empty list. `connectedDebugAndroidTest` fails with no connected devices. Verify authorization (`adb devices` showing `device`, not `unauthorized`) before trusting any instrumented-test run. Add `$ADB wait-for-device` in scripts.

## 7. AGP 9.x breaking changes, if you ignore my recommendation
`android.newDsl=true` hides `BaseExtension` and removes `applicationVariants`/`libraryVariants` (migrate to `androidComponents { onVariants { } }`). `android.builtInKotlin=true` requires KGP ≥ 2.2.10 and makes the `kotlin-android` plugin unnecessary. `android.proguard.failOnMissingFiles=true` turns a missing ProGuard file into a build failure. `targetSdk` now defaults to `compileSdk`. Opt-outs `android.newDsl=false` / `android.builtInKotlin=false` exist but are **removed in AGP 10.0 (mid-2026)**.

## 8. Windows path-length and Gradle daemon JVM
Keep the project path short. `D:\tea\tea-leaf-android-feia\android` is fine. If the daemon OOMs during R8/dex, set `org.gradle.jvmargs=-Xmx4g` in `gradle.properties` (the machine has 6.91 GB RAM, so do not go higher).

## 9. `--no-daemon` forks a single-use daemon anyway
Observed verbatim: *"To honour the JVM settings for this build a single-use Daemon process will be forked."* This is normal, just slower. Prefer leaving the daemon on for iterative work.

## 10. Do not commit `local.properties`
It hardcodes `C:/Users/<username>/...`. Ensure it is in `.gitignore` — the repo's existing `.gitignore` should be checked, since `android/` currently has no Gradle project at all.

### Reference code

// ============ settings.gradle.kts (project root: D:/tea/tea-leaf-android-feia/android) ============
pluginManagement {
    repositories { google(); mavenCentral(); gradlePluginPortal() }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories { google(); mavenCentral() }
}
rootProject.name = "tealeaf"
include(":app")

// ============ build.gradle.kts (root) ============
plugins {
    id("com.android.application") version "8.13.0" apply false
    id("org.jetbrains.kotlin.android") version "2.2.10" apply false
}

// ============ gradle/wrapper/gradle-wrapper.properties (generated; verified content) ============
// distributionBase=GRADLE_USER_HOME
// distributionPath=wrapper/dists
// distributionUrl=https\://services.gradle.org/distributions/gradle-8.14-all.zip
// networkTimeout=10000
// validateDistributionUrl=true
// zipStoreBase=GRADLE_USER_HOME
// zipStorePath=wrapper/dists

// ============ local.properties (gitignored; FORWARD SLASHES) ============
// sdk.dir=C:/Users/<username>/AppData/Local/Android/Sdk

// ============ gradle.properties ============
// org.gradle.jvmargs=-Xmx4g -Dfile.encoding=UTF-8
// org.gradle.parallel=true
// org.gradle.caching=true
// android.useAndroidX=true
// android.nonTransitiveRClass=true
// # set to false only if you need fully offline builds:
// # android.builder.sdkDownload=false

// ============ app/build.gradle.kts ============
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.tealeaf"
    compileSdk = 36                       // android-36 is installed; AGP 8.13 max is 36.1

    defaultConfig {
        applicationId = "com.example.tealeaf"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        // Uncomment together with test-services if you adopt TestStorage screenshots:
        // testInstrumentationRunnerArguments["useTestStorageService"] = "true"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }

    // .tflite must not be compressed or the interpreter cannot mmap it
    androidResources { noCompress += "tflite" }

    testOptions {
        unitTests.isReturnDefaultValues = true
        // execution = "ANDROIDX_TEST_ORCHESTRATOR"   // optional, needs androidTestUtil below
    }
}

dependencies {
    testImplementation("junit:junit:4.13.2")

    androidTestImplementation("androidx.test:core:1.7.0")
    androidTestImplementation("androidx.test:runner:1.7.0")
    androidTestImplementation("androidx.test:rules:1.7.0")
    androidTestImplementation("androidx.test.ext:junit:1.3.0")
    androidTestImplementation("androidx.test.ext:truth:1.7.0")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.7.0")

    // Only if you enable orchestrator / TestStorage:
    // androidTestUtil("androidx.test:orchestrator:1.6.1")
    // androidTestUtil("androidx.test.services:test-services:1.6.0")
}

// ============ Full command-line flow (PowerShell or Git Bash) ============
// cd D:/tea/tea-leaf-android-feia/android
// ./gradlew.bat :app:assembleDebug
// ./gradlew.bat :app:testDebugUnitTest
// ./gradlew.bat :app:connectedDebugAndroidTest
//
// $ADB = "C:/Users/<username>/AppData/Local/Android/Sdk/platform-tools/adb.exe"
// & $ADB wait-for-device
// & $ADB install -r -t -g app/build/outputs/apk/debug/app-debug.apk
// & $ADB exec-out screencap -p > shot.png

### Not confirmed

## Could NOT confirm

**1. In-test screenshot APIs (`takeScreenshot()`, `captureToBitmap()`, `writeToTestStorage()`).** These surfaced in search snippets, but every official reference page I fetched failed to render the actual API content — `developer.android.com/reference/androidx/test/core/graphics/package-summary` and `.../androidx/test/runner/screenshot/Screenshot` both returned only navigation chrome (they are SPA-rendered). `developer.android.com/training/testing/ui-tests/screenshot` explicitly punts: *"You can use third-party tools to create both instrumented and local screenshot tests."* **I will not state signatures I did not read.** Recommendation: use the fully-documented `adb exec-out screencap -p > shot.png`, which is sufficient for capturing classifier UI states and which I verified is documented on `developer.android.com/tools/adb`.

**2. The on-device additional-test-output path.** A search snippet gave `/sdcard/Android/media/<APP_NAME>/additional_test_output` and `build/outputs/connected_android_test_additional_output/`, but this came from **non-official blog/issue-tracker sources**, not developer.android.com. I fetched `studio/test/advanced-test-setup` hoping to confirm and it does not cover additional test output at all. Treat those paths as unverified.

**3. Espresso 3.7.0.** Taken from the AndroidX Test release-notes page. Unlike the other coordinates, I did not cross-check it against `dl.google.com/dl/android/maven2/androidx/test/espresso/group-index.xml`. All other versions I quote were verified against Google's Maven group-index XML directly.

**4. A discrepancy worth flagging.** The release-notes page rendered the test-services coordinate as `androidx.test.services:services`. I checked Google's Maven group-index and the real artifact IDs under that group are **`test-services`** and **`storage`** — there is no `services` artifact. Use `androidx.test.services:test-services:1.6.0`. Similarly the release notes said `androidx.test:monitor:1.8.0` but group-index shows latest stable **1.9.0**.

**5. Whether AGP officially certifies JDK 21 (as opposed to merely permitting it).** Android's docs state only a *minimum* of JDK 17 for AGP 8.x and 9.x and never name a maximum or a tested-upper-bound. Gradle's compatibility page independently confirms JDK 21 support since Gradle 8.5. That JDK 21 works here is something **I verified empirically** with a real successful build — it is not an explicit Android-docs guarantee. I infer, rather than quote, that this is a supported configuration.

**6. Instrumented tests have not actually been executed.** No device was attached during this research (`adb devices -l` empty), so `connectedDebugAndroidTest`, the report paths under `build/reports/androidTests/connected/`, and the adb screenshot commands are documented-but-not-run in this environment. The task-name existence I did verify by listing the verification task group. Everything in the build path (wrapper bootstrap, assembleDebug, compileSdk 36, JDK 21) was genuinely executed.

**7. NDK.** Both 27.0.12077973 and 28.2.13676358 are installed and AGP 8.13's default (27.0.12077973) matches, but my test project had no native code, so the NDK path is unexercised. Irrelevant unless you add JNI — the LiteRT/TFLite AAR ships prebuilt `.so` files and needs no NDK.

### Sources

- https://developer.android.com/build/releases/about-agp
- https://developer.android.com/build/releases/gradle-plugin
- https://developer.android.com/build/releases/agp-8-13-0-release-notes
- https://developer.android.com/build/releases/agp-9-0-0-release-notes
- https://developer.android.com/build/jdks
- https://developer.android.com/build
- https://developer.android.com/studio/test/command-line
- https://developer.android.com/studio/test/advanced-test-setup
- https://developer.android.com/studio/intro/update
- https://developer.android.com/tools/adb
- https://developer.android.com/training/testing/instrumented-tests/androidx-test-libraries/test-setup
- https://developer.android.com/training/testing/instrumented-tests/androidx-test-libraries/runner
- https://developer.android.com/training/testing/ui-tests/screenshot
- https://developer.android.com/jetpack/androidx/releases/test
- https://docs.gradle.org/current/userguide/compatibility.html
- https://dl.google.com/dl/android/maven2/androidx/test/group-index.xml
- https://dl.google.com/dl/android/maven2/androidx/test/ext/group-index.xml
- https://dl.google.com/dl/android/maven2/androidx/test/services/group-index.xml

---

## annotation-schema

### Findings

## Bottom line on coverage

There is **no annotation-schema standard for agricultural ML field feedback**. I searched for one and did not find it. What exists is a set of adjacent, genuinely authoritative sources that collectively determine the fields you should record. I fetched each of the following; item numbering and wording below is quoted from pages I actually loaded.

### 1. CLAIM (medical imaging AI reporting; EQUATOR-indexed) — annotator fields
Fetched https://pmc.ncbi.nlm.nih.gov/articles/PMC8017414/ (CLAIM 2020, the version I could read in full). Verbatim items:

- **Item 14** — "Include detailed, specific definitions of the ground truth annotations, ideally referencing common data elements."
- **Item 15** — "Describe the rationale for the choice of the reference standard and the potential errors, biases, and limitations."
- **Item 16** — "Specify the number of human annotators and their qualifications. Describe the instructions and training given to annotators."
- **Item 17** — "Specify the software used for manual, semiautomated, or automated annotation, including the version number."
- **Item 18** — "Describe the methods to measure inter- and intrarater variability, and any steps taken to reduce or mitigate this variability."

CLAIM 2024 update: the full text at https://pubs.rsna.org/doi/full/10.1148/ryai.240300 returned **HTTP 403**, so I could not verify 2024 item numbers. From the journal's own summary page (https://radiologyai.substack.com/p/claim-2024-update) I confirmed two terminology changes: (a) **"reference standard"** is now the preferred term over "ground truth"/"gold standard" — "the benchmark against which an AI system is being judged"; (b) the term **"validation" is discouraged** because clinicians read it as "clinical validation" — use "evaluation" or "tuning". *I infer, but did not verify, that items 14–18 survive with similar content under different numbers.*

### 2. DECIDE-AI (Nature Medicine / BMJ 2022) — this is the closest fit to your actual situation
Fetched https://pmc.ncbi.nlm.nih.gov/articles/PMC9116198/. DECIDE-AI covers *early, live, small-scale evaluation of AI decision support where a human is in the loop* — structurally identical to a farmer confirming/correcting a tea-leaf prediction. 17 AI-specific items (28 subitems) + 10 generic. The directly load-bearing item, verbatim:

- **Item 12 (Human–computer agreement)** — "Report on the user agreement with the AI system. Describe any instances of and reasons for user variation from the AI system's recommendations and, if applicable, users changing their mind based on the AI system's recommendations."

Also verbatim:
- **3b** — "Describe how users were recruited, stating the inclusion and exclusion criteria, and how the intended number of recruited users was decided"
- **3c** — "Describe steps taken to familiarise the users with the AI system, including any training received prior to the study"
- **6a** — "Provide a description of how significant errors/malfunctions were defined and identified"
- **7 (Human factors)** — "Describe the human factors tools, methods or frameworks used, the use cases considered, and the users involved"
- **11 (Modifications)** — "Report any changes made to the AI system or its hardware platform during the study. Report the timing of these modifications, the rationale for each..."
- **13a** — "List any significant errors/malfunctions related to: AI system recommendations, supporting software/hardware, or users"
- **14b** — "Report on the user learning curves evaluation"

Item 11 matters concretely: if you ship a new .tflite mid-study, every record must carry the model version or the dataset is unanalysable.

### 3. CrowdWorkSheets (Díaz et al., ACM FAccT 2022) — annotator documentation
Fetched https://ar5iv.labs.arxiv.org/html/2206.08931. Five sections. Verbatim questions relevant to you:

- *Task formulation*: "At a high level, what are the subjective aspects of your task?"; "What assumptions do you make about annotators?"; "What are the precise instructions that were provided to annotators?"
- *Selecting annotators*: "Are there certain perspectives that should be privileged? If so, how did you seek these?"; "If you have any aggregated sociodemographic statistics about your annotator pool, please describe."
- *Platform*: "What annotation platform did you utilize?"; "How much were annotators compensated? Did you consider any particular pay standards?"
- *Analysis*: "Have you conducted any analysis on disagreement patterns?"; "Did you analyze potential sources of disagreement?"; **"How do the individual annotator responses relate to the final labels released in the dataset?"**
- *Release/maintenance*: "Do you have reason to believe the annotations in this dataset may change over time?"; "Were annotators informed about how the data is externalized?"; **"Is there a process by which annotators can later choose to withdraw their data?"**

The last one is a hard schema requirement, not a policy note: withdrawal is impossible unless each record carries a stable annotator/session identifier.

### 4. Croissant-RAI (MLCommons) — machine-readable property names that already exist
Fetched https://docs.mlcommons.org/croissant/docs/croissant-rai-spec.html. Use these exact names rather than inventing your own:

`rai:dataAnnotationProtocol`, `rai:dataAnnotationPlatform`, `rai:dataAnnotationAnalysis`, `rai:annotatorDemographics`, `rai:annotationsPerItem`, `rai:machineAnnotationTools`, `rai:dataCollection`, `rai:dataCollectionType`, `rai:dataCollectionRawData`, `rai:dataCollectionTimeframe`, `rai:personalSensitiveInformation`, `rai:dataReleaseMaintenance`.

Note `rai:dataCollectionType` has recommended values including "Direct measurement", "Manual Human Curator", "Surveys". The spec page I read contains **no explicit consent property** — privacy is handled indirectly via `rai:personalSensitiveInformation`.

### 5. Darwin Core (TDWG) — a real, ratified vocabulary for verified vs unverified field IDs
Fetched https://dwc.tdwg.org/terms/. This is the single best find for your question and it is directly reusable. Exact terms and definitions:

- **`identificationVerificationStatus`** — "A categorical measure of how thoroughly a taxonomic determination has been verified as accurate." Controlled vocabulary drawn from HISPID/ABCD; example value `0` = unverified.
- **`identifiedBy`** — "The name of an agent responsible for assigning the species identification." Multiple values separated by space-pipe-space (` | `). **Documented examples include `MegaDetector V5`** — i.e. Darwin Core explicitly contemplates a *machine* as the identifying agent, exactly your model-vs-human case.
- **`identifiedByID`** — persistent identifier for the identifying agent (e.g. an ORCID URI).
- **`dateIdentified`** — ISO 8601-1:2019 format.
- **`identificationQualifier`** — "A concise phrase or standard term (e.g., 'cf.', 'aff.') reflecting the determiner's reservations about the identification." This is the established way to record *hedged* identification.
- **`identificationRemarks`** — free-text commentary on the determination process.
- **`identificationReferences`** — bibliographic materials consulted.
- **`coordinateUncertaintyInMeters`** — "the smallest circle encompassing the entire location"; zero is invalid; leave empty if undeterminable. Documented example: `30` for post-2000 GPS in good conditions.
- **`georeferenceProtocol`** — description/citation of the georeferencing procedure.
- **`basisOfRecord`** — controlled values include `HumanObservation` and `MachineObservation`.

### 6. iNaturalist Data Quality Assessment — a working tiered verification model
Fetched https://help.inaturalist.org/en/support/solutions/articles/151000169936-what-is-the-data-quality-assessment-and-how-do-observations-qualify-to-become-research-grade-. Three grades: **Casual / Needs ID / Research Grade**. Verbatim mechanics:

- A "verifiable" observation needs accurate date, georeferenced coordinates, photo/sound, and must not be captive/cultivated or human.
- Research Grade requires **more than 2/3 of identifiers agreeing** at species level, and the community taxon must agree with the observation taxon.
- Downgrade-to-Casual flags are individually voted and include: "the date doesn't look accurate", "the location doesn't look accurate", "the organism isn't wild/naturalized", "doesn't present evidence of an organism", "doesn't present recent (~100 years) evidence", "doesn't present accurate evidence" (explicitly covering AI manipulation), and evidence involving multiple unrelated subjects.

The design lesson: verification status is a **derived, recomputable tier over independently-stored atomic votes**, not a boolean someone sets.

### 7. What non-expert field annotation actually supports — hard numbers
Fetched https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2020.590889/full (PlantVillage Nuru, cassava CMD/CBSD/CGM, East Africa). Reported overall accuracy under the CaSRAT assessment:

- **Trained researchers: 86%**
- **Trained agricultural extension officers: 49%**
- **Trained farmers: 23%**
- **Nuru (2020 version): 65%**

Single-leaf in-field Nuru accuracy: CMD 59%, CBSD 21%, CGM-damage 56%. With six leaves (3 upper + 3 lower): "CMD, CBSD and CGM-damage were diagnosed with 93, 73 and 93% accuracy." Extension officers improved only from 34.9% to 49.9% after two weeks of using Nuru.

I also fetched the companion methods paper https://pmc.ncbi.nlm.nih.gov/articles/PMC7775399/: the reference standard was set by "two researchers with at least three years of experience on cassava pests and diseases", and separately confirmed on a 75-plant subset by **molecular diagnosis — conventional PCR/qPCR** for EACMV and cassava brown streak viruses, with CTAB nucleic acid extraction. Notably, the paper does **not** describe a conflict-resolution procedure between the two experts; it reports "percentage matches".

This is the single most important finding for your project: **in the closest published analogue, untrained-user field labels were roughly 23% accurate — materially worse than the model.** Non-expert field annotation in this domain cannot serve as a reference standard.

### 8. Confidence tiering that outperformed a consensus grade
Fetched https://pmc.ncbi.nlm.nih.gov/articles/PMC11461752/. Proposes a four-step protocol producing three tiers (Low 4–8, Medium 9–13, High 14–18) scored across species characteristics (visible diagnostic features; similar species regionally), media quality, and georeferencing (sub-1 km scores higher). It defines explicit "stop points" where scoring is impossible. Validation: Kendall's τ = 0.82, p < 0.001 against accuracy, "substantially outperforming iNaturalist's built-in quality grade system." Caveat quoted: "it does require that those applying it hold foundational knowledge on the taxa being considered."

### Decisions

## Decisions

**1. Do not call the user's input a "label". Call it a `UserResponse`, and store it alongside — never overwriting — the model output.** Store `modelPrediction`, `modelConfidence`, the full 7-way `softmaxVector`, `modelVersion`, and the user's `assertedClass` as separate columns. DECIDE-AI item 12 is unanswerable if you overwrite.

**2. Adopt Darwin Core term names verbatim for the fields where they exist.** Use `identificationVerificationStatus`, `identifiedBy`, `identifiedByID`, `dateIdentified`, `identificationQualifier`, `identificationRemarks`, `coordinateUncertaintyInMeters`, `basisOfRecord`. `identifiedBy` accepts a model name (`MegaDetector V5` is a documented example) — so record `identifiedBy = "tea-leaf-classifier v1.3.0"` for the machine row and the annotator ID for the human row. Two rows, not one.

**3. Model the verification tier as derived, not stored-as-truth.** Follow iNaturalist: store atomic votes/assertions; compute a tier. Concretely, four tiers:
- `UNVERIFIED` — one non-expert response only
- `COMMUNITY` — ≥2 independent non-expert responses agreeing (iNaturalist uses a >2/3 threshold)
- `EXPERT` — an agronomist/plant pathologist confirmed
- `LAB` — culture/PCR confirmed

Only `EXPERT` and `LAB` may be used as a reference standard. Nuru used PCR/qPCR for exactly this.

**4. Capture the response *before* revealing the prediction, for a designated subset.** Ship two UI modes and record which one produced each record in a `elicitationMode` field: `BLIND_FIRST` (user picks, then sees model) vs `ASSISTED` (model shown first, user confirms/corrects). Without `BLIND_FIRST` records you cannot separate user skill from anchoring. Target a fixed fraction (e.g. every 5th capture) rather than user choice.

**5. Record self-reported expertise as a closed enum matching the Nuru strata**, since that is the only published calibration you can anchor to: `FARMER`, `EXTENSION_OFFICER`, `RESEARCHER`, `PLANT_PATHOLOGIST`, `UNKNOWN`. Do not use a free-text field.

**6. Record confidence as a coarse ordinal, not a 0–100 slider.** The literature I read notes humans are poorly calibrated at numerical estimates and that forced-choice formats yield higher inter-rater reliability than fine Likert scales. Use 3 levels + an explicit escape hatch: `SURE` / `FAIRLY_SURE` / `GUESSING` / `CANNOT_TELL`. The `CANNOT_TELL` option is the analogue of the confidence paper's "stop points" and of Darwin Core `identificationQualifier` — without it, uncertainty is silently coerced into a class.

**7. Store one row per annotator per image, never an aggregate.** CrowdWorkSheets: "How do the individual annotator responses relate to the final labels released in the dataset?" Aggregation is a lossy, irreversible operation — do it in analysis, not in the app.

**8. Persist a stable `annotatorId` (random UUID, generated on device, not an IMEI/ANDROID_ID/account email) and a `consentVersion` + `consentTimestamp`.** The UUID is what makes CrowdWorkSheets' withdrawal question ("Is there a process by which annotators can later choose to withdraw their data?") answerable. Do not use a hardware identifier — on API 36 you cannot read most of them anyway, and it defeats withdrawal.

**9. GPS: store `coordinateUncertaintyInMeters` from `Location.getAccuracy()`, and truncate coordinates before export.** Darwin Core says zero is invalid — if accuracy is unavailable, leave it null rather than writing 0. A tea plot's exact coordinates plus a disease label is commercially sensitive to the grower; store full precision locally, coarsen on upload.

**10. Record capture conditions, because the Nuru result showed accuracy is dominated by them.** Six leaves beat one leaf by 34 percentage points for CMD (59%→93%). Record `leavesInFrame`, `leafSurface` (`UPPER`/`LOWER`), and whether the image passed a blur/exposure check. This is also iNaturalist's "media quality" scoring axis.

**11. Log the runtime, not just the model.** Given your device notes: record `modelVersion`, `modelSha256`, `delegate` (`CPU`/`GPU`/`XNNPACK`), `numThreads`, and `androidApiLevel`. Since `android.hardware.neuralnetworks` is absent on this SM-S916B, an NNAPI delegate path is not available and any record claiming it would be wrong — record what was actually used, resolved at runtime.

**12. Record the preprocessing contract explicitly, once, in the dataset-level metadata**: input 224×224×3 float32, **raw [0,255], normalization embedded in the graph**. Store this as `rai:dataAnnotationProtocol` free text. A future maintainer who re-normalizes to [0,1] will silently halve accuracy with no error.

**13. Dataset-level metadata: emit a Croissant file with the RAI extension**, using the exact property names `rai:dataAnnotationProtocol`, `rai:dataAnnotationPlatform`, `rai:annotationsPerItem`, `rai:annotatorDemographics`, `rai:dataCollectionType` (value: `Direct measurement`), `rai:dataCollectionTimeframe`, `rai:personalSensitiveInformation`.

**14. Agreement statistic: report Krippendorff's α, not Cohen's κ, and always alongside the full 7×7 confusion matrix and per-class recall.** κ assumes two fixed raters with equal marginals; you will have variable raters per item and missing data, which is precisely α's design case.

## Claims your data can and cannot support

Can support:
- User–model agreement rate, per class, per expertise stratum (DECIDE-AI item 12).
- Correction *patterns* — which classes users move predictions away from. This is a hypothesis generator.
- Distribution shift evidence: field image statistics vs training set.
- Usability, learning curve, latency, thermal behaviour (DECIDE-AI 14b).

Cannot support, and must not be claimed:
- **"Field accuracy is X%"** from non-expert confirmations. Nuru measured farmers at 23%.
- **"The model was validated in the field."** CLAIM 2024 explicitly discourages "validation"; say "evaluated".
- **Per-class sensitivity/specificity** from a partially-verified subset without an explicit sampling-fraction correction.
- Any retraining claim: user corrections at 23% expected accuracy are a label-noise injection, not a training signal.

### Pitfalls

**1. Showing the prediction before the user answers — the single biggest threat here.**
*Symptom:* agreement rate climbs toward 90%+ and looks like validation. It is anchoring/automation bias. The literature I surveyed documents that users are less likely to correct erroneous suggestions when correcting requires extra effort or when they hold favourable attitudes toward AI, and that anchoring intensifies under time pressure — both true of a farmer in a field. A confirm-heavy UI (large "Correct" button, correction buried behind a picker) manufactures agreement. *Mitigation:* the `BLIND_FIRST` subset in decision 4, plus making confirm and correct symmetric in tap cost.

**2. Verifying only the cases the model flagged as uncertain — partial verification bias.**
*Symptom:* sensitivity is biased **up** and specificity **down**. This is the classic diagnostic-accuracy trap: when a positive index test makes reference-standard application more likely, accuracy estimates are systematically wrong. If you only send low-confidence or disagreed images to an agronomist, every accuracy number you compute is inflated. *Mitigation:* expert-verify all disagreements **plus a random sample of agreements**, store the sampling fraction per record (`verificationSamplingFraction`), and correct for it in analysis.

**3. Cohen's κ collapsing on your 7-class imbalanced problem — the kappa paradox.**
*Symptom:* raters visibly agree on most images, but κ comes out near 0.2 and you conclude annotation is unreliable. κ is deflated when the class distribution is skewed toward one category. Scott's π and Krippendorff's α share the problem to a degree; Gwet's AC1 is designed to be stable under it. *Mitigation:* never report a bare κ; report α plus the confusion matrix plus per-class recall, and consider AC1.

**4. Overwriting the model prediction with the user's correction.**
*Symptom:* silent and terminal — the dataset looks fine, but agreement rate, correction patterns, and DECIDE-AI item 12 are all permanently uncomputable. There is no recovery. Enforce append-only at the DB layer.

**5. Aggregating multiple responses to a majority label at write time.**
*Symptom:* inter-annotator agreement is uncomputable, and the disagreement signal — which the "learning from disagreement" line of work treats as informative rather than as noise to be adjudicated away — is destroyed. CrowdWorkSheets asks directly how individual responses relate to released labels. Aggregate in analysis only.

**6. No model version on the record.**
*Symptom:* you ship a retrained .tflite, accuracy appears to shift, and you cannot tell whether the model changed or the season did. DECIDE-AI item 11 exists for exactly this. Equally: no delegate/threads field means a GPU-vs-CPU numerical difference is invisible.

**7. Forcing a choice from 7 classes with no "cannot tell".**
*Symptom:* a fat spurious diagonal on the most visually generic class. Users who can't tell will pick the option the model suggested, or the most familiar one. The confidence paper's "stop points" and Darwin Core's `identificationQualifier` both exist because hedged determinations are real and must be recordable.

**8. Free-text expertise fields.**
*Symptom:* unstratifiable data. "farmer", "Farmer", "grower", "I have a tea garden" cannot be pooled, so you can never reproduce the Nuru-style stratified comparison that is the entire point of recording expertise.

**9. Hardware identifiers as annotator ID.**
*Symptom:* consent withdrawal becomes impossible and you acquire a privacy liability. On Android 10+ non-resettable identifiers are restricted; on API 36 you will get a permission failure or a null. Generate a UUID on first run, store in encrypted prefs, and make it user-resettable.

**10. Treating a single expert's opinion as the reference standard.**
*Symptom:* unmeasurable error floor. CLAIM item 18 requires methods to measure inter- and intra-rater variability. Note that even the Nuru study used two experienced researchers and did not report a conflict-resolution procedure — a documented weakness in the closest analogue, not a template to copy. Use ≥2 experts, record both, and report disagreement.

**11. Assuming "more community agreement = correct" without checking.**
*Symptom:* a confident, wrong reference standard. A 2023 field study of iNaturalist lichen records found fewer than half of user-logged species matched specialists' determinations, and roughly 70% of species-level IDs appearing only on iNaturalist were wrong; the authors recommended raising the research-grade threshold. Consensus among non-experts who share the same misconception converges on the same error. *(This finding came via search summary; I did not fetch the primary lichen paper.)*

**12. Silently re-normalizing input during a pipeline rewrite.**
*Symptom:* accuracy degrades by tens of points with no exception thrown. Your model takes raw [0,255] with normalization in-graph. Record the preprocessing contract in dataset metadata and assert on it in a unit test.

**13. Publishing photos with full-precision GPS.**
*Symptom:* a grower's diseased plot is publicly geolocated. Darwin Core provides `coordinateUncertaintyInMeters` precisely so coarsened coordinates remain scientifically usable — use it rather than dropping location entirely.

### Reference code

// Room entity. Append-only: no @Update, no mutable label column.
// Field names follow Darwin Core (dwc.tdwg.org/terms/) where a term exists.

@Entity(
    tableName = "observation_response",
    indices = [Index("observationId"), Index("annotatorId")]
)
data class ObservationResponse(
    @PrimaryKey val responseId: String,          // UUID v4
    val observationId: String,                    // groups rows about ONE image

    // ---- machine determination (dwc: identifiedBy = model, basisOfRecord = MachineObservation)
    val modelIdentifiedBy: String,                // "tea-leaf-classifier"
    val modelVersion: String,                     // "1.3.0"
    val modelSha256: String,
    val modelPredictedClass: TeaLeafClass,
    val modelSoftmax: FloatArray,                 // all 7 — not just argmax
    val delegate: Delegate,                       // CPU | GPU | XNNPACK  (no NNAPI on SM-S916B)
    val numThreads: Int,
    val inferenceMillis: Long,
    val androidApiLevel: Int,                     // 36 on target device

    // ---- human determination
    val annotatorId: String,                      // device-generated UUID; user-resettable
    val annotatorExpertise: Expertise,
    val elicitationMode: ElicitationMode,         // BLIND_FIRST | ASSISTED  <-- load-bearing
    val assertedClass: TeaLeafClass?,             // null iff CANNOT_TELL
    val assertedConfidence: Confidence,
    val identificationQualifier: String?,         // dwc: hedge, e.g. "cf."
    val identificationRemarks: String?,           // dwc: free text
    val dateIdentified: String,                   // dwc: ISO 8601-1:2019

    // ---- capture conditions (Nuru: 6 leaves >> 1 leaf, 59% -> 93% for CMD)
    val leavesInFrame: Int,
    val leafSurface: LeafSurface,                 // UPPER | LOWER | MIXED
    val blurScore: Float,

    // ---- provenance (dwc)
    val decimalLatitude: Double?,
    val decimalLongitude: Double?,
    val coordinateUncertaintyInMeters: Int?,      // dwc: 0 is INVALID -> use null
    val consentVersion: String,
    val consentTimestamp: Long,
) {
    val basisOfRecord: String get() = "HumanObservation"   // dwc controlled value
}

enum class Delegate { CPU, GPU, XNNPACK }
enum class Expertise { FARMER, EXTENSION_OFFICER, RESEARCHER, PLANT_PATHOLOGIST, UNKNOWN }
enum class Confidence { SURE, FAIRLY_SURE, GUESSING, CANNOT_TELL }
enum class ElicitationMode { BLIND_FIRST, ASSISTED }
enum class LeafSurface { UPPER, LOWER, MIXED }

// Verification tier is DERIVED, never stored as truth (iNaturalist DQA pattern).
enum class VerificationStatus { UNVERIFIED, COMMUNITY, EXPERT, LAB }

object Verification {
    private const val CONSENSUS_THRESHOLD = 2.0 / 3.0   // iNaturalist uses >2/3

    fun statusOf(rows: List<ObservationResponse>): VerificationStatus {
        if (rows.any { it.annotatorExpertise == Expertise.PLANT_PATHOLOGIST })
            return VerificationStatus.EXPERT
        val votes = rows.mapNotNull { it.assertedClass }
        if (votes.size >= 2) {
            val top = votes.groupingBy { it }.eachCount().maxByOrNull { it.value }!!
            if (top.value.toDouble() / votes.size > CONSENSUS_THRESHOLD)
                return VerificationStatus.COMMUNITY
        }
        return VerificationStatus.UNVERIFIED
    }

    // Only EXPERT/LAB may serve as a reference standard.
    // Nuru (Frontiers 2020): trained farmers scored 23% overall.
    fun usableAsReferenceStandard(s: VerificationStatus) =
        s == VerificationStatus.EXPERT || s == VerificationStatus.LAB
}

// ---- Preprocessing contract. Model takes RAW [0,255] float32; normalization is IN-GRAPH.
// Guard it with a test, because re-normalizing fails silently rather than throwing.
fun toInputTensor(bitmap: Bitmap): TensorBuffer {
    require(bitmap.width == 224 && bitmap.height == 224)
    val buf = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.FLOAT32)
    val px = IntArray(224 * 224).also { bitmap.getPixels(it, 0, 224, 0, 0, 224, 224) }
    val out = FloatArray(224 * 224 * 3)
    var i = 0
    for (p in px) {
        out[i++] = ((p shr 16) and 0xFF).toFloat()   // R, raw 0..255 — do NOT divide by 255
        out[i++] = ((p shr 8) and 0xFF).toFloat()
        out[i++] = (p and 0xFF).toFloat()
    }
    buf.loadArray(out)
    return buf
}

### Not confirmed

**Could not verify:**

1. **CLAIM 2024 item numbers and full wording.** https://pubs.rsna.org/doi/full/10.1148/ryai.240300 and https://pubs.rsna.org/page/ai/claim both returned **HTTP 403**. I verified items 14–18 from the **2020** version at PMC8017414, and verified only two 2024 terminology changes ("reference standard" preferred; "validation" discouraged) from the journal's own summary page. Do not cite 2024 item numbers on my authority — obtain the 2024 PDF and re-check.

2. **CrowdWorkSheets question wording is from the ar5iv HTML rendering** (https://ar5iv.labs.arxiv.org/html/2206.08931). Both the arXiv PDF and the FAccT PDF failed to extract as text. The ar5iv rendering is a faithful LaTeX conversion of the arXiv source, but I did not cross-check against the ACM version of record.

3. **The iNaturalist "95% accurate" blog post returned 403.** I therefore have no verified methodology or number for iNaturalist's own accuracy self-estimate, and I have deliberately not cited a figure from it.

4. **The 2023 lichen study finding** (fewer than half of species matching specialists; ~70% of iNaturalist-only species IDs wrong) reached me through a search-result summary only. I did not fetch the primary paper and cannot give you its authors, journal, or exact sample. Verify before citing.

5. **The "learning from disagreement" literature** (Uma, Fornaciari, Hovy, Paun, Plank & Poesio, *JAIR* vol. 72, 2021) — I confirmed the citation exists via search but did **not** fetch it. My characterisation of soft-label methods is from search summaries, not the paper.

6. **Datasheets for Datasets** (Gebru et al., arXiv:1803.09010) — cited from search summary only; I did not fetch the paper. The specific questions I paraphrased (who collected the data, how they were compensated, whether subjects consented) are widely reported but unverified by me at source.

7. **Partial verification bias** — the definition and the "sensitivity up, specificity down" direction come from search summaries of STARD 2015 and the diagnostic-accuracy bias literature, not from a page I fetched. The direction of the bias is standard and I am confident in it, but I did not read the primary text.

8. **Kappa paradox / Gwet's AC1** — from search summaries only, including a non-peer-reviewed methodology blog. The κ prevalence problem is textbook, but treat my specific recommendation of AC1 as a lead to verify, not a settled finding.

9. **Croissant-RAI version.** The spec page I fetched (docs.mlcommons.org/croissant/docs/croissant-rai-spec.html) did not display a version banner I could confirm. Croissant 1.1 was announced in Feb 2026 per search results and adds provenance and "structured usage policies for automated enforcement of consent and licensing" — the property names I listed may have been extended since. **Confirm current property names against the live spec before writing an exporter.**

10. **`identificationVerificationStatus` controlled vocabulary.** Darwin Core points at HISPID and ABCD for values but the terms page gave only one example (`0` = unverified). I did not locate the full enumerated value list. You will need to either fetch the HISPID/ABCD vocabulary or define and document your own four-tier enum — the latter is defensible provided you document it.

11. **Nothing agriculture-specific exists.** I searched explicitly for an ML-in-agriculture reporting guideline covering annotation and found none — only review papers calling for standardization ("the absence of standardized evaluation protocols hinders fair comparison across studies"). **This is a genuine gap, not a gap in my search.** Your schema will be an adaptation of medical and biodiversity practice, and should say so in writing rather than implying conformance to an agricultural standard.

12. **Consent and data-protection specifics for Indonesian / smallholder field data collection** — out of scope of what I fetched. Croissant-RAI has no explicit consent property; CrowdWorkSheets asks about withdrawal but prescribes no mechanism. Get local legal/ethics review; do not treat my `consentVersion` suggestion as sufficient.

13. **The `elicitationMode` / `BLIND_FIRST` design in decision 4 is my inference**, built from the anchoring literature plus DECIDE-AI item 12. I found no source prescribing this specific two-mode capture pattern. It is a reasonable design, not an established standard.

### Sources

- FETCHED — CLAIM 2020 full text, items 14-18 verbatim: https://pmc.ncbi.nlm.nih.gov/articles/PMC8017414/
- FETCHED — DECIDE-AI (Nature Medicine 2022), items 3b/3c/6a/7/11/12/13a/14b verbatim: https://pmc.ncbi.nlm.nih.gov/articles/PMC9116198/
- FETCHED — CrowdWorkSheets (Diaz et al., FAccT 2022), ar5iv HTML rendering of arXiv:2206.08931: https://ar5iv.labs.arxiv.org/html/2206.08931
- FETCHED — Darwin Core Quick Reference Guide (TDWG), identification and georeference terms: https://dwc.tdwg.org/terms/
- FETCHED — Croissant-RAI specification, rai:* property names: https://docs.mlcommons.org/croissant/docs/croissant-rai-spec.html
- FETCHED — iNaturalist Data Quality Assessment, Casual/Needs ID/Research Grade criteria and >2/3 threshold: https://help.inaturalist.org/en/support/solutions/articles/151000169936-what-is-the-data-quality-assessment-and-how-do-observations-qualify-to-become-research-grade-
- FETCHED — PlantVillage Nuru field accuracy (Frontiers Plant Sci 2020), researcher 86% / extension 49% / farmer 23% / Nuru 65%: https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2020.590889/full
- FETCHED — Nuru companion methods paper, expert reference standard + PCR/qPCR molecular confirmation: https://pmc.ncbi.nlm.nih.gov/articles/PMC7775399/
- FETCHED — Method for Conveying Confidence in iNaturalist Observations, 3-tier confidence protocol, Kendall tau 0.82: https://pmc.ncbi.nlm.nih.gov/articles/PMC11461752/
- FETCHED (partial) — CLAIM 2024 update summary by Radiology:AI, 'reference standard' preferred / 'validation' discouraged: https://radiologyai.substack.com/p/claim-2024-update
- BLOCKED 403 — CLAIM 2024 update, item numbers UNVERIFIED: https://pubs.rsna.org/doi/full/10.1148/ryai.240300
- BLOCKED 403 — CLAIM official checklist landing page: https://pubs.rsna.org/page/ai/claim
- BLOCKED 403 — iNaturalist Research Grade accuracy self-estimate, no figure cited from it: https://www.inaturalist.org/blog/89255-we-estimate-the-accuracy-of-research-grade-observations-to-be-95-correct
- SEARCH SUMMARY ONLY, NOT FETCHED — Uma, Fornaciari, Hovy, Paun, Plank & Poesio, 'Learning from Disagreement: A Survey', JAIR vol. 72 (2021): https://dl.acm.org/doi/10.1613/jair.1.12752
- SEARCH SUMMARY ONLY, NOT FETCHED — Gebru et al., 'Datasheets for Datasets', arXiv:1803.09010: https://arxiv.org/abs/1803.09010
- SEARCH SUMMARY ONLY, NOT FETCHED — STARD 2015 explanation and elaboration (partial verification bias): https://pmc.ncbi.nlm.nih.gov/articles/PMC5128957/
- SEARCH SUMMARY ONLY, NOT FETCHED — partial verification bias and test-result-based sampling, J Clin Epidemiol: https://www.jclinepi.com/article/S0895-4356(22)00035-X/fulltext
- SEARCH SUMMARY ONLY, NOT FETCHED — automation bias and anchoring in computational pathology, arXiv:2603.11821: https://arxiv.org/html/2603.11821v2
- SEARCH SUMMARY ONLY, NOT FETCHED — 'Bias in the Loop: How Humans Evaluate AI-Generated Suggestions', HDSR 8.2: https://hdsr.mitpress.mit.edu/pub/nrcn4h7d
- SEARCH SUMMARY ONLY, NON-PEER-REVIEWED — kappa paradox and Gwet's AC1: https://inter-rater-reliability.blogspot.com/2013/12/the-paradoxes-of-agreement-coefficients.html
- SEARCH SUMMARY ONLY, NOT FETCHED — Croissant 1.1 release notes, consent/licensing policy enforcement: https://mlcommons.org/2026/02/croissant-1-1-standard/
