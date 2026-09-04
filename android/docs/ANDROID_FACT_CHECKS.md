# Adversarial Fact-Checks of Load-Bearing Claims

Each claim was re-verified against official sources by an agent instructed to
assume it was wrong until proven otherwise.


## [PARTLY_WRONG] The exact current Gradle dependency coordinates and version for running .tflite on Android, and whether org.tensorflow:tensorflow-lite is superseded by com.google.ai.edge.litert. Previous researcher reported: implementation("com.google.ai.edge.litert:litert:2.1.6"), google() required because litert is NOT on Maven Central, minSdk>=23, pulls litert-api:2.1.6 transitively, no noCompress needed, and recommended the Interpreter API on CPU+XNNPACK over CompiledModel.

**Correction:** CORE CLAIM CONFIRMED — I verified it down to the bytecode, not just the prose.

CONFIRMED (primary sources):
1. `com.google.ai.edge.litert:litert:2.1.6` IS current. https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/maven-metadata.xml shows <latest>2.1.6</latest>, <release>2.1.6</release>, lastUpdated 2026-07-01 22:13:38 UTC — 19 days before today. Not stale.
2. NOT on Maven Central — CONFIRMED, and the researcher was right where the official docs are wrong. repo1.maven.org/maven2/com/google/ai/edge/litert/litert/maven-metadata.xml returns HTTP 404; Maven Central solrsearch for g:com.google.ai.edge.litert returns numFound:0. `google()` is genuinely REQUIRED. NOTE: the official page https://developers.google.com/edge/litert/android/development contradicts itself and is STALE — it says "the AAR hosted at MavenCentral" and shows `implementation 'com.google.ai.edge.litert:+'`. That text is factually wrong; the migration guide correctly says maven.google.com. Do not follow the development page here.
3. Transitive dep CONFIRMED — litert-2.1.6.pom (packaging aar) has exactly one compile dependency: com.google.ai.edge.litert:litert-api:2.1.6.
4. minSdk 23 CONFIRMED from the AAR itself: AndroidManifest.xml inside litert-2.1.6.aar contains `<uses-sdk android:minSdkVersion="23"/>`. minSdk=24 is safe.
5. tensorflow-lite IS superseded CONFIRMED. org.tensorflow:tensorflow-lite on Maven Central tops out at 2.17.0 with lastUpdated 20250109 — ~18 months frozen. The LiteRT repo self-describes as "successor to TensorFlow Lite" and the migration guide says to replace the dependency.
6. No noCompress needed CONFIRMED — AGP 4.1+ adds .tflite to the noCompress list by default.
7. API surface CONFIRMED via javap on the real jars. litert:2.1.6 ships org/tensorflow/lite/Interpreter.class (12 classes, org.tensorflow.lite only); litert-api:2.1.6 ships com/google/ai/edge/litert/CompiledModel + Environment + Accelerator + TensorBuffer. All six CompiledModel.create() overloads, createInputBuffers/createOutputBuffers, run(List,List), Options(vararg Accelerator), Options.CPU, and Environment.getAvailableAccelerators():Set<Accelerator> exist exactly as reported. Interpreter.Options setNumThreads/setUseXNNPACK/setCancellable/setRuntime all exist exactly as reported.
8. NNAPI and CompatibilityList absent CONFIRMED — zero classes matching nnapi/CompatibilityList/GpuDelegate in either jar. Native libs are only libLiteRt.so + libLiteRtClGlAccelerator.so (arm64-v8a, armeabi-v7a, x86_64). Also confirms litert-gpu is dead: group-index.xml shows litert-gpu stops at 1.4.2, only litert and litert-api reached 2.1.6.

TWO DEFECTS:

A) MATERIAL OMISSION — recommending the Interpreter API without flagging upstream deprecation. The LiteRT README (raw.githubusercontent.com/google-ai-edge/LiteRT/main/README.md) states verbatim: "MUST USE: The Compiled Model API for all new kotlin and C++ native execution tasks. DO NOT USE: `tflite::Interpreter`, `InterpreterBuilder`, or manual delegate creation. The legacy Interpreter API is strictly deprecated for new features." The researcher recommended precisely the API upstream tells you not to use for new code, and never mentioned this. Fair caveat (my inference, not doc text): the DO-NOT-USE list names C++ symbols, and the Java org.tensorflow.lite.Interpreter still ships and works in 2.1.6 — so the researcher's code will run. But "strictly deprecated for new features" applies to Kotlin per the MUST-USE clause, and this should have been disclosed as a tradeoff rather than presented as the maintained path.

B) WRONG SIGNATURE — CompiledModel.CpuOptions. Reported as `CpuOptions(numThreads = 4, xn...)` implying a Boolean xnnpack parameter. Actual javap output: `CompiledModel$CpuOptions(java.lang.Integer, java.lang.Integer, java.lang.String)` plus a no-arg `CpuOptions()`. There is NO Boolean parameter — the second param is Integer, the third String. Any code written against the reported signature will not compile. I did not decode the Kotlin parameter names, so I cannot state what the three params are called; use the no-arg constructor plus named copy() or verify names before use.

BOTTOM LINE: the Gradle block as written is correct and will build. Keep `implementation("com.google.ai.edge.litert:litert:2.1.6")` with `google()` in repositories. The Interpreter-API recommendation is defensible for a 224x224x3 → 7-class model but must be labeled as using an API upstream calls deprecated for new features, and the CpuOptions signature must be corrected.

**Source:** https://dl.google.com/dl/android/maven2/com/google/ai/edge/litert/litert/maven-metadata.xml


## [CONFIRMED] The exact Gradle configuration required to stop .tflite assets being compressed (androidResources noCompress / aaptOptions), and what breaks if it is omitted. Previous researcher asserted: "NO noCompress block needed — AGP already excludes .tflite".

**Correction:** The core assertion is CORRECT, and I verified it in primary source (AGP bytecode), not just docs. But the previous write-up is thin on the "what breaks" half and omits caveats that can bite this project.

=== 1. WHAT THE OFFICIAL DOC STATES ===
Fetched https://developers.google.com/edge/litert/android/metadata/codegen (note: ai.google.dev/edge/litert/* now 301-redirects to developers.google.com/edge/litert/* — I hit that redirect directly). Exact quote from the page:
  "starting from version 4.1 of the Android Gradle plugin, .tflite will be added to the noCompress list by default"
and it shows the legacy block it replaces: aaptOptions { noCompress "tflite" }

=== 2. PRIMARY VERIFICATION (I did not take the doc's word for it) ===
I downloaded AGP's builder artifact from Google Maven and disassembled it with javap:
  https://dl.google.com/dl/android/maven2/com/android/tools/build/builder/8.13.0/builder-8.13.0.jar
  https://dl.google.com/dl/android/maven2/com/android/tools/build/builder/9.3.0/builder-9.3.0.jar
In com.android.builder.packaging.PackagingUtils, the private method
  getAllNoCompressExtensions(ImmutableList.Builder, Collection<String>, NativeLibrariesPackagingMode, DexPackagingMode)
does, unconditionally:
  builder.addAll(DEFAULT_AAPT_NO_COMPRESS_EXTENSIONS);   // .jpg .jpeg .png .gif .opus .wav .mp3 .ogg .aac .mp4 .webp .webm .mkv etc.
  builder.add(".tflite");                                 // <-- hardcoded, no flag, no opt-in
  // then conditionally ".so" / ".dex", then finally the user's androidResources.noCompress collection
This is byte-identical in AGP 8.13.0 and in AGP 9.3.0 (current release, July 2026). So the exemption is still live today, not a legacy behaviour.

That method feeds all three packaging paths (verified by javap on the same class + grep over gradle-8.13.0.jar):
  - getNoCompressPredicate(...)        -> used by PackageAndroidArtifact and CompressAssetsTask (APK assets)
  - getNoCompressPredicateForJavaRes(...) -> java resources
  - getNoCompressGlobsForBundle(...)   -> the uncompressedGlob list handed to bundletool, so AAB/Play delivery is covered too
Conclusion: for a file literally named *.tflite in app/src/main/assets/, NO Gradle block is needed for APK or AAB. Confirmed.

=== 3. CAVEAT THE PREVIOUS RESEARCHER MISSED (matching is by EXTENSION SUFFIX ONLY) ===
The predicate is getNoCompressPredicateForExtensions, whose lambda compiles to kotlin.text.StringsKt.endsWith(path, ext, /*ignoreCase=*/true). So:
  - "model.tflite" / "MODEL.TFLITE" -> exempt (case-insensitive suffix match).
  - "model.lite", "model.bin", "model.litertlm", "model.tflite.gz", or a model with no extension -> NOT exempt, silently deflated.
Only ".tflite" is hardcoded — I grepped the whole builder jar; ".lite" and ".litertlm" are absent from the default list. Keep the asset named exactly model.tflite (the previous researcher's recommended path app/src/main/assets/model.tflite is fine).

=== 4. WHAT ACTUALLY BREAKS IF THE FILE IS COMPRESSED ===
Compression only matters if you memory-map. Official statement, from https://developer.android.com/ndk/reference/group/asset — AAsset_openFileDescriptor: "Returns < 0 if direct fd access is not possible (for example, if the asset is compressed)." The Java equivalent AssetManager.openFd() throws java.io.FileNotFoundException with the message "This file can not be opened as a file descriptor; it is probably compressed" (I did NOT find that exact string on an official developer.android.com page I fetched — I only saw it in third-party bug reports, so treat the wording as unverified; the < 0 / failure behaviour is what the official NDK doc states).
Consequence: the classic mmap loader (assets.openFd(name) -> FileInputStream.channel.map(READ_ONLY, startOffset, declaredLength) -> Interpreter(MappedByteBuffer)) fails at load time, not inference time. CompiledModel.create(context.assets, "mymodel.tflite", ...) — the signature shown on https://developers.google.com/edge/litert/next/android_kotlin — takes the AssetManager itself; I did NOT read its implementation, so I can only INFER it uses the fd/mmap path and would fail the same way. If instead you read the asset with assets.open(name).readBytes() into a direct ByteBuffer, compression is irrelevant and nothing breaks (you just pay a copy). This distinction is not stated on any page I fetched — it follows from the AAsset_openFileDescriptor doc.

=== 5. IF YOU DO WANT THE BLOCK ANYWAY (belt-and-braces, harmless) ===
Verified against gradle-api-8.13.0.jar and gradle-api-9.3.0.jar via javap:
  com.android.build.api.dsl.AaptOptions carries @kotlin.Deprecated(message = "Renamed to AndroidResources", replaceWith = ReplaceWith("AndroidResources")) — so aaptOptions is deprecated, though NOT removed as of AGP 9.3.0. AGP 9.0.1 release notes (https://developer.android.com/build/releases/agp-9-0-0-release-notes) do not list aaptOptions among removed DSL items.
  com.android.build.api.dsl.AndroidResources declares: Collection<String> getNoCompress(); void noCompress(String); void noCompress(String...). There is NO setNoCompress on the interface (only on the internal impl), so the guaranteed-correct Kotlin DSL form is the method call:
    android { androidResources { noCompress("tflite") } }
  ("noCompress += \"tflite\"" also generally works since the getter returns a mutable collection, but the method call is the form actually declared in the API.)
  Also note AGP 9.3.0's AndroidResources dropped the `namespaced` property that 8.13.0 had — unrelated to this claim but a real 9.x delta.

=== 6. BOTTOM LINE FOR THIS PROJECT ===
Previous researcher's "NO noCompress block needed" — correct, and still correct on the newest AGP. Add nothing to app/build.gradle.kts for this. The only real risk is renaming the asset away from a .tflite suffix.

**Source:** https://developers.google.com/edge/litert/android/metadata/codegen


## [CONFIRMED] That getExternalFilesDir() content under /sdcard/Android/data/<pkg>/files/ is pullable via adb pull on a NON-ROOTED Android 16 device without extra permissions.

**Correction:** CORE CLAIM HOLDS. I tried to break it and could not. Two independent legs:

LEG 1 — "without extra permissions" (writing there). CONFIRMED by official docs, verbatim.
- AOSP Context.java javadoc on getExternalFilesDir(): "Starting in android.os.Build.VERSION_CODES#KITKAT, no permissions are required to read or write to the returned path; it's always accessible to the calling app." Also annotated @Nullable — the researcher's "null-check it" is correct.
  https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/master/core/java/android/content/Context.java
- https://developer.android.com/training/data-storage/app-specific : "On Android 4.4 (API level 19) or higher, your app doesn't need to request any storage-related permissions to access app-specific directories within external storage."

LEG 2 — "pullable via adb pull on non-rooted Android 16". CONFIRMED, but NOT by any developer.android.com page. No official doc states this. I found the actual enforcement code instead. The Android 11+ Android/data lockdown is enforced by MediaProvider's FUSE daemon in is_app_accessible_path():
    static bool is_app_accessible_path(struct fuse* fuse, const string& path, uid_t uid) {
        MediaProviderWrapper* mp = fuse->mp;
        if (uid < AID_APP_START || uid == MY_UID) {
            return true;
        }
        ...
AID_APP_START is 10000; adb shell runs as uid 2000 (shell), so the Android/data restriction is bypassed outright for adb before any package-ownership check runs. This is a UID bypass, not a documented API contract.
  https://raw.githubusercontent.com/aosp-mirror/platform_packages_providers_mediaprovider/master/jni/FuseDaemon.cpp

NOTHING CHANGED IN ANDROID 16. I read both behavior-changes pages end to end; neither contains any external-storage, scoped-storage, Android/data, getExternalFilesDir, or adb/shell file-access item.
  https://developer.android.com/about/versions/16/behavior-changes-all
  https://developer.android.com/about/versions/16/behavior-changes-16
The Android 11 restrictions (https://developer.android.com/about/versions/11/privacy/storage) are scoped to app-level access and ACTION_OPEN_DOCUMENT_TREE — they never mention adb or shell.

FOUR CORRECTIONS / FLAGS on the previous researcher's write-up:

1. WRONG COMMAND. "adb shell echo /sdcard/Android/data/<pkg>/files/research", presented as "Discover the real path rather than assuming (works on any device/user profile)", discovers nothing — echo prints its literal argument back. It verifies zero. The adjacent `adb shell ls -la <path>` is the command that actually verifies existence; drop the echo line.

2. GOOGLE'S ONLY EXPLICIT GUIDANCE FOR THIS EXACT USE CASE POINTS SOMEWHERE ELSE. https://developer.android.com/training/data-storage/use-cases says verbatim: "For test output, it's better to instead write to app-scoped storage that's readable by the shell. You can then pull that app-scoped directory. To determine which directory to pull from, call getExternalMediaDirs()." That is Android/media/<pkg>/, not Android/data/<pkg>/files/. So the officially-blessed pull target is getExternalMediaDirs(), and the researcher chose the undocumented-but-working one.

3. ...BUT DO NOT SWITCH ON THAT ADVICE — IT IS DEPRECATED. getExternalMediaDirs() is marked @Deprecated in AOSP Context.java: "These directories still exist and are scanned, but developers are encouraged to migrate to inserting content into a MediaStore collection directly, as any app can contribute new media to MediaStore with no permissions required, starting in ...VERSION_CODES#Q." The use-cases page recommending it is stale relative to the API reference. Recommendation: KEEP getExternalFilesDir(null). It is non-deprecated, permission-free, and reachable by adb. Its only downside is that reachability rests on a UID bypass rather than a documented promise. Also note Android/media is MediaStore-scanned (research JPEGs would land in the gallery); Android/data is not — another reason the researcher's choice is better for this project.

4. NOT EMPIRICALLY CONFIRMED ON THE TARGET HARDWARE. I attempted it: adb 1.0.41 / 37.0.0-14910828 is installed under the local user profile, but "adb devices -l" returns an empty list after a kill-server/start-server cycle — the SM-S916B is not currently attached. So the Android 16 / One UI verdict above is from AOSP source, not from the physical device. This matters because Samsung has historically added OEM restrictions on Android/data; per the AOSP code those target app UIDs (>=10000) via SAF/FUSE and cannot affect uid 2000, but I am flagging that as INFERENCE, not something I observed on the S23+. Run "adb shell ls -la /sdcard/Android/data/<applicationId>/files/" once the device is plugged in to close this out.

VERIFIED CORRECT IN THE WRITE-UP: the -a flag. I ran adb help locally; it prints "pull [-a] [-z ALGORITHM] [-Z] REMOTE... LOCAL / -a: preserve file timestamp and mode". (Note the help text also lists -q to suppress progress, and -Z to disable compression.) The recursive directory pull, the FileOutputStream.getFD().sync() durability step, Bitmap.compress(JPEG, 90, out), Environment.getExternalStorageState() vs MEDIA_MOUNTED, "add nothing to the manifest", "no Gradle change", and the run-as/"package not debuggable" fallback are all consistent with the official docs I read — I found no error in them. Note the directory will not exist until the app has actually run and created it, so a pull before first launch fails with a "does not exist" error rather than a permission error; do not misread that as the restriction biting.

**Source:** https://raw.githubusercontent.com/aosp-mirror/platform_packages_providers_mediaprovider/master/jni/FuseDaemon.cpp


## [PARTLY_WRONG] The correct handling of rotation for a CameraX still capture so the bitmap fed to the model is upright: use ImageCapture in-memory takePicture(OnImageCapturedCallback), read imageProxy.imageInfo.rotationDegrees, apply it with a Matrix to the bitmap from imageProxy.toBitmap(), keep targetRotation updated via OrientationEventListener, and do NOT rely on setTargetRotation alone (it only writes metadata). CameraX 1.6.1 / exifinterface 1.4.2.

**Correction:** The ROTATION ARCHITECTURE IS FULLY CONFIRMED against official sources. Verified verbatim:

1. setTargetRotation is metadata-only for ImageCapture (CONFIRMED). ImageCapture.java javadoc: "This will affect the EXIF rotation metadata in images saved by takePicture calls and the ImageInfo#getRotationDegrees() value of the ImageProxy returned by OnImageCapturedCallback." And onCaptureSuccess: the image "is provided as captured by the underlying ImageReader without rotation applied. The value in image.getImageInfo().getRotationDegrees() describes the magnitude of clockwise rotation, which if applied to the image will make it match the currently configured target rotation." Source: https://raw.githubusercontent.com/androidx/androidx/androidx-main/camera/camera-core/src/main/java/androidx/camera/core/ImageCapture.java

2. toBitmap() does NOT rotate (CONFIRMED, and this is the crux). ImageProxy.toBitmap() javadoc lists supported formats YUV_420_888 / JPEG / RGBA_8888 with no mention of rotation. The implementation in ImageUtil.createBitmapFromImageProxy dispatches JPEG/JPEG_R to createBitmapFromJpegImage, which is `byte[] bytes = jpegImageToJpegByteArray(imageProxy); Bitmap bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.length, null);` — BitmapFactory ignores EXIF orientation. So an explicit Matrix rotation is genuinely mandatory. Sources: .../androidx/camera/core/ImageProxy.java and .../androidx/camera/core/internal/utils/ImageUtil.java

3. Clockwise convention CONFIRMED. ImageInfo.getRotationDegrees(): "Returns the rotation needed to transform the image to the correct orientation. This is a clockwise rotation in degrees that needs to be applied to the image buffer." Value in {0, 90, 180, 270}; for JPEG it matches the EXIF rotation.

4. OrientationEventListener for locked-orientation activities CONFIRMED, with the exact when(orientation) -> Surface.ROTATION_* mapping and enable()/disable() in onStart/onStop, at https://developer.android.com/media/camera/camerax/orientation-rotation (note: the researcher's implied URL /orientations-and-rotations is a 404; the real page is /orientation-rotation, singular).

5. Versions CONFIRMED. androidx.camera 1.6.1 is latest stable, released May 06 2026 (1.6.0 Mar 25 2026; 1.7.0-alpha02 Jul 01 2026 is preview only). androidx.exifinterface:exifinterface:1.4.2 is latest stable, Dec 03 2025.

6. setOutputImageRotationEnabled is @RequiresApi(23) — CONFIRMED. Javadoc also documents "the output image will be a rotated ImageProxy and ImageInfo#getRotationDegrees() will return 0", a 10-15ms/frame overhead for 640x480 on a mid-range device, and that it is not allowed with OUTPUT_IMAGE_FORMAT_PRIVATE.

7. ImageProxy#close() is the app's responsibility — CONFIRMED.

8. CAPTURE_MODE_MAXIMIZE_QUALITY — CONFIRMED ("Optimizes capture pipeline to prioritize image quality over latency").

WHAT IS WRONG OR INCOMPLETE:

A. WRONG REASONING on crop order. The report says "Rotate first, then center-crop to square, then scale to 224x224. Cropping before rotating crops the wrong axis." The conclusion is harmless but the justification is false. This is my geometric inference, not a docs claim: a CENTERED square crop of side min(W,H) commutes with 90/180/270-degree rotation — rotate-then-center-crop and center-crop-then-rotate produce identical pixels. Order only matters for a non-square crop region or a non-90-multiple rotation. Do not let anyone reason from the stated rule; it will mislead on other crop geometries.

B. INCOMPLETE MANIFEST — will cause Google Play filtering. The report gives only `<uses-feature android:name="android.hardware.camera.any" android:required="false" />`. Official docs at https://developer.android.com/guide/topics/manifest/uses-feature-element state: "The CAMERA permission implies that your app also uses android.hardware.camera. A back camera is a required feature unless android.hardware.camera is declared with android:required="false"." So declaring only camera.any leaves an implied HARD requirement on android.hardware.camera. Add BOTH lines:
  <uses-feature android:name="android.hardware.camera.any" android:required="false" />
  <uses-feature android:name="android.hardware.camera" android:required="false" />
The `<uses-permission android:name="android.permission.CAMERA" />` part is correct, as is the claim that no storage permission is needed for the in-memory path.

C. UNDERSTATED BUG — actually strengthens the recommendation. The report calls the setOutputImageRotationEnabled issue "a recorded rotation bug". Per the release notes, it was fixed ONLY in 1.7.0-alpha01: "Fixed a bug in ImageAnalysis where images were not correctly rotated when output image rotation is enabled and the initial relative rotation is 0 degrees" (Id46c2, b/487160584). It is NOT fixed in any 1.6.x release, so it is live in the recommended stable 1.6.1. The advice to use explicit Matrix rotation is therefore more strongly justified than stated.

D. BUILD NOTE. CameraX moved its default minSdk from API 21 to API 23 in 1.5.0-rc01 (Ibdfca, b/380448311), carried into 1.6.x. The report's minSdk = 24 satisfies this, but the floor is 23 — a 1.6.1 dependency will not build against minSdk 21/22.

NOT VERIFIED IN THIS SESSION (do not treat as confirmed): the TFLite preprocessing specifics (raw [0,255] floats, RGB channel order, ByteBuffer.allocateDirect(1*224*224*3*4) sizing) were outside the camerax-orientation area and were not checked against LiteRT docs. The ByteBuffer arithmetic is self-consistent (602112 bytes) but the normalization claim depends on the user's specific model graph. Also not verified: the exact `.\gradlew.bat :app:assembleDebug` invocation and the AGP/JDK toolchain assertions.

**Source:** https://developer.android.com/media/camera/camerax/orientation-rotation


## [PARTLY_WRONG] The Android Gradle Plugin and Gradle versions compatible with JDK 21 and compileSdk 36, and that assembleDebug works without Android Studio or cmdline-tools — reported as: use AGP 8.13.0 + Gradle 8.14, do NOT use AGP 9.x (because Gradle 8.14 is cached, AGP 8.13 defaults to build-tools 35.0.0 which is installed while AGP 9.x defaults to 36.0.0 which is not, AGP 8.13 max API is 36.1, and AGP 9.0 flips many defaults), plus wrapper bootstrap, local.properties, build/test commands and report output paths.

**Correction:** The actionable core is CORRECT and I verified it on official docs and on the actual machine. The supporting rationale contains one non-differentiator, one significant omission, and is stale by three AGP minor releases.

VERIFIED CORRECT:
- AGP 8.13 compatibility table (https://developer.android.com/build/releases/agp-8-13-0-release-notes): Gradle min AND default 8.13; SDK Build Tools min/default 35.0.0; JDK min/default 17; NDK default 27.0.12077973; maximum API level 36.1. So compileSdk 36 is supported and build-tools 35.0.0 is indeed the default.
- Gradle 8.14 runs on JDK 21: Gradle's official matrix (https://docs.gradle.org/current/userguide/compatibility.html) lists JDK 21 "Running Gradle" as supported from 8.5 and after. JDK 24 requires 8.14+; JDK 25 requires 9.1.0+.
- assembleDebug without Android Studio: confirmed at https://developer.android.com/build/building-cmdline — "gradlew assembleDebug", output in project_name/module_name/build/outputs/apk/, already signed with the debug key and zipaligned. The page explicitly offers Studio only as an alternative, never a requirement.
- All test commands and ALL FOUR report paths match https://developer.android.com/studio/test/command-line verbatim: unit HTML build/reports/tests/, unit XML build/test-results/, instrumented HTML build/reports/androidTests/connected/, instrumented XML build/outputs/androidTest-results/connected/. The "cAT" abbreviation for connectedAndroidTest and the --tests '*.sampleTestMethod' filter are both official. The -Pandroid.testInstrumentationRunnerArguments.class= and .size=medium syntax is official (also documented under advanced-test-setup and the benchmarking pages).
- AGP 9.0 defaults (https://developer.android.com/build/releases/agp-9-0-0-release-notes): all five flipped properties the researcher named are real, and the runtime dependency on KGP 2.2.10 is confirmed.

MACHINE STATE I VERIFIED DIRECTLY: gradle-8.14-all is the ONLY cached wrapper distribution and the exact claimed gradle.bat path resolves True; build-tools installed are 35.0.0 and 36.1.0 (36.0.0 is genuinely absent); platforms android-35/36/36.1; adb.exe present at the claimed path; cmdline-tools directory does NOT exist; JDK is Temurin 21.0.11.

DEFECTS:
1. STALE. As of today (2026-07-20) the current AGP is 9.3.0, released July 2026 (https://developer.android.com/build/releases/agp-9-3-0-release-notes), requiring Gradle 9.5.0 and supporting max API level 37. AGP 9.2.0/9.2.1 (April 2026) requires Gradle 9.4.1 and already raised max API to 37.0. The report is written as though AGP 9.0 were the frontier.
2. NON-DIFFERENTIATOR. "AGP 8.13 max API is 36.1, so compileSdk 36 is fully supported" is true, but AGP 9.0's maximum API level is ALSO 36.1. This distinguishes nothing between the two options and should not be listed as a reason to prefer 8.13.
3. MISSED THE ACTUAL HARD BLOCKER. The decisive offline argument against AGP 9.x is that AGP 9.0 requires Gradle 9.1.0 minimum (9.2 needs 9.4.1, 9.3 needs 9.5.0), and only gradle-8.14-all is in the wrapper cache — that forces a mandatory full Gradle distribution download. The build-tools 36.0.0 argument the report leads with is weaker, because it is trivially defeated by pinning buildToolsVersion = "36.1.0" (36.1.0 IS installed).
4. UNDERSPECIFIED GRADLE FLOOR. AGP 8.13's minimum and default Gradle is 8.13, not 8.14. Gradle 8.14 is a valid choice since it exceeds the floor, so the recommendation works, but the report never states the real requirement.
5. INCOMPLETE FLIP LIST. AGP 9.0 also flips android.useAndroidx (false->true), android.default.androidx.test.runner (false->true), android.enableAppCompileTimeRClass, android.onlyEnableUnitTestForTheTestedBuildType, android.r8.optimizedResourceShrinking, android.r8.strictFullModeForKeepRules, android.defaults.buildfeatures.resvalues (true->false), android.defaults.buildfeatures.shaders (true->false), and android.dependency.useConstraints (true->false). Also, the targetSdk change is governed by the property android.sdk.defaultTargetSdkToCompileSdkIfUnset, which the report described by effect but did not name.
6. INFERENCE FLAGGED, NOT DOC-STATED. No official Android page states "AGP 8.13 supports JDK 21." https://developer.android.com/build/jdks states only that AGP 8.x requires JDK 17, expressed as a MINIMUM with no documented maximum. JDK 21 viability follows from Gradle 8.14's own compatibility matrix, not from an AGP statement. This is a sound inference, not a quoted fact.

BOTTOM LINE: keep AGP 8.13.0 + Gradle 8.14 + JDK 21 + compileSdk 36. It is correct and fully offline-capable on this machine. Fix the rationale: cite Gradle 9.1.0+ not being cached as the reason to avoid AGP 9.x, drop the max-API-36.1 argument, and note that AGP 9.3 is the current release if a future online upgrade is ever considered.

**Source:** https://developer.android.com/build/releases/agp-8-13-0-release-notes
