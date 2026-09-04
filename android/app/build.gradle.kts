import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

// Kotlin 2.3 removed the kotlinOptions block; jvmTarget now lives in the compilerOptions
// DSL. Using the old form is a hard error, not a deprecation warning.
kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

android {
    namespace = "org.tealeafai.screening"
    compileSdk = 36

    defaultConfig {
        applicationId = "org.tealeafai.screening"
        // LiteRT 2.1.6 declares minSdkVersion 23 in its own manifest; 24 is a safe floor.
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        // Only arm64 is shipped: the benchmark device is arm64-v8a and unused ABIs would
        // roughly triple the APK for no measurable benefit.
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
        }
        release {
            // No signing config is declared. Release signing requires credentials that must
            // never live in a repository, so only the debug APK is produced here.
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures { compose = true }

    // No androidResources.noCompress block: AGP has shipped ".tflite" as the FIRST entry of
    // its default no-compress extension list since 4.1, verified in the AGP builder jar.
    // Adding one would be cargo-cult.

    packaging {
        resources { excludes += setOf("/META-INF/{AL2.0,LGPL2.1}") }
    }

    testOptions {
        unitTests.isIncludeAndroidResources = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.activity:activity-compose:1.9.3")

    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")

    // CameraX 1.6.1 is the current stable line.
    val camerax = "1.4.1"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    implementation("androidx.exifinterface:exifinterface:1.3.7")

    // LiteRT, the maintained successor to org.tensorflow:tensorflow-lite (frozen at 2.17.0
    // since January 2025). The Java package is still org.tensorflow.lite -- only the Gradle
    // coordinate changed -- so imports are unaffected.
    implementation("com.google.ai.edge.litert:litert:2.1.6")

    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.robolectric:robolectric:4.14.1")
    testImplementation("androidx.test:core:1.6.1")

    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:rules:1.6.1")
    androidTestImplementation("androidx.test.uiautomator:uiautomator:2.3.0")
    androidTestImplementation(composeBom)
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
