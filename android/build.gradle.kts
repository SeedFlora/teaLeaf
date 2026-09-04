// Versions pinned against what is actually installed on the build host rather than the
// newest available: AGP 8.13 requires Gradle >= 8.13 (8.14 is the only cached wrapper),
// JDK >= 17 (Temurin 21.0.11 present), build-tools 35.0.0 (installed), and supports
// compileSdk 36. Newer AGP 9.x would demand Gradle 9.4+, which is not cached and would
// need a download this build host cannot be assumed to have.
plugins {
    id("com.android.application") version "8.13.0" apply false
    id("org.jetbrains.kotlin.android") version "2.3.0" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.3.0" apply false
}

