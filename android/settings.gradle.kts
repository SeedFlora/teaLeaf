pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        // google() is REQUIRED, not optional. The LiteRT artifacts are published only to
        // Google Maven; repo1.maven.org returns 404 for com.google.ai.edge.litert. The
        // official developer page claiming they are "hosted at MavenCentral" is wrong,
        // and following it produces an unresolvable dependency.
        google()
        mavenCentral()
    }
}

rootProject.name = "TeaLeafAI"
include(":app")
