plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
    id("maven-publish")
}

group = "io.graphai"
version = "0.1.0"

android {
    namespace = "io.graphai.ecovector.benchmark"
    compileSdk = 36

    defaultConfig {
        minSdk = 24

        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                targets += listOf("ecovector-benchmark")
                cppFlags += listOf("-std=c++17", "-O2")
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DECOVECTOR_CPP_DIR=${project(":ecovector").projectDir}/src/main/cpp",
                    "-DECOVECTOR_SO_DIR=${project(":ecovector").buildDir}/intermediates/cmake/debug/obj/arm64-v8a"
                )
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }

    buildFeatures {
        compose = true
    }

    publishing {
        singleVariant("release") {
            withSourcesJar()
        }
    }
}

dependencies {
    implementation(project(":ecovector"))

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui.tooling.preview)

    testImplementation(libs.junit)
}

afterEvaluate {
    publishing {
        publications {
            register<MavenPublication>("release") {
                from(components["release"])
                groupId = "io.graphai"
                artifactId = "ecovector-benchmark"
                version = "0.1.0"
            }
        }
    }
}
