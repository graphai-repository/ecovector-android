plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    id("maven-publish")
}

group = "io.graphai"
version = "0.1.0"

android {
    namespace = "io.graphai.ecovector"
    compileSdk = 36

    defaultConfig {
        minSdk = 24

        ndk {
            abiFilters += listOf("arm64-v8a")
        }

        externalNativeBuild {
            cmake {
                targets += listOf("ecovector")
                cppFlags += listOf("-std=c++17", "-O2")
                arguments += listOf("-DANDROID_STL=c++_shared")
            }
        }

        consumerProguardFiles("consumer-rules.pro")
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
        prefabPublishing = true
    }

    prefab {
        create("ecovector") {
            headers = "src/main/cpp"
        }
    }

    publishing {
        singleVariant("release") {
            withSourcesJar()
        }
    }
}

dependencies {
    implementation("com.tom-roush:pdfbox-android:2.0.27.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}

afterEvaluate {
    publishing {
        publications {
            register<MavenPublication>("release") {
                from(components["release"])
                groupId = "io.graphai"
                artifactId = "ecovector"
                version = "0.1.0"

                pom {
                    name.set("EcoVector")
                    description.set("Hybrid vector search SDK for Android — vector + BM25 + RRF ranking")
                    url.set("https://github.com/graphai-io/ecovector")
                    licenses {
                        license {
                            name.set("Apache-2.0")
                            url.set("https://www.apache.org/licenses/LICENSE-2.0")
                        }
                    }
                }
            }
        }
    }
}
