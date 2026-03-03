# EcoVector SDK - Consumer ProGuard Rules
# Preserve JNI native methods
-keepclasseswithmembernames class io.graphai.ecovector.** {
    native <methods>;
}

# Preserve Public API
-keep class io.graphai.ecovector.EcoVector { *; }
-keep class io.graphai.ecovector.EcoVectorConfig { *; }
-keep class io.graphai.ecovector.SearchResult { *; }
-keep class io.graphai.ecovector.TextCleaner { *; }
-keep class io.graphai.ecovector.PdfTextExtractor { *; }
