#!/usr/bin/env bash
# Compiles the project into ./out using only the JDK (no Maven/Gradle needed).
set -e
rm -rf out
mkdir -p out
javac -d out $(find src/main/java -name "*.java")
echo "Build OK -> ./out"
