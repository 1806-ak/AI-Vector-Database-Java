@echo off
REM Compiles the project into .\out using only the JDK (no Maven/Gradle needed).
if exist out rmdir /s /q out
mkdir out
dir /s /b src\main\java\*.java > sources.txt
javac -d out @sources.txt
del sources.txt
echo Build OK -^> .\out
