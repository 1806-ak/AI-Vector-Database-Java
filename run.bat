@echo off
if not exist out (
  echo No build found, running build.bat first...
  call build.bat
)
java -cp out com.vectordb.Main
