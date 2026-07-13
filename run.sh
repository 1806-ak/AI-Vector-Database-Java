#!/usr/bin/env bash
# Run from the project root so index.html is found (matches the C++ version's behavior).
set -e
if [ ! -d out ]; then
  echo "No build found, running build.sh first..."
  ./build.sh
fi
java -cp out com.vectordb.Main
