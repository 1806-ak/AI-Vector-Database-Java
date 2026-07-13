# VectorDB — Java Port

A line-for-line Java port of the original C++ **VectorDB** project. Same HNSW / KD-Tree /
Brute-Force engine, same REST API, same RAG pipeline over Ollama, same `index.html` frontend —
just running on the JVM instead of native code.

No external dependencies. No Maven/Gradle needed. The HTTP server uses the JDK's built-in
`com.sun.net.httpserver.HttpServer`, and the Ollama client uses the JDK's built-in
`java.net.http.HttpClient`.

This has been **compiled and smoke-tested** (all endpoints below were exercised end-to-end)
in the environment this was built in.

---

## Prerequisites

1. **JDK 17 or newer** (JDK 11+ technically works, 17+ recommended)
   - Windows: install [Eclipse Temurin](https://adoptium.net/) or `winget install EclipseAdoptium.Temurin.17.JDK`
   - Verify: `java -version` and `javac -version`
2. **Ollama** (same as the original project, only needed for the Documents/RAG tabs)
   - https://ollama.com
   - `ollama pull nomic-embed-text`
   - `ollama pull llama3.2`

You do **not** need MSYS2, g++, or any C++ toolchain for this version.

---

## Build & Run

### Linux / macOS
```bash
cd VectorDB-Java
chmod +x build.sh run.sh
./run.sh
```

### Windows (PowerShell or cmd)
```bat
cd VectorDB-Java
run.bat
```

Either script compiles everything into `./out` and starts the server. You should see:

```
=== VectorDB Engine (Java port) ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
```

Open **http://localhost:8080** — it's the same UI as the C++ version (search, PCA scatter
plot, document embedding, RAG chat).

**Important:** run from the `VectorDB-Java` folder (the one containing `index.html`), the
same way the C++ binary expected `index.html` next to it.

### Manual build/run (no scripts)
```bash
javac -d out $(find src/main/java -name "*.java")   # Linux/macOS
java -cp out com.vectordb.Main
```

---

## Project Structure

```
VectorDB-Java/
├── src/main/java/com/vectordb/
│   ├── Main.java            ← HTTP server + all REST routes (was the httplib section of main.cpp)
│   ├── VectorItem.java       ← struct VectorItem
│   ├── DistanceFn.java       ← the DistFn typedef, as a functional interface
│   ├── DistanceMetrics.java  ← euclidean / cosine / manhattan
│   ├── ScoredId.java         ← std::pair<float,int> equivalent, used by all 3 indexes
│   ├── BruteForce.java       ← class BruteForce
│   ├── KDTree.java           ← class KDTree
│   ├── HNSW.java             ← class HNSW (graph, insert, search, remove, getInfo)
│   ├── VectorDatabase.java   ← class VectorDB (demo 16D index over all 3 algos)
│   ├── DocItem.java          ← struct DocItem
│   ├── DocumentDB.java       ← class DocumentDB (HNSW over real Ollama embeddings)
│   ├── OllamaClient.java     ← class OllamaClient (uses java.net.http.HttpClient)
│   ├── TextChunker.java      ← chunkText()
│   ├── JsonUtil.java         ← jS / jVec / parseVec / extractStr / extractInt (hand-rolled JSON, like the original)
│   └── DemoData.java         ← loadDemo() — the 20 pre-loaded vectors
├── index.html                ← unchanged frontend (pure HTML/CSS/JS, calls the same REST API)
├── build.sh / build.bat
├── run.sh / run.bat
└── README.md
```

### What changed vs. the C++ version, and why

| C++ | Java | Note |
|---|---|---|
| `cpp-httplib` (`httplib.h`) | `com.sun.net.httpserver.HttpServer` | Built into the JDK — no dependency needed |
| `std::function<float(vec,vec)>` | `DistanceFn` functional interface | Same role, method references (`DistanceMetrics::cosine`) used as callbacks |
| `std::priority_queue` | `java.util.PriorityQueue` (with `Collections.reverseOrder()` for the max-heap cases) | Matches min/max-heap semantics 1:1 |
| `std::unordered_map` | `java.util.HashMap` | — |
| `std::mutex` / `lock_guard` | `java.util.concurrent.locks.ReentrantLock` | Same coarse per-database locking |
| Manual JSON string building (`jS`, `ostringstream`) | Same manual JSON building with `StringBuilder`/`String.format` | Kept identical, byte-for-byte compatible output — no JSON library needed |
| `httplib::Client` → Ollama | `java.net.http.HttpClient` → Ollama | Same `/api/tags`, `/api/embeddings`, `/api/generate` calls |

The REST API surface is **identical** to the original — every endpoint, method, path, query
parameter, and JSON field name matches, so `index.html` works completely unmodified and any
existing `curl` scripts against the C++ server work unmodified too.

---

## REST API Reference

Same as the original project:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/search?v=f1,f2,...&k=5&metric=cosine&algo=hnsw` | K-NN search |
| `POST` | `/insert` | Insert a demo vector |
| `DELETE` | `/delete/:id` | Delete by ID |
| `GET` | `/items` | List all demo vectors |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | Compare all 3 algorithms |
| `GET` | `/hnsw-info` | HNSW graph structure and layer stats |
| `GET` | `/stats` | Database statistics |
| `POST` | `/doc/insert` | `{"title":"...","text":"..."}` — embed and store document |
| `GET` | `/doc/list` | List all stored documents |
| `DELETE` | `/doc/delete/:id` | Delete document chunk |
| `POST` | `/doc/search` | Fast retrieval preview for the UI |
| `POST` | `/doc/ask` | `{"question":"...","k":3}` — full RAG: retrieve + generate |
| `GET` | `/status` | Ollama status and model info |

---

## Verified during build

These were run against the compiled server and confirmed working:
- `/stats`, `/items` (20 demo vectors present)
- `/search` with `algo=hnsw`, `algo=kdtree`, `algo=bruteforce` — **all three return identical
  top-3 rankings and distances**, confirming the HNSW/KD-Tree ports are correct
- `/benchmark` — timing comparison across all 3 algorithms
- `/hnsw-info` — graph structure (nodes/edges per layer)
- `POST /insert` and `DELETE /delete/:id` — insert/delete round-trip verified (21 → 20 items)
- `/status` — correctly reports Ollama offline/online
- `/` — serves `index.html` correctly
- `/doc/list` — empty list on fresh start
- `OPTIONS` preflight — returns 204 with CORS headers
- Unknown routes — return 404

The Ollama-dependent endpoints (`/doc/insert`, `/doc/ask`, `/doc/search`) use the same
`/api/embeddings` and `/api/generate` calls as the original and only need a running
`ollama serve` with the two models pulled, exactly like the C++ version — this part of the
logic is a direct line-for-line port and wasn't independently re-verified against a live
Ollama instance in this environment (no network/Ollama available here). Test it against your
local Ollama the same way you would have with the C++ build.

---

## License

MIT — same as the original.
