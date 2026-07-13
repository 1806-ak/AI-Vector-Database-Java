package com.vectordb;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Executors;

public class Main implements HttpHandler {

    static final int DIMS = 16;

    final VectorDatabase db = new VectorDatabase(DIMS);
    final DocumentDB docDB = new DocumentDB();
    final OllamaClient ollama = new OllamaClient();

    public static void main(String[] args) throws IOException {
        Main app = new Main();
        DemoData.loadDemo(app.db);

        boolean ollamaUp = app.ollama.isAvailable();
        System.out.println("=== VectorDB Engine (Java port) ===");
        System.out.println("http://localhost:8080");
        System.out.println(app.db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"));
        if (ollamaUp) {
            System.out.println("  embed model: " + app.ollama.embedModel + "  gen model: " + app.ollama.genModel);
        }

        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", 8080), 0);
        server.createContext("/", app);
        server.setExecutor(Executors.newFixedThreadPool(8));
        server.start();
    }

    // ── HELPERS ──────────────────────────────────────────────────────

    private static void cors(HttpExchange ex) {
        ex.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
        ex.getResponseHeaders().add("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
        ex.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type");
    }

    private static void sendJson(HttpExchange ex, int status, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "application/json");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static void sendHtml(HttpExchange ex, int status, String body) throws IOException {
        byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
        ex.getResponseHeaders().set("Content-Type", "text/html");
        ex.sendResponseHeaders(status, bytes.length);
        try (OutputStream os = ex.getResponseBody()) {
            os.write(bytes);
        }
    }

    private static String readBody(HttpExchange ex) throws IOException {
        try (InputStream is = ex.getRequestBody()) {
            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            byte[] buf = new byte[4096];
            int n;
            while ((n = is.read(buf)) != -1) bos.write(buf, 0, n);
            return bos.toString(StandardCharsets.UTF_8);
        }
    }

    private static Map<String, String> parseQuery(String rawQuery) {
        Map<String, String> map = new HashMap<>();
        if (rawQuery == null || rawQuery.isEmpty()) return map;
        for (String pair : rawQuery.split("&")) {
            int idx = pair.indexOf('=');
            if (idx == -1) continue;
            try {
                String k = URLDecoder.decode(pair.substring(0, idx), "UTF-8");
                String v = URLDecoder.decode(pair.substring(idx + 1), "UTF-8");
                map.put(k, v);
            } catch (Exception ignored) {
            }
        }
        return map;
    }

    private static int countSpaces(String s) {
        int c = 0;
        for (int i = 0; i < s.length(); i++) if (s.charAt(i) == ' ') c++;
        return c;
    }

    // ── ROUTING ──────────────────────────────────────────────────────

    @Override
    public void handle(HttpExchange exchange) {
        try {
            route(exchange);
        } catch (Exception e) {
            e.printStackTrace();
            try {
                sendJson(exchange, 500, "{\"error\":" + JsonUtil.jS(String.valueOf(e.getMessage())) + "}");
            } catch (IOException ignored) {
            }
        } finally {
            exchange.close();
        }
    }

    private void route(HttpExchange exchange) throws IOException {
        cors(exchange);
        String method = exchange.getRequestMethod();
        String path = exchange.getRequestURI().getPath();
        String rawQuery = exchange.getRequestURI().getRawQuery();

        if ("OPTIONS".equalsIgnoreCase(method)) {
            exchange.sendResponseHeaders(204, -1);
            return;
        }

        if ("GET".equalsIgnoreCase(method) && path.equals("/")) {
            serveIndex(exchange);
            return;
        }
        if ("GET".equalsIgnoreCase(method) && path.equals("/search")) {
            handleSearch(exchange, parseQuery(rawQuery));
            return;
        }
        if ("POST".equalsIgnoreCase(method) && path.equals("/insert")) {
            handleInsert(exchange);
            return;
        }
        if ("DELETE".equalsIgnoreCase(method) && path.matches("/delete/\\d+")) {
            handleDelete(exchange, path);
            return;
        }
        if ("GET".equalsIgnoreCase(method) && path.equals("/items")) {
            handleItems(exchange);
            return;
        }
        if ("GET".equalsIgnoreCase(method) && path.equals("/benchmark")) {
            handleBenchmark(exchange, parseQuery(rawQuery));
            return;
        }
        if ("GET".equalsIgnoreCase(method) && path.equals("/hnsw-info")) {
            handleHnswInfo(exchange);
            return;
        }
        if ("POST".equalsIgnoreCase(method) && path.equals("/doc/insert")) {
            handleDocInsert(exchange);
            return;
        }
        if ("DELETE".equalsIgnoreCase(method) && path.matches("/doc/delete/\\d+")) {
            handleDocDelete(exchange, path);
            return;
        }
        if ("GET".equalsIgnoreCase(method) && path.equals("/doc/list")) {
            handleDocList(exchange);
            return;
        }
        if ("POST".equalsIgnoreCase(method) && path.equals("/doc/search")) {
            handleDocSearch(exchange);
            return;
        }
        if ("POST".equalsIgnoreCase(method) && path.equals("/doc/ask")) {
            handleDocAsk(exchange);
            return;
        }
        if ("GET".equalsIgnoreCase(method) && path.equals("/status")) {
            handleStatus(exchange);
            return;
        }
        if ("GET".equalsIgnoreCase(method) && path.equals("/stats")) {
            handleStats(exchange);
            return;
        }

        sendJson(exchange, 404, "{\"error\":\"not found\"}");
    }

    // ── HANDLERS ─────────────────────────────────────────────────────

    private void serveIndex(HttpExchange exchange) throws IOException {
        Path p = Path.of("index.html");
        if (!Files.exists(p)) {
            exchange.sendResponseHeaders(404, -1);
            return;
        }
        String content = Files.readString(p, StandardCharsets.UTF_8);
        sendHtml(exchange, 200, content);
    }

    private void handleSearch(HttpExchange exchange, Map<String, String> params) throws IOException {
        float[] q = JsonUtil.parseVec(params.getOrDefault("v", ""));
        if (q.length != DIMS) {
            sendJson(exchange, 200, "{\"error\":\"need " + DIMS + "D vector\"}");
            return;
        }
        int k = 5;
        try {
            k = Integer.parseInt(params.getOrDefault("k", "5"));
        } catch (Exception ignored) {
        }
        String metric = params.getOrDefault("metric", "cosine");
        if (metric.isEmpty()) metric = "cosine";
        String algo = params.getOrDefault("algo", "hnsw");
        if (algo.isEmpty()) algo = "hnsw";

        VectorDatabase.SearchOut out = db.search(q, k, metric, algo);
        StringBuilder ss = new StringBuilder();
        ss.append("{\"results\":[");
        for (int i = 0; i < out.hits.size(); i++) {
            if (i > 0) ss.append(',');
            VectorDatabase.Hit h = out.hits.get(i);
            ss.append("{\"id\":").append(h.id)
                    .append(",\"metadata\":").append(JsonUtil.jS(h.meta))
                    .append(",\"category\":").append(JsonUtil.jS(h.cat))
                    .append(",\"distance\":").append(String.format(Locale.US, "%.6f", h.dist))
                    .append(",\"embedding\":").append(JsonUtil.jVec(h.emb)).append('}');
        }
        ss.append("],\"latencyUs\":").append(out.us)
                .append(",\"algo\":").append(JsonUtil.jS(out.algo))
                .append(",\"metric\":").append(JsonUtil.jS(out.metric)).append('}');
        sendJson(exchange, 200, ss.toString());
    }

    private void handleInsert(HttpExchange exchange) throws IOException {
        String body = readBody(exchange);
        String meta = JsonUtil.extractStr(body, "metadata");
        String cat = JsonUtil.extractStr(body, "category");
        String arrRaw = JsonUtil.extractArrRaw(body, "embedding");
        float[] emb = arrRaw == null ? new float[0] : JsonUtil.parseVec(arrRaw);
        if (meta.isEmpty() || emb.length == 0 || emb.length != DIMS) {
            sendJson(exchange, 200, "{\"error\":\"invalid body\"}");
            return;
        }
        int id = db.insert(meta, cat, emb, DistanceMetrics::cosine);
        sendJson(exchange, 200, "{\"id\":" + id + "}");
    }

    private void handleDelete(HttpExchange exchange, String path) throws IOException {
        int id = Integer.parseInt(path.substring(path.lastIndexOf('/') + 1));
        boolean ok = db.remove(id);
        sendJson(exchange, 200, "{\"ok\":" + ok + "}");
    }

    private void handleItems(HttpExchange exchange) throws IOException {
        List<VectorItem> items = db.all();
        StringBuilder ss = new StringBuilder("[");
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) ss.append(',');
            VectorItem v = items.get(i);
            ss.append("{\"id\":").append(v.id)
                    .append(",\"metadata\":").append(JsonUtil.jS(v.metadata))
                    .append(",\"category\":").append(JsonUtil.jS(v.category))
                    .append(",\"embedding\":").append(JsonUtil.jVec(v.emb)).append('}');
        }
        ss.append(']');
        sendJson(exchange, 200, ss.toString());
    }

    private void handleBenchmark(HttpExchange exchange, Map<String, String> params) throws IOException {
        float[] q = JsonUtil.parseVec(params.getOrDefault("v", ""));
        if (q.length != DIMS) {
            sendJson(exchange, 200, "{\"error\":\"need " + DIMS + "D vector\"}");
            return;
        }
        int k = 5;
        try {
            k = Integer.parseInt(params.getOrDefault("k", "5"));
        } catch (Exception ignored) {
        }
        String metric = params.getOrDefault("metric", "cosine");
        if (metric.isEmpty()) metric = "cosine";
        VectorDatabase.BenchOut b = db.benchmark(q, k, metric);
        String ss = "{\"bruteforceUs\":" + b.bfUs + ",\"kdtreeUs\":" + b.kdUs
                + ",\"hnswUs\":" + b.hnswUs + ",\"itemCount\":" + b.n + '}';
        sendJson(exchange, 200, ss);
    }

    private void handleHnswInfo(HttpExchange exchange) throws IOException {
        HNSW.GraphInfo gi = db.hnswInfo();
        StringBuilder ss = new StringBuilder();
        ss.append("{\"topLayer\":").append(gi.topLayer).append(",\"nodeCount\":").append(gi.nodeCount)
                .append(",\"nodesPerLayer\":[");
        for (int i = 0; i < gi.nodesPerLayer.size(); i++) {
            if (i > 0) ss.append(',');
            ss.append(gi.nodesPerLayer.get(i));
        }
        ss.append("],\"edgesPerLayer\":[");
        for (int i = 0; i < gi.edgesPerLayer.size(); i++) {
            if (i > 0) ss.append(',');
            ss.append(gi.edgesPerLayer.get(i));
        }
        ss.append("],\"nodes\":[");
        for (int i = 0; i < gi.nodes.size(); i++) {
            if (i > 0) ss.append(',');
            HNSW.NV n = gi.nodes.get(i);
            ss.append("{\"id\":").append(n.id).append(",\"metadata\":").append(JsonUtil.jS(n.metadata))
                    .append(",\"category\":").append(JsonUtil.jS(n.category)).append(",\"maxLyr\":").append(n.maxLyr).append('}');
        }
        ss.append("],\"edges\":[");
        for (int i = 0; i < gi.edges.size(); i++) {
            if (i > 0) ss.append(',');
            HNSW.EV e = gi.edges.get(i);
            ss.append("{\"src\":").append(e.src).append(",\"dst\":").append(e.dst).append(",\"lyr\":").append(e.lyr).append('}');
        }
        ss.append("]}");
        sendJson(exchange, 200, ss.toString());
    }

    private void handleDocInsert(HttpExchange exchange) throws IOException {
        String body = readBody(exchange);
        String title = JsonUtil.extractStr(body, "title");
        String text = JsonUtil.extractStr(body, "text");
        if (title.isEmpty() || text.isEmpty()) {
            sendJson(exchange, 200, "{\"error\":\"need title and text\"}");
            return;
        }

        List<String> chunks = TextChunker.chunkText(text, 250, 30);
        List<Integer> ids = new ArrayList<>();

        for (int i = 0; i < chunks.size(); i++) {
            float[] emb = ollama.embed(chunks.get(i));
            if (emb.length == 0) {
                sendJson(exchange, 200,
                        "{\"error\":\"Ollama unavailable. Install from https://ollama.com then run: "
                                + "ollama pull nomic-embed-text && ollama pull llama3.2\"}");
                return;
            }
            String chunkTitle = chunks.size() > 1
                    ? title + " [" + (i + 1) + "/" + chunks.size() + "]"
                    : title;
            ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
        }

        StringBuilder ss = new StringBuilder();
        ss.append("{\"ids\":[");
        for (int i = 0; i < ids.size(); i++) {
            if (i > 0) ss.append(',');
            ss.append(ids.get(i));
        }
        ss.append("],\"chunks\":").append(chunks.size())
                .append(",\"dims\":").append(docDB.getDims()).append('}');
        sendJson(exchange, 200, ss.toString());
    }

    private void handleDocDelete(HttpExchange exchange, String path) throws IOException {
        int id = Integer.parseInt(path.substring(path.lastIndexOf('/') + 1));
        boolean ok = docDB.remove(id);
        sendJson(exchange, 200, "{\"ok\":" + ok + "}");
    }

    private void handleDocList(HttpExchange exchange) throws IOException {
        List<DocItem> docs = docDB.all();
        StringBuilder ss = new StringBuilder("[");
        for (int i = 0; i < docs.size(); i++) {
            if (i > 0) ss.append(',');
            DocItem d = docs.get(i);
            String preview = d.text.length() > 120 ? d.text.substring(0, 120) + "…" : d.text;
            int words = countSpaces(d.text) + 1;
            ss.append("{\"id\":").append(d.id)
                    .append(",\"title\":").append(JsonUtil.jS(d.title))
                    .append(",\"preview\":").append(JsonUtil.jS(preview))
                    .append(",\"words\":").append(words).append('}');
        }
        ss.append(']');
        sendJson(exchange, 200, ss.toString());
    }

    private void handleDocSearch(HttpExchange exchange) throws IOException {
        String body = readBody(exchange);
        String question = JsonUtil.extractStr(body, "question");
        int k = JsonUtil.extractInt(body, "k", 3);
        if (question.isEmpty()) {
            sendJson(exchange, 200, "{\"error\":\"need question\"}");
            return;
        }
        float[] qEmb = ollama.embed(question);
        if (qEmb.length == 0) {
            sendJson(exchange, 200, "{\"error\":\"Ollama unavailable\"}");
            return;
        }
        List<Map.Entry<Float, DocItem>> hits = docDB.search(qEmb, k);

        StringBuilder ss = new StringBuilder();
        ss.append("{\"contexts\":[");
        for (int i = 0; i < hits.size(); i++) {
            if (i > 0) ss.append(',');
            Map.Entry<Float, DocItem> h = hits.get(i);
            ss.append("{\"id\":").append(h.getValue().id)
                    .append(",\"title\":").append(JsonUtil.jS(h.getValue().title))
                    .append(",\"distance\":").append(String.format(Locale.US, "%.4f", h.getKey())).append('}');
        }
        ss.append("]}");
        sendJson(exchange, 200, ss.toString());
    }

    private void handleDocAsk(HttpExchange exchange) throws IOException {
        String body = readBody(exchange);
        String question = JsonUtil.extractStr(body, "question");
        int k = JsonUtil.extractInt(body, "k", 3);
        if (question.isEmpty()) {
            sendJson(exchange, 200, "{\"error\":\"need question\"}");
            return;
        }

        // Step 1: embed the question
        float[] qEmb = ollama.embed(question);
        if (qEmb.length == 0) {
            sendJson(exchange, 200, "{\"error\":\"Ollama unavailable\"}");
            return;
        }

        // Step 2: retrieve top-k relevant chunks
        List<Map.Entry<Float, DocItem>> hits = docDB.search(qEmb, k);

        // Step 3: build prompt
        StringBuilder ctx = new StringBuilder();
        for (int i = 0; i < hits.size(); i++) {
            DocItem d = hits.get(i).getValue();
            ctx.append('[').append(i + 1).append("] ").append(d.title).append(":\n")
                    .append(d.text).append("\n\n");
        }
        String prompt =
                "You are a helpful assistant. Answer the user's question directly. "
                        + "Use the provided context if it contains relevant information. "
                        + "If it doesn't, just use your own general knowledge. "
                        + "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
                        + "Just answer the question naturally.\n\n"
                        + "Context:\n" + ctx
                        + "Question: " + question + "\n\n"
                        + "Answer:";

        // Step 4: generate answer
        String answer = ollama.generate(prompt);

        // Step 5: return everything
        StringBuilder ss = new StringBuilder();
        ss.append("{\"answer\":").append(JsonUtil.jS(answer))
                .append(",\"model\":").append(JsonUtil.jS(ollama.genModel))
                .append(",\"contexts\":[");
        for (int i = 0; i < hits.size(); i++) {
            if (i > 0) ss.append(',');
            Map.Entry<Float, DocItem> h = hits.get(i);
            ss.append("{\"id\":").append(h.getValue().id)
                    .append(",\"title\":").append(JsonUtil.jS(h.getValue().title))
                    .append(",\"text\":").append(JsonUtil.jS(h.getValue().text))
                    .append(",\"distance\":").append(String.format(Locale.US, "%.4f", h.getKey())).append('}');
        }
        ss.append("],\"docCount\":").append(docDB.size()).append('}');
        sendJson(exchange, 200, ss.toString());
    }

    private void handleStatus(HttpExchange exchange) throws IOException {
        boolean up = ollama.isAvailable();
        String ss = "{\"ollamaAvailable\":" + up
                + ",\"embedModel\":" + JsonUtil.jS(ollama.embedModel)
                + ",\"genModel\":" + JsonUtil.jS(ollama.genModel)
                + ",\"docCount\":" + docDB.size()
                + ",\"docDims\":" + docDB.getDims()
                + ",\"demoDims\":" + DIMS
                + ",\"demoCount\":" + db.size() + '}';
        sendJson(exchange, 200, ss);
    }

    private void handleStats(HttpExchange exchange) throws IOException {
        String ss = "{\"count\":" + db.size()
                + ",\"dims\":" + DIMS
                + ",\"algorithms\":[\"bruteforce\",\"kdtree\",\"hnsw\"]"
                + ",\"metrics\":[\"euclidean\",\"cosine\",\"manhattan\"]}";
        sendJson(exchange, 200, ss);
    }
}
