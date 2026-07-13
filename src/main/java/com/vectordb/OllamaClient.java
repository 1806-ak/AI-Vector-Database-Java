package com.vectordb;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * Thin wrapper around the local Ollama REST API.
 * Install: https://ollama.com
 * Models:  ollama pull nomic-embed-text
 *          ollama pull llama3.2
 */
public class OllamaClient {
    private final String host;
    private final int port;
    public String embedModel = "nomic-embed-text";
    public String genModel = "llama3.2";

    public OllamaClient() {
        this("127.0.0.1", 11434);
    }

    public OllamaClient(String h, int p) {
        host = h;
        port = p;
    }

    private String esc(String s) {
        StringBuilder o = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"': o.append("\\\""); break;
                case '\\': o.append("\\\\"); break;
                case '\n': o.append("\\n"); break;
                case '\r': o.append("\\r"); break;
                case '\t': o.append("\\t"); break;
                default: o.append(c);
            }
        }
        return o.toString();
    }

    private float[] parseEmbedding(String body) {
        int p = body.indexOf("\"embedding\"");
        if (p == -1) return new float[0];
        p = body.indexOf('[', p);
        if (p == -1) return new float[0];
        int e = p + 1, depth = 1;
        while (e < body.length() && depth > 0) {
            char c = body.charAt(e);
            if (c == '[') depth++;
            else if (c == ']') depth--;
            e++;
        }
        return JsonUtil.parseVec(body.substring(p + 1, e - 1));
    }

    private String parseGenResponse(String body) {
        return JsonUtil.extractStr(body, "response");
    }

    public boolean isAvailable() {
        try {
            HttpClient cli = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(2))
                    .build();
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + host + ":" + port + "/api/tags"))
                    .timeout(Duration.ofSeconds(2))
                    .GET()
                    .build();
            HttpResponse<String> res = cli.send(req, HttpResponse.BodyHandlers.ofString());
            return res.statusCode() == 200;
        } catch (Exception e) {
            return false;
        }
    }

    /** Returns an empty array if Ollama is not running or the model isn't pulled. */
    public float[] embed(String text) {
        try {
            HttpClient cli = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(3))
                    .build();
            String body = "{\"model\":\"" + embedModel + "\",\"prompt\":\"" + esc(text) + "\"}";
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + host + ":" + port + "/api/embeddings"))
                    .timeout(Duration.ofSeconds(30))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> res = cli.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200) return new float[0];
            return parseEmbedding(res.body());
        } catch (Exception e) {
            return new float[0];
        }
    }

    /** Returns an error string if Ollama is unavailable. */
    public String generate(String prompt) {
        try {
            HttpClient cli = HttpClient.newBuilder()
                    .connectTimeout(Duration.ofSeconds(3))
                    .build();
            String body = "{\"model\":\"" + genModel + "\","
                    + "\"prompt\":\"" + esc(prompt) + "\","
                    + "\"stream\":false}";
            HttpRequest req = HttpRequest.newBuilder()
                    .uri(URI.create("http://" + host + ":" + port + "/api/generate"))
                    .timeout(Duration.ofSeconds(180)) // LLMs can be slow
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body))
                    .build();
            HttpResponse<String> res = cli.send(req, HttpResponse.BodyHandlers.ofString());
            if (res.statusCode() != 200) return "ERROR: Ollama unavailable. Run: ollama serve";
            return parseGenResponse(res.body());
        } catch (Exception e) {
            return "ERROR: Ollama unavailable. Run: ollama serve";
        }
    }
}
