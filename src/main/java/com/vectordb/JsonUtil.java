package com.vectordb;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class JsonUtil {

    public static String jS(String s) {
        if (s == null) s = "";
        StringBuilder o = new StringBuilder("\"");
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
        o.append('"');
        return o.toString();
    }

    public static String jVec(float[] v) {
        StringBuilder ss = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            if (i > 0) ss.append(',');
            ss.append(String.format(Locale.US, "%.4f", v[i]));
        }
        ss.append(']');
        return ss.toString();
    }

    public static float[] parseVec(String s) {
        if (s == null || s.isEmpty()) return new float[0];
        String[] parts = s.split(",");
        List<Float> list = new ArrayList<>();
        for (String t : parts) {
            try {
                list.add(Float.parseFloat(t.trim()));
            } catch (Exception ignored) {
            }
        }
        float[] arr = new float[list.size()];
        for (int i = 0; i < arr.length; i++) arr[i] = list.get(i);
        return arr;
    }

    /** Extract a JSON string field value (handles basic escape sequences). */
    public static String extractStr(String body, String key) {
        if (body == null) return "";
        int p = body.indexOf('"' + key + '"');
        if (p == -1) return "";
        p = body.indexOf(':', p);
        if (p == -1) return "";
        p++;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        if (p >= body.length() || body.charAt(p) != '"') return "";
        p++;
        StringBuilder result = new StringBuilder();
        while (p < body.length()) {
            char c = body.charAt(p);
            if (c == '"') break;
            if (c == '\\' && p + 1 < body.length()) {
                p++;
                char e = body.charAt(p);
                switch (e) {
                    case '"': result.append('"'); break;
                    case '\\': result.append('\\'); break;
                    case 'n': result.append('\n'); break;
                    case 'r': result.append('\r'); break;
                    case 't': result.append('\t'); break;
                    default: result.append(e);
                }
            } else {
                result.append(c);
            }
            p++;
        }
        return result.toString();
    }

    /** Extract a JSON integer field value. */
    public static int extractInt(String body, String key, int def) {
        if (body == null) return def;
        int p = body.indexOf('"' + key + '"');
        if (p == -1) return def;
        p = body.indexOf(':', p);
        if (p == -1) return def;
        p++;
        while (p < body.length() && (body.charAt(p) == ' ' || body.charAt(p) == '\t')) p++;
        int start = p;
        while (p < body.length() && (Character.isDigit(body.charAt(p)) || body.charAt(p) == '-')) p++;
        try {
            return Integer.parseInt(body.substring(start, p));
        } catch (Exception e) {
            return def;
        }
    }

    /** Extract the raw inner text of a JSON array field, e.g. "embedding":[1,2,3] -> "1,2,3". */
    public static String extractArrRaw(String body, String key) {
        int p = body.indexOf('"' + key + '"');
        if (p == -1) return null;
        p = body.indexOf('[', p);
        if (p == -1) return null;
        int e = body.indexOf(']', p);
        if (e == -1) return null;
        return body.substring(p + 1, e);
    }
}
