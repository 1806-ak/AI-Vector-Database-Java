package com.vectordb;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TextChunker {

    public static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String trimmed = text == null ? "" : text.trim();
        String[] wordsArr = trimmed.isEmpty() ? new String[0] : trimmed.split("\\s+");
        List<String> words = Arrays.asList(wordsArr);

        List<String> result = new ArrayList<>();
        if (words.isEmpty()) return result;

        if (words.size() <= chunkWords) {
            result.add(text);
            return result;
        }

        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.size(); i += step) {
            int end = Math.min(i + chunkWords, words.size());
            StringBuilder chunk = new StringBuilder();
            for (int j = i; j < end; j++) {
                if (j > i) chunk.append(' ');
                chunk.append(words.get(j));
            }
            result.add(chunk.toString());
            if (end == words.size()) break;
        }
        return result;
    }
}
