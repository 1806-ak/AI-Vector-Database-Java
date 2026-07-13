package com.vectordb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class BruteForce {
    public List<VectorItem> items = new ArrayList<>();

    public void insert(VectorItem v) {
        items.add(v);
    }

    public List<ScoredId> knn(float[] q, int k, DistanceFn dist) {
        List<ScoredId> r = new ArrayList<>(items.size());
        for (VectorItem v : items) r.add(new ScoredId(dist.dist(q, v.emb), v.id));
        Collections.sort(r);
        if (r.size() > k) r = new ArrayList<>(r.subList(0, k));
        return r;
    }

    public void remove(int id) {
        items.removeIf(v -> v.id == id);
    }
}
