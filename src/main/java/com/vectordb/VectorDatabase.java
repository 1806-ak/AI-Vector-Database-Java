package com.vectordb;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantLock;

public class VectorDatabase {
    private final Map<Integer, VectorItem> store = new HashMap<>();
    private final BruteForce bf = new BruteForce();
    private final KDTree kdt;
    private final HNSW hnsw = new HNSW(16, 200);
    private final ReentrantLock lock = new ReentrantLock();
    private int nextId = 1;
    public final int dims;

    public VectorDatabase(int d) {
        dims = d;
        kdt = new KDTree(d);
    }

    public int insert(String meta, String cat, float[] emb, DistanceFn dist) {
        lock.lock();
        try {
            VectorItem v = new VectorItem(nextId++, meta, cat, emb);
            store.put(v.id, v);
            bf.insert(v);
            kdt.insert(v);
            hnsw.insert(v, dist);
            return v.id;
        } finally {
            lock.unlock();
        }
    }

    public boolean remove(int id) {
        lock.lock();
        try {
            if (!store.containsKey(id)) return false;
            store.remove(id);
            bf.remove(id);
            hnsw.remove(id);
            List<VectorItem> rem = new ArrayList<>(store.values());
            kdt.rebuild(rem);
            return true;
        } finally {
            lock.unlock();
        }
    }

    public static class Hit {
        public int id;
        public String meta, cat;
        public float[] emb;
        public float dist;

        public Hit(int id, String meta, String cat, float[] emb, float dist) {
            this.id = id;
            this.meta = meta;
            this.cat = cat;
            this.emb = emb;
            this.dist = dist;
        }
    }

    public static class SearchOut {
        public List<Hit> hits = new ArrayList<>();
        public long us;
        public String algo, metric;
    }

    public SearchOut search(float[] q, int k, String metric, String algo) {
        lock.lock();
        try {
            DistanceFn dfn = DistanceMetrics.get(metric);
            long t0 = System.nanoTime();

            List<ScoredId> raw;
            if ("bruteforce".equals(algo)) raw = bf.knn(q, k, dfn);
            else if ("kdtree".equals(algo)) raw = kdt.knn(q, k, dfn);
            else raw = hnsw.knn(q, k, 50, dfn);

            long us = (System.nanoTime() - t0) / 1000;

            SearchOut out = new SearchOut();
            out.us = us;
            out.algo = algo;
            out.metric = metric;
            for (ScoredId s : raw) {
                VectorItem v = store.get(s.id);
                if (v != null) out.hits.add(new Hit(s.id, v.metadata, v.category, v.emb, s.dist));
            }
            return out;
        } finally {
            lock.unlock();
        }
    }

    public static class BenchOut {
        public long bfUs, kdUs, hnswUs;
        public int n;
    }

    public BenchOut benchmark(float[] q, int k, String metric) {
        lock.lock();
        try {
            DistanceFn dfn = DistanceMetrics.get(metric);
            BenchOut b = new BenchOut();
            long t;

            t = System.nanoTime();
            bf.knn(q, k, dfn);
            b.bfUs = (System.nanoTime() - t) / 1000;

            t = System.nanoTime();
            kdt.knn(q, k, dfn);
            b.kdUs = (System.nanoTime() - t) / 1000;

            t = System.nanoTime();
            hnsw.knn(q, k, 50, dfn);
            b.hnswUs = (System.nanoTime() - t) / 1000;

            b.n = store.size();
            return b;
        } finally {
            lock.unlock();
        }
    }

    public List<VectorItem> all() {
        lock.lock();
        try {
            return new ArrayList<>(store.values());
        } finally {
            lock.unlock();
        }
    }

    public HNSW.GraphInfo hnswInfo() {
        lock.lock();
        try {
            return hnsw.getInfo();
        } finally {
            lock.unlock();
        }
    }

    public int size() {
        lock.lock();
        try {
            return store.size();
        } finally {
            lock.unlock();
        }
    }
}
