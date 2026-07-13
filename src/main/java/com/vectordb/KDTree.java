package com.vectordb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;

public class KDTree {

    private static class Node {
        VectorItem item;
        Node left, right;
        Node(VectorItem v) { item = v; }
    }

    private Node root;
    private final int dims;

    public KDTree(int d) {
        dims = d;
    }

    public void insert(VectorItem v) {
        root = ins(root, v, 0);
    }

    private Node ins(Node n, VectorItem v, int d) {
        if (n == null) return new Node(v);
        int ax = d % dims;
        if (v.emb[ax] < n.item.emb[ax]) n.left = ins(n.left, v, d + 1);
        else n.right = ins(n.right, v, d + 1);
        return n;
    }

    public List<ScoredId> knn(float[] q, int k, DistanceFn dist) {
        // Max-heap on distance, capped at size k (mirrors std::priority_queue<pair<float,int>>)
        PriorityQueue<ScoredId> heap = new PriorityQueue<>(Collections.reverseOrder());
        knnRec(root, q, k, 0, dist, heap);
        List<ScoredId> r = new ArrayList<>(heap);
        Collections.sort(r);
        return r;
    }

    private void knnRec(Node n, float[] q, int k, int d, DistanceFn dist, PriorityQueue<ScoredId> heap) {
        if (n == null) return;
        float dn = dist.dist(q, n.item.emb);
        if (heap.size() < k || dn < heap.peek().dist) {
            heap.add(new ScoredId(dn, n.item.id));
            if (heap.size() > k) heap.poll();
        }
        int ax = d % dims;
        float diff = q[ax] - n.item.emb[ax];
        Node closer = diff < 0 ? n.left : n.right;
        Node farther = diff < 0 ? n.right : n.left;
        knnRec(closer, q, k, d + 1, dist, heap);
        if (heap.size() < k || Math.abs(diff) < heap.peek().dist) {
            knnRec(farther, q, k, d + 1, dist, heap);
        }
    }

    public void rebuild(List<VectorItem> items) {
        root = null;
        for (VectorItem v : items) insert(v);
    }
}
