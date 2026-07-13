package com.vectordb;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;

/** Hierarchical Navigable Small World graph — ported 1:1 from the C++ HNSW class. */
public class HNSW {

    private static class Node {
        VectorItem item;
        int maxLyr;
        List<List<Integer>> nbrs; // nbrs.get(layer) -> neighbor ids at that layer

        Node(VectorItem item, int maxLyr, List<List<Integer>> nbrs) {
            this.item = item;
            this.maxLyr = maxLyr;
            this.nbrs = nbrs;
        }
    }

    private final Map<Integer, Node> G = new HashMap<>();
    private final int M, M0, efBuild;
    private final double mL;
    private int topLayer = -1;
    private int entryPt = -1;
    private final Random rng = new Random(42);

    public HNSW() {
        this(16, 200);
    }

    public HNSW(int m, int efBuildIn) {
        M = m;
        M0 = 2 * m;
        efBuild = efBuildIn;
        mL = 1.0 / Math.log(m);
    }

    private int randLevel() {
        double u = rng.nextDouble();
        if (u <= 0) u = 1e-12;
        return (int) Math.floor(-Math.log(u) * mL);
    }

    private List<ScoredId> searchLayer(float[] q, int ep, int ef, int lyr, DistanceFn dist) {
        Set<Integer> vis = new HashSet<>();
        PriorityQueue<ScoredId> cands = new PriorityQueue<>();                       // min-heap
        PriorityQueue<ScoredId> found = new PriorityQueue<>(Collections.reverseOrder()); // max-heap

        Node epNode = G.get(ep);
        float d0 = dist.dist(q, epNode.item.emb);
        vis.add(ep);
        cands.add(new ScoredId(d0, ep));
        found.add(new ScoredId(d0, ep));

        while (!cands.isEmpty()) {
            ScoredId c = cands.poll();
            if (found.size() >= ef && c.dist > found.peek().dist) break;
            Node cn = G.get(c.id);
            if (cn == null || lyr >= cn.nbrs.size()) continue;
            for (int nid : cn.nbrs.get(lyr)) {
                if (vis.contains(nid) || !G.containsKey(nid)) continue;
                vis.add(nid);
                float nd = dist.dist(q, G.get(nid).item.emb);
                if (found.size() < ef || nd < found.peek().dist) {
                    cands.add(new ScoredId(nd, nid));
                    found.add(new ScoredId(nd, nid));
                    if (found.size() > ef) found.poll();
                }
            }
        }

        List<ScoredId> res = new ArrayList<>(found);
        Collections.sort(res);
        return res;
    }

    private List<Integer> selectNbrs(List<ScoredId> cands, int maxM) {
        List<Integer> r = new ArrayList<>();
        int lim = Math.min(cands.size(), maxM);
        for (int i = 0; i < lim; i++) r.add(cands.get(i).id);
        return r;
    }

    public void insert(VectorItem item, DistanceFn dist) {
        int id = item.id;
        int lvl = randLevel();
        List<List<Integer>> nbrs = new ArrayList<>();
        for (int i = 0; i <= lvl; i++) nbrs.add(new ArrayList<>());
        G.put(id, new Node(item, lvl, nbrs));

        if (entryPt == -1) {
            entryPt = id;
            topLayer = lvl;
            return;
        }

        int ep = entryPt;
        for (int lc = topLayer; lc > lvl; lc--) {
            Node epNode = G.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<ScoredId> W = searchLayer(item.emb, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = W.get(0).id;
            }
        }
        for (int lc = Math.min(topLayer, lvl); lc >= 0; lc--) {
            List<ScoredId> W = searchLayer(item.emb, ep, efBuild, lc, dist);
            int maxM = (lc == 0) ? M0 : M;
            List<Integer> sel = selectNbrs(W, maxM);
            G.get(id).nbrs.set(lc, sel);

            for (int nid : sel) {
                Node nn = G.get(nid);
                if (nn == null) continue;
                while (nn.nbrs.size() <= lc) nn.nbrs.add(new ArrayList<>());
                List<Integer> conn = nn.nbrs.get(lc);
                conn.add(id);
                if (conn.size() > maxM) {
                    List<ScoredId> ds = new ArrayList<>();
                    for (int c : conn) {
                        if (G.containsKey(c)) {
                            ds.add(new ScoredId(dist.dist(nn.item.emb, G.get(c).item.emb), c));
                        }
                    }
                    Collections.sort(ds);
                    conn.clear();
                    for (int i = 0; i < maxM && i < ds.size(); i++) conn.add(ds.get(i).id);
                }
            }
            if (!W.isEmpty()) ep = W.get(0).id;
        }
        if (lvl > topLayer) {
            topLayer = lvl;
            entryPt = id;
        }
    }

    public List<ScoredId> knn(float[] q, int k, int ef, DistanceFn dist) {
        if (entryPt == -1) return new ArrayList<>();
        int ep = entryPt;
        for (int lc = topLayer; lc > 0; lc--) {
            Node epNode = G.get(ep);
            if (epNode != null && lc < epNode.nbrs.size()) {
                List<ScoredId> W = searchLayer(q, ep, 1, lc, dist);
                if (!W.isEmpty()) ep = W.get(0).id;
            }
        }
        List<ScoredId> W = searchLayer(q, ep, Math.max(ef, k), 0, dist);
        if (W.size() > k) W = new ArrayList<>(W.subList(0, k));
        return W;
    }

    public void remove(int id) {
        if (!G.containsKey(id)) return;
        for (Node nd : G.values()) {
            for (List<Integer> layer : nd.nbrs) {
                layer.removeIf(x -> x == id);
            }
        }
        if (entryPt == id) {
            entryPt = -1;
            for (Map.Entry<Integer, Node> e : G.entrySet()) {
                if (e.getKey() != id) {
                    entryPt = e.getKey();
                    break;
                }
            }
        }
        G.remove(id);
    }

    public static class NV {
        public int id;
        public String metadata, category;
        public int maxLyr;

        public NV(int id, String metadata, String category, int maxLyr) {
            this.id = id;
            this.metadata = metadata;
            this.category = category;
            this.maxLyr = maxLyr;
        }
    }

    public static class EV {
        public int src, dst, lyr;

        public EV(int src, int dst, int lyr) {
            this.src = src;
            this.dst = dst;
            this.lyr = lyr;
        }
    }

    public static class GraphInfo {
        public int topLayer, nodeCount;
        public List<Integer> nodesPerLayer = new ArrayList<>();
        public List<Integer> edgesPerLayer = new ArrayList<>();
        public List<NV> nodes = new ArrayList<>();
        public List<EV> edges = new ArrayList<>();
    }

    public GraphInfo getInfo() {
        GraphInfo gi = new GraphInfo();
        gi.topLayer = topLayer;
        gi.nodeCount = G.size();
        int maxL = Math.max(topLayer + 1, 1);
        for (int i = 0; i < maxL; i++) {
            gi.nodesPerLayer.add(0);
            gi.edgesPerLayer.add(0);
        }
        for (Map.Entry<Integer, Node> e : G.entrySet()) {
            int id = e.getKey();
            Node nd = e.getValue();
            gi.nodes.add(new NV(id, nd.item.metadata, nd.item.category, nd.maxLyr));
            for (int lc = 0; lc <= nd.maxLyr && lc < maxL; lc++) {
                gi.nodesPerLayer.set(lc, gi.nodesPerLayer.get(lc) + 1);
                if (lc < nd.nbrs.size()) {
                    for (int nid : nd.nbrs.get(lc)) {
                        if (id < nid) {
                            gi.edgesPerLayer.set(lc, gi.edgesPerLayer.get(lc) + 1);
                            gi.edges.add(new EV(id, nid, lc));
                        }
                    }
                }
            }
        }
        return gi;
    }

    public int size() {
        return G.size();
    }
}
