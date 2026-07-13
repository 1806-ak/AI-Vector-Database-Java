package com.vectordb;

/** Equivalent of C++'s std::pair<float,int> (distance, id), ordered by distance. */
public class ScoredId implements Comparable<ScoredId> {
    public float dist;
    public int id;

    public ScoredId(float dist, int id) {
        this.dist = dist;
        this.id = id;
    }

    @Override
    public int compareTo(ScoredId o) {
        return Float.compare(this.dist, o.dist);
    }
}
