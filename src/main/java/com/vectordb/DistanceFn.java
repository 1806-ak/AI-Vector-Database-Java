package com.vectordb;

/** Equivalent of the C++ std::function<float(vector,vector)> DistFn typedef. */
@FunctionalInterface
public interface DistanceFn {
    float dist(float[] a, float[] b);
}
