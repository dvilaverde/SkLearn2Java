package rocks.vilaverde.classifier;

import rocks.vilaverde.classifier.dt.TreeClassifier;

import java.util.Map;

/**
 * Abstract base class for Tree classifiers.
 */
public abstract class AbstractTreeClassifier<T> implements TreeClassifier<T> {

    /**
     * Predict class or regression value for features.
     *
     * @param samples input samples
     * @return class probabilities of the input sample
     */
    @Override
    public T predict(Map<String, Double> samples) {
        FeatureVector fv = toFeatureVector(samples);
        return predict(fv).get(0);
    }

    /**
     * Predict class probabilities of the input samples features.
     * The predicted class probability is the fraction of samples of the same class in a leaf.
     *
     * @param samples the input samples
     * @return the class probabilities of the input sample
     */
    @Override
    public double[] predict_proba(Map<String, Double> samples) {
        FeatureVector fv = toFeatureVector(samples);
        return predict_proba(fv)[0];
    }

    /**
     * Convert a Map of features to a {@link FeatureVector}.
     * @param samples a KV map of feature name to value
     * @return FeatureVector
     */
    private FeatureVector toFeatureVector(Map<String, Double> samples) {
        Features features = Features.fromSet(samples.keySet());
        FeatureVector fv = features.newSample();
        for (Map.Entry<String, Double> entry : samples.entrySet()) {
            fv.add(entry.getKey(), entry.getValue());
        }
        return fv;
    }
}
