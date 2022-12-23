package rocks.vilaverde.classifier;

/**
 * A container for the values for each feature of a sample that will be predicted.
 */
public class FeatureVector {

    private final Features features;
    private final double[] vector;

    /**
     * Constructor
     * @param features the features
     */
    public FeatureVector(Features features) {
        this.features = features;
        this.vector = new double[features.getLength()];
    }

    /**
     * Add a feature by name.
     * @param feature name of the feature
     * @param value the feature value
     * @return the FeatureVector
     */
    public FeatureVector add(String feature, boolean value) {
        add(feature, value ? 1.0 : 0.0);
        return this;
    }

    /**
     * Add a feature by index.
     * @param index index of the feature
     * @param value the feature value
     * @return the FeatureVector
     */
    public FeatureVector add(int index, boolean value) {
        add(index, value ? 1.0 : 0.0);
        return this;
    }

    /**
     * Add a feature by index.
     * @param index index of the feature
     * @param value the feature value
     * @return the FeatureVector
     */
    public FeatureVector add(int index, double value) {
        this.vector[index] = value;
        return this;
    }

    /**
     * Add a feature by name.
     * @param feature name of the feature
     * @param value the feature value
     * @return the FeatureVector
     */
    public FeatureVector add(String feature, double value) {
        int index = this.features.getFeatureIndex(feature);
        add(index, value);
        return this;
    }

    /**
     * Get the feature value by index.
     * @param index the feature index
     * @return the double value
     */
    public double get(int index) {
        if (index >= vector.length) {
            throw new IllegalArgumentException(String.format("index must be less than %d", index));
        }

        return vector[index];
    }

    /**
     * Get the feature value by name.
     * @param feature the feature name
     * @return the double value
     */
    public double get(String feature) {
        int index = features.getFeatureIndex(feature);
        return get(index);
    }

    /**
     * Returns true when the feature name is present in this feature vector.
     * @param feature feature name
     * @return boolean
     */
    public boolean hasFeature(String feature) {
        return this.features.getFeatureNames().contains(feature);
    }
}
