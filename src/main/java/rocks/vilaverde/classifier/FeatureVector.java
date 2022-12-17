package rocks.vilaverde.classifier;

/**
 * A container for the values for each feature of a sample that will be predicted.
 */
public class FeatureVector {

    private final Features features;
    private final double[] vector;

    public FeatureVector(Features features) {
        this.features = features;
        this.vector = new double[features.getLength()];
    }

    public FeatureVector add(String feature, boolean value) {
        add(feature, value ? 1.0 : 0.0);
        return this;
    }

    public FeatureVector add(int index, boolean value) {
        add(index, value ? 1.0 : 0.0);
        return this;
    }

    public FeatureVector add(int index, double value) {
        this.vector[index] = value;
        return this;
    }

    public FeatureVector add(String feature, double value) {
        int index = this.features.getFeatureIndex(feature);
        add(index, value);
        return this;
    }

    public double get(int index) {
        if (index >= vector.length) {
            throw new IllegalArgumentException(String.format("index must be less than %d", index));
        }

        return vector[index];
    }

    public double get(String feature) {
        int index = features.getFeatureIndex(feature);
        return get(index);
    }

    public boolean hasFeature(String feature) {
        return this.features.getFeatureNames().contains(feature);
    }
}
