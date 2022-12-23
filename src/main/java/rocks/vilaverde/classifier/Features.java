package rocks.vilaverde.classifier;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.OptionalInt;
import java.util.Set;

/**
 * Container class for the set of named features that will be provided for each sample.
 * Call {@link Features#newSample()} to create a {@link FeatureVector} to provide the
 * values for the sample.
 */
public class Features {

    /* Map of feature name to index */
    private final Map<String, Integer> features = new HashMap<>();
    /* Feature can be added as long as no samples have yet to be created,
       at that point this is immutable */
    private boolean allowFeatureAdd = true;

    /**
     * Convienence creation method.
     * @param features the set of features.
     * @return Features
     */
    public static Features of(String ... features) {
        
        // make sure all the features are unique
        Set<String> featureSet = new HashSet<>(Arrays.asList(features));
        if (featureSet.size() != features.length) {
            throw new IllegalArgumentException("features names are not unique");
        }
        
        return new Features(features);
    }

    /**
     * Convert a Set of Strings to a Features object.
     * @param features the Set of strings
     * @return a Features
     */
    public static Features fromSet(Set<String> features) {
        return new Features(features.toArray(new String[0]));
    }

    /**
     * Constructor
     * @param features order list of features.
     */
    private Features(String ... features) {
        for (int i = 0; i < features.length; i++) {
            this.features.put(features[i], i);
        }
    }

    FeatureVector newSample() {
        allowFeatureAdd = false;
        return new FeatureVector(this);
    }

    /**
     * Add a Feature to the set of features.
     * @param feature Feature name
     */
    public void addFeature(String feature) {
        if (!allowFeatureAdd) {
            throw new IllegalStateException("features are immutable");
        }

        if (!this.features.containsKey(feature)) {
            int next = 0;
            OptionalInt optionalInt = features.values().stream().mapToInt(integer -> integer).max();
            if (optionalInt.isPresent()) {
                next = optionalInt.getAsInt() + 1;
            }

            this.features.put(feature, next);
        }
    }

    /**
     * Get the number of features in this Features.
     * @return the size of teh features.
     */
    public int getLength() {
        return this.features.size();
    }

    /**
     * Get the index of a feature from the feature vector.
     * @param feature the feature name
     * @return the position in the array that will hold this feature
     */
    public int getFeatureIndex(String feature) {
        Integer index = this.features.get(feature);
        if (index == null) {
            throw new IllegalArgumentException(String.format("feature %s does not exist", feature));
        }

        return index;
    }

    /**
     * Get the names of the features in the model.
     * @return Set of feature names
     */
    public Set<String> getFeatureNames() {
        return Collections.unmodifiableSet(this.features.keySet());
    }
}
