package rocks.vilaverde.classifier;

import java.util.Map;
import java.util.Set;

public interface Classifier<T> {

  /**
   * Predict class or regression value for features.
   */
  T predict(Map<String, Double> features);

  /**
   * Predict class probabilities of the input samples features.
   * The predicted class probability is the fraction of samples of the same class in a leaf.
   */
  double[] predict_proba(Map<String, Double> features);

  /**
   * Get the names of all the features in the model.
   */
  Set<String> getFeatureNames();
}
