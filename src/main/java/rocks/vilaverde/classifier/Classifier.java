package rocks.vilaverde.classifier;

import java.util.Map;
import java.util.Set;

public interface Classifier<T> {

  /**
   * Predict class or regression value for features.
   * @param features input samples
   * @return class probabilities of the input sample
   */
  T predict(Map<String, Double> features);

  /**
   * Predict class probabilities of the input samples features.
   * The predicted class probability is the fraction of samples of the same class in a leaf.
   * @param features the input samples
   * @return the class probabilities of the input sample
   */
  double[] predict_proba(Map<String, Double> features);

  /**
   * Get the names of all the features in the model.
   * @return set of unique features used in the model
   */
  Set<String> getFeatureNames();
}
