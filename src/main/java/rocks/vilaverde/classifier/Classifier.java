package rocks.vilaverde.classifier;

import java.util.List;
import java.util.Map;
import java.util.Set;

public interface Classifier<T> {

  /**
   * Predict class or regression value for samples. Predictions will be
   * returned at the same index of the sample provided.
   * @param samples input samples
   * @return class probabilities of the input sample
   */
  List<T> predict(FeatureVector ... samples);

  /**
   * Predict class probabilities of the input samples. Probabilities will be
   * returned at the same index of the sample provided.
   * The predicted class probability is the fraction of samples of the same class in a leaf.
   * @param samples the input samples
   * @return the class probabilities of the input sample
   */
  double[][] predict_proba(FeatureVector ... samples);

  /**
   * Predict class or regression value for features.
   * @param samples input samples
   * @return class probabilities of the input sample
   */
  @Deprecated
  T predict(Map<String, Double> samples);

  /**
   * Predict class probabilities of the input samples features.
   * The predicted class probability is the fraction of samples of the same class in a leaf.
   * @param samples the input samples
   * @return the class probabilities of the input sample
   */
  @Deprecated
  double[] predict_proba(Map<String, Double> samples);

  /**
   * Get the names of all the features in the model.
   * @return set of unique features used in the model
   */
  Set<String> getFeatureNames();
}
