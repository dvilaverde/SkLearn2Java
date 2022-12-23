package rocks.vilaverde.classifier;

/**
 * Represents a prediction in the model.
 */
public interface Prediction<T> {

  /**
   * Gets the prediction value.
   * @return type T
   */
  T get();

  /**
   * Gets the probability of the prediction.
   * @return an array of double values
   */
  double[] getProbability();
}
