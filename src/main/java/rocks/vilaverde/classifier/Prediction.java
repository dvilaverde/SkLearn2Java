package rocks.vilaverde.classifier;

/**
 * Represents a prediction in the model.
 */
public interface Prediction<T> {

  T get();

  double[] getProbability();
}
