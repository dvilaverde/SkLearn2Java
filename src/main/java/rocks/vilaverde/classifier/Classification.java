package rocks.vilaverde.classifier;

import rocks.vilaverde.classifier.dt.Prediction;

public interface Classification<T> {

  Prediction<T> getPrediction();

  double[] getProbability();
}
