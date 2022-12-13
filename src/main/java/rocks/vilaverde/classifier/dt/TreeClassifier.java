package rocks.vilaverde.classifier.dt;

import rocks.vilaverde.classifier.Classifier;
import rocks.vilaverde.classifier.Prediction;

import java.util.Map;

/**
 * Implemented by Tree classifiers.
 */
public interface TreeClassifier<T> extends Classifier<T> {

    Prediction<T> getClassification(Map<String, Double> features);
}
