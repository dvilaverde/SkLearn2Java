package rocks.vilaverde.classifier.dt;

import rocks.vilaverde.classifier.Classifier;
import rocks.vilaverde.classifier.FeatureVector;
import rocks.vilaverde.classifier.Prediction;

/**
 * Implemented by Tree classifiers.
 */
public interface TreeClassifier<T> extends Classifier<T> {

    Prediction<T> getClassification(FeatureVector features);
}
