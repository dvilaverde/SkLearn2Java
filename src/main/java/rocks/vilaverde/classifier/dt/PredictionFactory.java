package rocks.vilaverde.classifier.dt;

/**
 * A prediction from the classifier
 */
@FunctionalInterface
public interface PredictionFactory<T> {

    PredictionFactory<Boolean> BOOLEAN = value ->  Boolean.valueOf(value.toLowerCase());
    PredictionFactory<Integer> INTEGER = Integer::valueOf;
    PredictionFactory<Double> DOUBLE = Double::parseDouble;

    T create(String value);
}
