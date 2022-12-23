package rocks.vilaverde.classifier.dt;

/**
 * A prediction from the classifier can be of type Double, Integer, or Boolean. This
 * factory is used while parsing the text export of the DecisionTreeClassifier to construct
 * the correct Java type for the serialized value.
 */
@FunctionalInterface
public interface PredictionFactory<T> {

    /**
     * The prediction class is of type boolean in the model.
     */
    PredictionFactory<Boolean> BOOLEAN = value ->  Boolean.valueOf(value.toLowerCase());
    /**
     * The prediction class is of type Integer in the model.
     */
    PredictionFactory<Integer> INTEGER = Integer::valueOf;
    /**
     * The prediction class is of type Double in the model.
     */
    PredictionFactory<Double> DOUBLE = Double::parseDouble;

    /**
     * Convert a String value to the appropriate type for the model.
     * @param value the serialized text value of the prediction from the exported decision tree.
     * @return the deserialized type
     */
    T create(String value);
}
