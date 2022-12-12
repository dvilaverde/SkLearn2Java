package rocks.vilaverde.classifier;

/**
 * Represents features that are Boolean as a Double `1.0` or `0.0`.
 */
public enum BooleanFeature {

  FALSE(0.0),
  TRUE(1.0);

  private final Double value;
  BooleanFeature(double v) {
    value = v;
  }

  public Double asDouble() {
    return value;
  }
}
