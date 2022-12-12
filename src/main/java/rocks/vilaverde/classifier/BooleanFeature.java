package rocks.vilaverde.classifier;

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
