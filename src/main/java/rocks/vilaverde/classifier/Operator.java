package rocks.vilaverde.classifier;

public enum Operator {
  LT("<"),
  GT(">"),
  LT_EQ("<="),
  GT_EQ(">="),
  EQ("=");

  private final String operator;

  Operator(String op) {
    this.operator = op;
  }

  public static Operator from(String op) {
    for (Operator o : values()) {
      if (o.operator.equals(op)) {
        return o;
      }
    }
    throw new RuntimeException(String.format("invalid operator %s", op));
  }

  @Override
  public String toString() {
    return operator;
  }
}
