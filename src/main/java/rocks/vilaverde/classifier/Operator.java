package rocks.vilaverde.classifier;

import java.util.function.BiFunction;

public enum Operator {
  LT("<", (leftOperand, rightOperand) -> leftOperand < rightOperand),
  GT(">", (leftOperand, rightOperand) -> leftOperand > rightOperand),
  LT_EQ("<=", (leftOperand, rightOperand) -> leftOperand <= rightOperand),
  GT_EQ(">=", (leftOperand, rightOperand) -> leftOperand >= rightOperand),
  EQ("=", (leftOperand, rightOperand) -> doubleIsSame(leftOperand, rightOperand, .0001));

  private final String operator;
  private final BiFunction<Double, Double, Boolean> operation;

  /**
   * Constructor
   */
  Operator(String op, BiFunction<Double, Double, Boolean> operation) {
    this.operator = op;
    this.operation = operation;
  }

  /**
   * Parse an operator string to an enumeration.
   */
  public static Operator from(String op) {
    for (Operator o : values()) {
      if (o.operator.equals(op)) {
        return o;
      }
    }
    throw new RuntimeException(String.format("invalid operator %s", op));
  }

  private static boolean doubleIsSame(double d1, double d2, double delta) {
    if (Double.compare(d1, d2) == 0) {
      return true;
    } else {
      return (Math.abs(d1 - d2) <= delta);
    }
  }

  public boolean apply(Double leftOperand, Double rightOperand) {
    return this.operation.apply(leftOperand, rightOperand);
  }

  @Override
  public String toString() {
    return operator;
  }
}
