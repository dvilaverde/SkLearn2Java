package rocks.vilaverde.classifier;

import java.util.function.BiFunction;

public enum Operator {
  /** Less than */
  LT("<", (leftOperand, rightOperand) -> leftOperand < rightOperand),
  /** Greater than */
  GT(">", (leftOperand, rightOperand) -> leftOperand > rightOperand),
  /** Less than or Equal */
  LT_EQ("<=", (leftOperand, rightOperand) -> leftOperand <= rightOperand),
  /** Greater than or equal */
  GT_EQ(">=", (leftOperand, rightOperand) -> leftOperand >= rightOperand),
  /** Equal */
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
   * @param op the operation as string to convert to an enum.
   * @return the Operator
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

  /**
   * Apply this operation on the operands.
   * @param leftOperand the value left of the operator
   * @param rightOperand the value right of the operator
   * @return result of the operation
   */
  public boolean apply(Double leftOperand, Double rightOperand) {
    return this.operation.apply(leftOperand, rightOperand);
  }

  /**
   * For debugging
   * @return String
   */
  @Override
  public String toString() {
    return operator;
  }
}
