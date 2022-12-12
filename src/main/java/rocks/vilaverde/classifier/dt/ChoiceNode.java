package rocks.vilaverde.classifier.dt;

import rocks.vilaverde.classifier.Operator;

class ChoiceNode extends TreeNode {
  private final Operator op;
  private final Double value;

  private TreeNode child;

  public static ChoiceNode create(Operator op, Double value) {
    return new ChoiceNode(op, value);
  }

  private ChoiceNode(Operator op, Double value) {
    this.op = op;
    this.value = value;
  }

  public void addChild(TreeNode child) {
    this.child = child;
  }

  public TreeNode getChild() {
    return child;
  }

  public boolean eval(Double featureValue) {
    boolean result = false;
    switch (op) {
      case EQ:
        result = doubleIsSame(featureValue, value, .0001);
        break;
      case GT:
        result = featureValue > value;
        break;
      case LT:
        result = featureValue < value;
        break;
      case GT_EQ:
        result = featureValue >= value;
        break;
      case LT_EQ:
        result = featureValue <= value;
        break;
    }
    return result;
  }

  private boolean doubleIsSame(double d1, double d2, double delta) {
    if (Double.compare(d1, d2) == 0) {
      return true;
    } else {
      return (Math.abs(d1 - d2) <= delta);
    }
  }

  @Override
  public void accept(AbstractDecisionTreeVisitor visitor) {
    visitor.visit(this);
  }

  public String toString() {
    return String.format("%s %s", op.toString(), value.toString());
  }
}
