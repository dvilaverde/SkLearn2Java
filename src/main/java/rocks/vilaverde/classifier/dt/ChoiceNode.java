package rocks.vilaverde.classifier.dt;

import rocks.vilaverde.classifier.Operator;

/**
 * Represents a Choice in the decision tree, where when the expression is evaluated,
 * if true will result in the child node of the choice being selected.
 */
public class ChoiceNode extends TreeNode {
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

  @Override
  public void accept(AbstractDecisionTreeVisitor visitor) {
    visitor.visit(this);
  }

  public String toString() {
    return String.format("%s %s", op.toString(), value.toString());
  }

  public boolean eval(double featureValue) {
    return op.apply(featureValue, value);
  }
}
