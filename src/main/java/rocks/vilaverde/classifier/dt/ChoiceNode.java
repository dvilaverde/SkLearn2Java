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

  /**
   * Creates a ChoiceNode from an Operator and a value to be used for evaluation.
   * @param op the Operator
   * @param value the value
   * @return the ChoiceNode
   */
  public static ChoiceNode create(Operator op, Double value) {
    return new ChoiceNode(op, value);
  }

  /**
   * Private constructor, use the creator static function.
   * @param op the Operator
   * @param value the value
   */
  private ChoiceNode(Operator op, Double value) {
    this.op = op;
    this.value = value;
  }

  /**
   * Add a child node to the ChoiceNode.
   * @param child the child to add
   */
  public void addChild(TreeNode child) {
    this.child = child;
  }

  /**
   * Accessor for the child TreeNode
   * @return the TreeNode
   */
  public TreeNode getChild() {
    return child;
  }

  /**
   * Accepts a visitor and calls visit using this node.
   * @param visitor the visitor
   */
  @Override
  public void accept(AbstractDecisionTreeVisitor visitor) {
    visitor.visit(this);
  }

  /**
   * For debugging.
   * @return The formatted String.
   */
  public String toString() {
    return String.format("%s %s", op.toString(), value.toString());
  }

  /**
   * Evaluate this ChoiceNode against the provided value.
   * @param featureValue the feature value this choice will be evaluated against.
   * @return a boolean result of the evaluation.
   */
  public boolean eval(double featureValue) {
    return op.apply(featureValue, value);
  }
}
