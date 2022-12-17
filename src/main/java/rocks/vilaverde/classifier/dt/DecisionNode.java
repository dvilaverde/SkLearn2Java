package rocks.vilaverde.classifier.dt;

/**
 * Represents a decision in the DecisionTreeClassifier. The decision will have
 * a left and right hand {@link ChoiceNode} to be evaluated.
 * A {@link ChoiceNode} may have nested {@link DecisionNode} or {@link EndNode}.
 */
public class DecisionNode extends TreeNode {

  private final String featureName;

  private ChoiceNode left;
  private ChoiceNode right;

  /**
   * Factory method to create a {@link DecisionNode}.
   * @param feature the name of the feature.
   * @return DecisionNode
   */
  public static DecisionNode create(String feature) {
    return new DecisionNode(feature);
  }

  /**
   * Private Constructor.
   * @param featureName the name of the feature used in this decision
   */
  private DecisionNode(String featureName) {
    this.featureName = featureName.intern();
  }

  /**
   * Getter for the left.
   * @return ChoiceNode
   */
  public ChoiceNode getLeft() {
    return left;
  }

  /**
   * Left hand side of the decision.
   * @param left the left {@link ChoiceNode}
   */
  public void setLeft(ChoiceNode left) {
    this.left = left;
  }

  /**
   * Getter for the right.
   * @return ChoiceNode
   */
  public ChoiceNode getRight() {
    return right;
  }

  /**
   * Right hand side of a decision.
   * @param right the right {@link ChoiceNode}
   */
  public void setRight(ChoiceNode right) {
    this.right = right;
  }

  /**
   * @return true when the left and right choice are set on this decision.
   */
  public boolean isComplete() {
    return getLeft() != null && getRight() != null;
  }

  /**
   * Getter for the feature used in this decision node.
   * @return String
   */
  public String getFeatureName() {
    return featureName;
  }

  @Override
  public void accept(AbstractDecisionTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public String toString() {
    int count = 0;
    if (getLeft() != null) {
      count++;
    }

    if (getRight() != null) {
      count++;
    }

    return String.format("%s, %d operations", getFeatureName(), count);
  }
}
