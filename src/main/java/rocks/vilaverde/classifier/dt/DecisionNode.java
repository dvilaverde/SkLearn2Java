package rocks.vilaverde.classifier.dt;

import java.util.ArrayList;
import java.util.List;

class DecisionNode extends TreeNode {

  private final String featureName;

  private final List<TreeNode> children = new ArrayList<>(2);

  public List<TreeNode> getChildren() {
    return children;
  }

  public static DecisionNode create(String feature) {
    return new DecisionNode(feature);
  }

  private DecisionNode(String featureName) {
    this.featureName = featureName;
  }

  public String getFeatureName() {
    return featureName;
  }

  @Override
  public void accept(AbstractDecisionTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public String toString() {
    return String.format("%s, %d operations", getFeatureName(), getChildren().size());
  }
}
