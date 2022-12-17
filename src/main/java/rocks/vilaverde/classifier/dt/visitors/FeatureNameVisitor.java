package rocks.vilaverde.classifier.dt.visitors;

import rocks.vilaverde.classifier.dt.AbstractDecisionTreeVisitor;
import rocks.vilaverde.classifier.dt.DecisionNode;

import java.util.HashSet;
import java.util.Set;

/**
 * Visitor that will use Depth first traversal to get the names of the features
 * in the exported DecisionTreeClassifier model.
 */
public class FeatureNameVisitor extends AbstractDecisionTreeVisitor {

  private final Set<String> featureNames = new HashSet<>();

  /**
   * Visit a {@link DecisionNode} and collect the feature name used in the decision.
   * @param object the {@link DecisionNode} being visited.
   */
  @Override
  public void visit(DecisionNode object) {
    featureNames.add(object.getFeatureName());
    super.visit(object);
  }

  public Set<String> getFeatureNames() {
    return featureNames;
  }
}
