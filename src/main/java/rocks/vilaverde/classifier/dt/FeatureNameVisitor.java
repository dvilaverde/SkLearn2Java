package rocks.vilaverde.classifier.dt;

import java.util.HashSet;
import java.util.Set;

/**
 * Visitor that will use Depth first traversal to get the names of the features
 * in the exported DecisionTreeClassifier model.
 */
public class FeatureNameVisitor extends AbstractDecisionTreeVisitor {

  private final Set<String> featureNames = new HashSet<>();


  @Override
  public void visit(DecisionNode object) {

    featureNames.add(object.getFeatureName());

    super.visit(object);
  }

  public Set<String> getFeatureNames() {
    return featureNames;
  }
}
