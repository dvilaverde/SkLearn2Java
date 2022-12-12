package rocks.vilaverde.classifier.dt;

import java.util.HashSet;
import java.util.Set;

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
