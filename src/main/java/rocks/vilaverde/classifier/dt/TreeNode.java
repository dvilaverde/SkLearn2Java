package rocks.vilaverde.classifier.dt;

import rocks.vilaverde.classifier.Visitable;

public abstract class TreeNode implements Visitable<TreeNode, AbstractDecisionTreeVisitor> {

  public abstract void accept(AbstractDecisionTreeVisitor visitor);
}
