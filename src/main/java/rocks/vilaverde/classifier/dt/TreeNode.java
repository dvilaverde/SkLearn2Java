package rocks.vilaverde.classifier.dt;

import rocks.vilaverde.classifier.Visitable;

/**
 * Base class for all nodes in a Decision Tree.
 */
public abstract class TreeNode implements Visitable<TreeNode, AbstractDecisionTreeVisitor> {

  /**
   * Must be implemented by every implementation of TreeNode to call visit() on the visitor.
   * @param visitor the visitor
   */
  public abstract void accept(AbstractDecisionTreeVisitor visitor);
}
