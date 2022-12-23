package rocks.vilaverde.classifier.dt;

import rocks.vilaverde.classifier.Visitor;

/**
 * Visits a {@link DecisionTreeClassifier} using Depth first traversal.
 */
public abstract class AbstractDecisionTreeVisitor implements Visitor<TreeNode> {

  /**
   * Visit a generic TreeNode.
   * @param object the the object being visited
   */
  @Override
  public void visit(TreeNode object) {
    visitBase(object);
  }

  /**
   * Base method that is visited by all TreeNode objects.
   * @param object
   */
  protected void visitBase(TreeNode object) {
  }

  /**
   * Visit the EndNode, and by default will call visitBase first.
   * @param object
   */
  public void visit(EndNode object) {
    visitBase(object);
  }

  /**
   * Visit the ChoiceNode, and by default will call visitBase first.
   * @param object
   */
  public void visit(ChoiceNode object) {
    visitBase(object);

    object.getChild().accept(this);
  }

  /**
   * Visit the DecisionNode, and by default will call visitBase first.
   * @param object
   */
  public void visit(DecisionNode object) {
    visitBase(object);

    object.getLeft().accept(this);
    object.getRight().accept(this);
  }
}
