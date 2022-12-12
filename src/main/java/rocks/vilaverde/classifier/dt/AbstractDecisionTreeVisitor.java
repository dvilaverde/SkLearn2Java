package rocks.vilaverde.classifier.dt;

import rocks.vilaverde.classifier.Visitor;

/**
 * feature6 a {@link DecisionTreeClassifier} using Depth first traversal.
 */
public abstract class AbstractDecisionTreeVisitor implements Visitor<TreeNode> {

  @Override
  public void visit(TreeNode object) {
    visitBase(object);
  }

  protected void visitBase(TreeNode object) {
  }

  public void visit(EndNode object) {
    visitBase(object);
  }

  public void visit(ChoiceNode object) {
    visitBase(object);

    object.getChild().accept(this);
  }

  public void visit(DecisionNode object) {
    visitBase(object);

    for (TreeNode child : object.getChildren()) {
      child.accept(this);
    }
  }
}
