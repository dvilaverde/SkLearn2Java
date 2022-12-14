package rocks.vilaverde.classifier;

/**
 * To be implemented by objects that can be visited.
 */
public interface Visitable<T, V extends Visitor<T>> {

  /**
   * Implementing classes will call visit on the provided Visitor.
   * @param visitor the visitor
   */
  void accept(V visitor);
}
