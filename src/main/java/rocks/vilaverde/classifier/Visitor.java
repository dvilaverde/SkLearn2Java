package rocks.vilaverde.classifier;

/**
 * To be implemented by visitors of the ML model.
 */
public interface Visitor<T> {

  /**
   * Called when an object is being visited during model traversal.
   * @param object the the object being visited
   */
  void visit(T object);
}
