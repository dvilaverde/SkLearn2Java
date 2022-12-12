package rocks.vilaverde.classifier;

public interface Visitor<T> {

  void visit(T object);
}
