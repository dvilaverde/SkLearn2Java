package rocks.vilaverde.classifier;

public interface Visitable<T, V extends Visitor<T>> {

  void accept(V visitor);
}
