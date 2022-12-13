package rocks.vilaverde.classifier.util;

import java.util.function.Function;

/**
 * A {@link java.util.function.Function} that can throw exceptions.
 */
@FunctionalInterface
public interface ThrowingFunction<T, R, E extends Exception> {
    R apply(T value) throws E;

    static <T,R, E extends Exception> Function<T,R> wrap(ThrowingFunction<T, R, E> checkedFunction) {
        return t -> {
            try {
                return checkedFunction.apply(t);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        };
    }
}
