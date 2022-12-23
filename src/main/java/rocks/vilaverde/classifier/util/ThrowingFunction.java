package rocks.vilaverde.classifier.util;

import java.util.function.Function;

/**
 * A {@link java.util.function.Function} that can throw exceptions.
 */
@FunctionalInterface
public interface ThrowingFunction<T, R, E extends Exception> {
    /**
     * Apply the function.
     * @param value
     * @return
     * @throws E
     */
    R apply(T value) throws E;

    /**
     * Wrap a lambda that throws exceptions for use in streams.
     * @param checkedFunction the function to check exceptions on
     * @return wrapped Function
     * @param <T> T
     * @param <R> R
     * @param <E> E
     */
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
