/////////////////////////////////////////////////////////////////////////////
// PROPRIETARY RIGHTS STATEMENT
// The contents of this file represent confidential information that is the
// proprietary property of Edge2Web, Inc. Viewing or use of
// this information is prohibited without the express written consent of
// Edge2Web, Inc. Removal of this PROPRIETARY RIGHTS STATEMENT
// is strictly forbidden. Copyright (c) 2016 All rights reserved.
/////////////////////////////////////////////////////////////////////////////
package rocks.vilaverde.classifier.dt;

/**
 * A prediction from the classifier
 */
public interface PredictionFactory<T> {

    Prediction<T> create(String value);

    public class BooleanPredictionFactory implements PredictionFactory<Boolean> {

        @Override
        public Prediction<Boolean> create(final String value) {
            return () -> Boolean.valueOf(value.toLowerCase());
        }
    }

    public class IntegerPredictionFactory implements PredictionFactory<Integer> {

        @Override
        public Prediction<Integer> create(String value) {
            return () -> Integer.valueOf(value);
        }
    }
}
