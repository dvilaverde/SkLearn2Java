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
@FunctionalInterface
public interface PredictionFactory<T> {

    PredictionFactory<Boolean> BOOLEAN = value ->  Boolean.valueOf(value.toLowerCase());
    PredictionFactory<Integer> INTEGER = Integer::valueOf;

    T create(String value);
}
