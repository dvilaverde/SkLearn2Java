package rocks.vilaverde.classifier;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import rocks.vilaverde.classifier.dt.DecisionTreeClassifier;
import rocks.vilaverde.classifier.dt.PredictionFactory;

import java.io.InputStreamReader;
import java.io.Reader;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;


public class DecisionTreeClassifierTest {

    @Test
    public void parseSimpleTree() throws Exception {
        ClassLoader cl = DecisionTreeClassifierTest.class.getClassLoader();
        Reader tree = new InputStreamReader(cl.getResourceAsStream("simple-tree.model"));
        final Classifier<Boolean> decisionTree = DecisionTreeClassifier.parse(tree,
                new PredictionFactory.BooleanPredictionFactory());
        Assertions.assertNotNull(decisionTree);

        Assertions.assertFalse(decisionTree.predict(Collections.singletonMap("feature1", 1.2)));
        Assertions.assertTrue(decisionTree.predict(Collections.singletonMap("feature1", 2.4)));

        Assertions.assertNotNull(decisionTree.getFeatureNames());
        Assertions.assertEquals(1, decisionTree.getFeatureNames().size());
        Assertions.assertEquals("feature1", decisionTree.getFeatureNames().iterator().next());
    }

    @Test
    public void parseDeepTree() throws Exception {
        ClassLoader cl = DecisionTreeClassifierTest.class.getClassLoader();
        Reader tree = new InputStreamReader(cl.getResourceAsStream("decision-tree.model"));
        final Classifier<Boolean> decisionTree = DecisionTreeClassifier.parse(tree,
                new PredictionFactory.BooleanPredictionFactory());
        Assertions.assertNotNull(decisionTree);

        Map<String, Double> features = new HashMap<>();
        features.put("feature1", 0.0);
        features.put("feature2", 1.0);
        features.put("feature3", BooleanFeature.FALSE.asDouble());
        features.put("feature4", 0.0);
        features.put("feature5", BooleanFeature.FALSE.asDouble());
        features.put("feature6", 1.0);
        features.put("feature7", 1.0);
        features.put("feature8", 0.0);

        Assertions.assertFalse(decisionTree.predict(features));

        features.put("feature5", BooleanFeature.TRUE.asDouble());
        Assertions.assertTrue(decisionTree.predict(features));

        Assertions.assertNotNull(decisionTree.getFeatureNames());
        Assertions.assertEquals(8, decisionTree.getFeatureNames().size());
    }

    @Test
    public void invalidFeatureName() throws Exception {
        IllegalArgumentException ex = Assertions.assertThrows(IllegalArgumentException.class, () -> {
            ClassLoader cl = DecisionTreeClassifierTest.class.getClassLoader();
            Reader tree = new InputStreamReader(cl.getResourceAsStream("decision-tree.model"));
            final Classifier<Boolean> decisionTree = DecisionTreeClassifier.parse(tree,
                    new PredictionFactory.BooleanPredictionFactory());
            Assertions.assertNotNull(decisionTree);

            Map<String, Double> features = new HashMap<>();
            features.put("feature11", 0.0);
            features.put("feature2", 1.0);
            features.put("feature3", BooleanFeature.FALSE.asDouble());
            features.put("feature4", 0.0);
            features.put("feature5", BooleanFeature.FALSE.asDouble());
            features.put("feature6", 1.0);
            features.put("feature7", 1.0);
            features.put("feature8", 0.0);

            decisionTree.predict(features);
        });

        Assertions.assertEquals("expected feature named 'feature1' but none provided",
                ex.getMessage());
    }

    @Test
    public void invalidFeatureCount() throws Exception {
        IllegalArgumentException ex = Assertions.assertThrows(IllegalArgumentException.class, () -> {
            ClassLoader cl = DecisionTreeClassifierTest.class.getClassLoader();
            Reader tree = new InputStreamReader(cl.getResourceAsStream("decision-tree.model"));
            final Classifier<Boolean> decisionTree = DecisionTreeClassifier.parse(tree,
                    new PredictionFactory.BooleanPredictionFactory());
            Assertions.assertNotNull(decisionTree);

            Map<String, Double> features = new HashMap<>();
            features.put("feature1", 0.0);
            features.put("feature2", 1.0);
            features.put("feature3", BooleanFeature.FALSE.asDouble());
            features.put("feature4", 0.0);
            features.put("feature5", BooleanFeature.FALSE.asDouble());
            features.put("feature6", 1.0);

            decisionTree.predict(features);
        });

        Assertions.assertEquals("expected feature named 'feature7' but none provided",
                ex.getMessage());
    }

    @Test
    public void probability() throws Exception {
        ClassLoader cl = DecisionTreeClassifierTest.class.getClassLoader();
        Reader tree = new InputStreamReader(cl.getResourceAsStream("decision-tree.model"));
        final Classifier<Boolean> decisionTree = DecisionTreeClassifier.parse(tree,
                new PredictionFactory.BooleanPredictionFactory());
        Assertions.assertNotNull(decisionTree);

        Map<String, Double> features = new HashMap<>();
        features.put("feature1", 1.2);
        features.put("feature2", 88.33);
        features.put("feature3", BooleanFeature.FALSE.asDouble());
        features.put("feature4", 1.727);
        features.put("feature5", BooleanFeature.FALSE.asDouble());
        features.put("feature6", 1.0);
        features.put("feature7", 0.0048);
        features.put("feature8", 0.0);

        Assertions.assertFalse(decisionTree.predict(features));

        double[] prediction = decisionTree.predict_proba(features);
        Assertions.assertNotNull(prediction);
        Assertions.assertEquals(0.63636364, prediction[0], .00000001);
        Assertions.assertEquals(0.36363636, prediction[1], .00000001);
    }

    @Test
    public void noWeights() throws Exception {
        ClassLoader cl = DecisionTreeClassifierTest.class.getClassLoader();
        Reader tree = new InputStreamReader(cl.getResourceAsStream("iris.model"));
        final Classifier<Integer> decisionTree = DecisionTreeClassifier.parse(tree,
                new PredictionFactory.IntegerPredictionFactory());
        Assertions.assertNotNull(decisionTree);

        Map<String, Double> features = new HashMap<>();
        features.put("sepal length (cm)", 3.0);
        features.put("sepal width (cm)", 5.0);
        features.put("petal length (cm)", 4.0);
        features.put("petal width (cm)", 2.0);

        Integer prediction = decisionTree.predict(features);
        Assertions.assertNotNull(prediction);

        Assertions.assertEquals(1, prediction.intValue());

        features.put("sepal length (cm)", 6.0);
        prediction = decisionTree.predict(features);
        Assertions.assertEquals(2, prediction.intValue());
    }
}