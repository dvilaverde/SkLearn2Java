package rocks.vilaverde.classifier;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import rocks.vilaverde.classifier.dt.DecisionTreeClassifier;
import rocks.vilaverde.classifier.dt.PredictionFactory;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;


public class DecisionTreeClassifierTest {

    @Test
    public void parseSimpleTree() throws Exception {
        Reader tree = getExportedModel("simple-tree.model");
        final Classifier<Boolean> decisionTree = DecisionTreeClassifier.parse(tree, PredictionFactory.BOOLEAN);
        Assertions.assertNotNull(decisionTree);

        Features features = Features.of("feature1");
        FeatureVector sample1 = features.newSample().add(0, 1.2);
        Assertions.assertFalse(decisionTree.predict(sample1).get(0));

        FeatureVector sample2 = features.newSample().add(0, 2.4);
        Assertions.assertTrue(decisionTree.predict(sample2).get(0));

        Assertions.assertNotNull(decisionTree.getFeatureNames());
        Assertions.assertEquals(1, decisionTree.getFeatureNames().size());
        Assertions.assertEquals("feature1", decisionTree.getFeatureNames().iterator().next());
    }

    @Test
    public void parseDeepTree() throws Exception {
        Reader tree = getExportedModel("decision-tree.model");
        final Classifier<Boolean> decisionTree = DecisionTreeClassifier.parse(tree, PredictionFactory.BOOLEAN);
        Assertions.assertNotNull(decisionTree);

        Features features = Features.of("feature1", "feature2", "feature3", "feature4",
                "feature5", "feature6", "feature7", "feature8");
        FeatureVector fv = features.newSample();
        fv.add("feature1", 0.0)
            .add("feature2", 1.0)
            .add("feature3", false)
            .add("feature4", 0.0)
            .add("feature5", false)
            .add("feature6", 1.0)
            .add("feature7", 1.0)
            .add("feature8", 0.0);

        Assertions.assertFalse(decisionTree.predict(fv).get(0));

        fv.add("feature5", true);
        Assertions.assertTrue(decisionTree.predict(fv).get(0));

        Assertions.assertNotNull(decisionTree.getFeatureNames());
        Assertions.assertEquals(8, decisionTree.getFeatureNames().size());
    }

    @Test
    public void invalidFeatureName() {
        IllegalArgumentException ex = Assertions.assertThrows(IllegalArgumentException.class, () -> {
            Reader tree = getExportedModel("decision-tree.model");
            final Classifier<Boolean> decisionTree = DecisionTreeClassifier.parse(tree, PredictionFactory.BOOLEAN);
            Assertions.assertNotNull(decisionTree);

            Features features = Features.of("feature11", "feature2", "feature3", "feature4",
                    "feature5", "feature6", "feature7", "feature8");
            FeatureVector fv = features.newSample();
            fv.add(0, 0.0)
                    .add(1, 1.0)
                    .add(2, false)
                    .add(3, 0.0)
                    .add(4, false)
                    .add(5, 1.0)
                    .add(6, 1.0)
                    .add(7, 0.0);

            decisionTree.predict(fv);
        });

        Assertions.assertEquals("expected feature named 'feature1' but none provided",
                ex.getMessage());
    }

    @Test
    public void invalidFeatureCount() {
        IllegalArgumentException ex = Assertions.assertThrows(IllegalArgumentException.class, () -> {
            Reader tree = getExportedModel("decision-tree.model");
            final Classifier<Boolean> decisionTree = DecisionTreeClassifier.parse(tree, PredictionFactory.BOOLEAN);
            Assertions.assertNotNull(decisionTree);

            Features features = Features.of("feature1", "feature2", "feature3", "feature4",
                    "feature5", "feature6");
            FeatureVector fv = features.newSample();
            fv.add(0, 0.0)
                    .add(1, 1.0)
                    .add(2, false)
                    .add(3, 0.0)
                    .add(4, false)
                    .add(5, 1.0);

            decisionTree.predict(fv);
        });

        Assertions.assertEquals("expected feature named 'feature7' but none provided",
                ex.getMessage());
    }

    @Test
    public void probability() throws Exception {
        Reader tree = getExportedModel("decision-tree.model");
        final Classifier<Boolean> decisionTree = DecisionTreeClassifier.parse(tree, PredictionFactory.BOOLEAN);
        Assertions.assertNotNull(decisionTree);

        Features features = Features.of("feature1", "feature2", "feature3", "feature4",
                "feature5", "feature6", "feature7", "feature8");
        FeatureVector fv = features.newSample();
        fv.add("feature1", 1.2)
                .add("feature2", 88.33)
                .add("feature3", false)
                .add("feature4", 1.727)
                .add("feature5", false)
                .add("feature6", 1.0)
                .add("feature7", 0.0048)
                .add("feature8", 0.0);

        Assertions.assertFalse(decisionTree.predict(fv).get(0));

        double[] prediction = decisionTree.predict_proba(fv)[0];
        Assertions.assertNotNull(prediction);
        Assertions.assertEquals(0.63636364, prediction[0], .00000001);
        Assertions.assertEquals(0.36363636, prediction[1], .00000001);
    }

    @Test
    public void noWeights() throws Exception {
        Reader tree = getExportedModel("iris.model");
        final Classifier<Integer> decisionTree = DecisionTreeClassifier.parse(tree, PredictionFactory.INTEGER);
        Assertions.assertNotNull(decisionTree);

        Features features = Features.of("sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)");
        FeatureVector fv = features.newSample();
        fv.add(0, 3.0).add(1, 5.0).add(2, 4.0).add(3, 2.0);
        Integer prediction = decisionTree.predict(fv).get(0);
        Assertions.assertNotNull(prediction);

        Assertions.assertEquals(1, prediction.intValue());

        fv.add("sepal length (cm)", 6.0);
        prediction = decisionTree.predict(fv).get(0);
        Assertions.assertEquals(2, prediction.intValue());
    }

    private Reader getExportedModel(String fileName) {
        ClassLoader cl = DecisionTreeClassifierTest.class.getClassLoader();
        InputStream stream = cl.getResourceAsStream(fileName);
        if (stream == null) {
            throw new RuntimeException(String.format("no model found with name %s", fileName));
        }
        return new InputStreamReader(stream);
    }
}