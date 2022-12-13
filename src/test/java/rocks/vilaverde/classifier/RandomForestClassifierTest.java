package rocks.vilaverde.classifier;

import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import rocks.vilaverde.classifier.dt.PredictionFactory;
import rocks.vilaverde.classifier.ensemble.RandomForestClassifier;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Tests for the RandomForestClassifier
 */
public class RandomForestClassifierTest {

    private static ExecutorService executorService;

    @BeforeAll
    public static void setup() {
        executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    }

    @AfterAll
    public static void teardown() {
        executorService.shutdownNow();
    }

    @Test
    public void randomForestParallel() throws Exception {
        TarArchiveInputStream exported = getExportedModel("rf/iris.tgz");
        final Classifier<Double> decisionTree = RandomForestClassifier.parse(exported,
                PredictionFactory.DOUBLE, executorService);
        Assertions.assertNotNull(decisionTree);

        double[] proba = decisionTree.predict_proba(getSample1());
        assertSample(proba, .06, .62, .32);

        Double prediction = decisionTree.predict(getSample1());
        Assertions.assertNotNull(prediction);
        Assertions.assertEquals(1.0, prediction, .0);

        prediction = decisionTree.predict(getSample2());
        Assertions.assertEquals(2, prediction.intValue());

        proba = decisionTree.predict_proba(getSample2());
        assertSample(proba, 0.0, .44, .56);
    }

    @Test
    public void randomForest() throws Exception {
        TarArchiveInputStream exported = getExportedModel("rf/iris.tgz");
        final Classifier<Double> decisionTree = RandomForestClassifier.parse(exported, PredictionFactory.DOUBLE);
        Assertions.assertNotNull(decisionTree);

        double[] proba = decisionTree.predict_proba(getSample1());
        assertSample(proba, .06, .62, .32);

        Double prediction = decisionTree.predict(getSample1());
        Assertions.assertNotNull(prediction);
        Assertions.assertEquals(1.0, prediction, .0);

        prediction = decisionTree.predict(getSample2());
        Assertions.assertEquals(2, prediction.intValue());

        proba = decisionTree.predict_proba(getSample2());
        assertSample(proba, 0.0, .44, .56);
    }

    @Test
    public void invalidFeatureCount() {
        IllegalArgumentException ex = Assertions.assertThrows(IllegalArgumentException.class, () -> {
            TarArchiveInputStream exported = getExportedModel("rf/iris.tgz");
            final Classifier<Double> decisionTree = RandomForestClassifier.parse(exported, PredictionFactory.DOUBLE);
            Assertions.assertNotNull(decisionTree);

            // create invalid number of features.
            Map<String, Double> features = getSample1();
            features.remove("sepal length (cm)");

            decisionTree.predict(features);
        });

        Assertions.assertEquals("expected feature named 'sepal length (cm)' but none provided",
                ex.getMessage());
    }

    @Test
    public void featureNames() throws Exception {
        TarArchiveInputStream exported = getExportedModel("rf/iris.tgz");
        final Classifier<Double> decisionTree = RandomForestClassifier.parse(exported, PredictionFactory.DOUBLE);
        Assertions.assertNotNull(decisionTree);

        Assertions.assertEquals(4, decisionTree.getFeatureNames().size());
    }

    private Map<String, Double> getSample1() {
        Map<String, Double> features = new HashMap<>();
        features.put("sepal length (cm)", 3.0);
        features.put("sepal width (cm)", 5.0);
        features.put("petal length (cm)", 4.0);
        features.put("petal width (cm)", 2.0);
        return features;
    }

    private Map<String, Double> getSample2() {
        Map<String, Double> features = new HashMap<>();
        features.put("sepal length (cm)", 1.0);
        features.put("sepal width (cm)", 2.0);
        features.put("petal length (cm)", 3.0);
        features.put("petal width (cm)", 4.0);
        return features;
    }

    private void assertSample(double[] proba, double expected, double expected1, double expected2) {
        Assertions.assertNotNull(proba);
        Assertions.assertEquals(expected, proba[0], .0);
        Assertions.assertEquals(expected1, proba[1], .0);
        Assertions.assertEquals(expected2, proba[2], .0);
    }

    private TarArchiveInputStream getExportedModel(String fileName) throws IOException {
        ClassLoader cl = DecisionTreeClassifierTest.class.getClassLoader();
        InputStream stream = cl.getResourceAsStream(fileName);
        if (stream == null) {
            throw new RuntimeException(String.format("no zip found with name %s", fileName));
        }
        return new TarArchiveInputStream(new GzipCompressorInputStream(stream));
    }
}
