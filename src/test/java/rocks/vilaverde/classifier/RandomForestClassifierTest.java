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
import java.util.ArrayList;
import java.util.List;
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

        double[] proba = decisionTree.predict_proba(getSample1())[0];
        assertSample(proba, .06, .62, .32);

        Double prediction = decisionTree.predict(getSample1()).get(0);
        Assertions.assertNotNull(prediction);
        Assertions.assertEquals(1.0, prediction, .0);

        prediction = decisionTree.predict(getSample2()).get(0);
        Assertions.assertEquals(2, prediction.intValue());

        proba = decisionTree.predict_proba(getSample2())[0];
        assertSample(proba, 0.0, .44, .56);
    }

    @Test
    public void randomForestParallel10000() throws Exception {
        TarArchiveInputStream exported = getExportedModel("rf/iris.tgz");
        final Classifier<Double> decisionTree = RandomForestClassifier.parse(exported,
                PredictionFactory.DOUBLE, executorService);
        Assertions.assertNotNull(decisionTree);

        List<FeatureVector> vectorList = new ArrayList<>(10000);
        for (int i = 0; i<  5000; i++) {
            vectorList.add(getSample1());
            vectorList.add(getSample2());
        }

        double[][] proba = decisionTree.predict_proba(vectorList.toArray(new FeatureVector[0]));
        assertSample(proba[0], .06, .62, .32);
        assertSample(proba[1], 0.0, .44, .56);
    }

    @Test
    public void randomForest() throws Exception {
        TarArchiveInputStream exported = getExportedModel("rf/iris.tgz");
        final Classifier<Double> decisionTree = RandomForestClassifier.parse(exported, PredictionFactory.DOUBLE);
        Assertions.assertNotNull(decisionTree);

        double[] proba = decisionTree.predict_proba(getSample1())[0];
        assertSample(proba, .06, .62, .32);

        Double prediction = decisionTree.predict(getSample1()).get(0);
        Assertions.assertNotNull(prediction);
        Assertions.assertEquals(1.0, prediction, .0);

        prediction = decisionTree.predict(getSample2()).get(0);
        Assertions.assertEquals(2, prediction.intValue());

        proba = decisionTree.predict_proba(getSample2())[0];
        assertSample(proba, 0.0, .44, .56);
    }

    @Test
    public void invalidFeatureCount() {
        IllegalArgumentException ex = Assertions.assertThrows(IllegalArgumentException.class, () -> {
            TarArchiveInputStream exported = getExportedModel("rf/iris.tgz");
            final Classifier<Double> decisionTree = RandomForestClassifier.parse(exported, PredictionFactory.DOUBLE);
            Assertions.assertNotNull(decisionTree);

            // create invalid number of features.
            Features features = Features.of("sepal width (cm)", "petal length (cm)", "petal width (cm)");
            FeatureVector fv = features.newSample();
            fv.add(0, 3.0).add(1, 5.0).add(2, 4.0);
            decisionTree.predict(fv);
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

    private FeatureVector getSample1() {
        Features features = Features.of("sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)");
        FeatureVector fv = features.newSample();
        return fv.add(0, 3.0)
                .add(1, 5.0)
                .add(2, 4.0)
                .add(3, 2.0);
    }

    private FeatureVector getSample2() {
        Features features = Features.of("sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)");
        FeatureVector fv = features.newSample();
        return fv.add(0, 1.0)
                .add(1, 2.0)
                .add(2, 3.0)
                .add(3, 4.0);
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
