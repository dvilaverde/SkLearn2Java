package rocks.vilaverde.classifier.ensemble;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import rocks.vilaverde.classifier.Classifier;
import rocks.vilaverde.classifier.Prediction;
import rocks.vilaverde.classifier.dt.DecisionTreeClassifier;
import rocks.vilaverde.classifier.dt.PredictionFactory;
import rocks.vilaverde.classifier.dt.TreeClassifier;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * A forest of DecisionTreeClassifiers.
 */
public class RandomForestClassifier<T> implements Classifier<T> {
    private static final Logger LOG = LoggerFactory.getLogger(RandomForestClassifier.class);

    /**
     * Accept a TAR of exported DecisionTreeClassifiers from sklearn and product a
     * RandomForestClassifier. This default to running in a single (current) thread.
     * @param tar the Tar Archive input stream
     * @param factory the factory for creating the prediction class
     * @return the {@link Classifier}
     * @param <T> the classifer type
     * @throws Exception when the model could no be parsed
     */
    public static <T> Classifier<T> parse(final TarArchiveInputStream tar,
                                          PredictionFactory<T> factory) throws Exception {
        return RandomForestClassifier.parse(tar, factory, 0);
    }

    /**
     * Accept a TAR of exported DecisionTreeClassifiers from sklearn and product a
     * RandomForestClassifier. This can be run in Parallel by
     * @param tar the Tar Archive input stream
     * @param factory the factory for creating the prediction class
     * @param jobs the number of threads to use to search the forest. Using -1 will use all
     *             available threads.
     * @return the {@link Classifier}
     * @param <T> the classifer type
     * @throws Exception when the model could no be parsed
     */
    public static <T> Classifier<T> parse(final TarArchiveInputStream tar,
                                          PredictionFactory<T> factory,
                                          int jobs) throws Exception {
        List<TreeClassifier<T>> forest = new ArrayList<>();

        try (tar) {
            TarArchiveEntry exportedTree = null;
            while ((exportedTree = tar.getNextTarEntry()) != null) {
                if (!exportedTree.isDirectory()) {
                    LOG.debug("Parsing tree {}", exportedTree.getName());
                    final InputStream noCloseStream = new InputStream() {
                        @Override
                        public int read() throws IOException {
                            return tar.read();
                        }

                        @Override
                        public void close() throws IOException {
                            // don't close otherwise next file in tar won't be read.
                        }
                    };
                    BufferedReader reader = new BufferedReader(new InputStreamReader(noCloseStream));
                    TreeClassifier<T> tree = (TreeClassifier<T>) DecisionTreeClassifier.parse(reader, factory);
                    forest.add(tree);
                }
            }
        }
        
        return new RandomForestClassifier<>(forest, jobs);
    }

    private int jobs;
    private List<TreeClassifier<T>> forest;

    /**
     * Private Constructor
     * @param forest
     * @param jobs
     */
    private RandomForestClassifier(List<TreeClassifier<T>> forest, int jobs) {
        this.forest = forest;
        this.jobs = jobs;
    }

    @Override
    public T predict(Map<String, Double> features) {

        List<Prediction<T>> predictions = getPredictions(features);

        Map<T, Long> map = predictions.stream().collect(
                Collectors.groupingBy(Prediction::get, Collectors.counting())
        );

        long max = map.values().stream().mapToLong(Long::longValue).max().getAsLong();
        for (Map.Entry<T, Long> entry : map.entrySet()) {
            if (entry.getValue() == max) {
                return entry.getKey();
            }
        }

        throw new IllegalStateException("no classification");
    }

    @Override
    public double[] predict_proba(Map<String, Double> features) {
        if (forest.size() == 1) {
            return forest.get(0).getClassification(features).getProbability();
        }

        List<Prediction<T>> predictions = getPredictions(features);

        double[] result = null;

        for (Prediction<T> prediction : predictions) {
            double[] prob = prediction.getProbability();

            if (result == null) {
                result = prob;
            } else {
                for (int i = 0; i < prob.length; i++) {
                    result[i] += prob[i];
                }
            }
        }

        int forestSize = forest.size();
        for (int i = 0; i < result.length; i++) {
            result[i] /= forestSize;
        }

        return result;
    }

    protected List<Prediction<T>> getPredictions(Map<String, Double> features) {
        final List<Prediction<T>> predictions = new ArrayList<>(forest.size());

        if (jobs == -1) {
            jobs = Runtime.getRuntime().availableProcessors();
        }

        if (jobs > 0) {
            ExecutorService executor = Executors.newFixedThreadPool(jobs);

            try {
                for (TreeClassifier<T> tree : forest) {
                    executor.submit(() -> {
                        Prediction<T> pred = tree.getClassification(features);
                        synchronized (predictions) {
                            predictions.add(pred);
                        }
                    });
                }
            } finally {
                executor.shutdown();
                try {
                    executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
                } catch (Exception e) {
                    LOG.error("interrupted while searching trees", e);
                }
            }
        } else {
            for (TreeClassifier<T> tree : forest) {
                Prediction<T> prediction = tree.getClassification(features);
                predictions.add(prediction);
            }
        }

        return predictions;
    }

    @Override
    public Set<String> getFeatureNames() {
        Set<String> features = new HashSet<>();
        for (Classifier<T> tree : forest) {
            features.addAll(tree.getFeatureNames());
        }
        return features;
    }
}
