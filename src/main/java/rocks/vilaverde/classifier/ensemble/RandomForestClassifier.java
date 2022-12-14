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
import rocks.vilaverde.classifier.util.ThrowingFunction;

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
     * @param <T> the classifier type
     * @throws Exception when the model could no be parsed
     */
    public static <T> Classifier<T> parse(final TarArchiveInputStream tar,
                                          PredictionFactory<T> factory) throws Exception {
        return RandomForestClassifier.parse(tar, factory, null);
    }

    /**
     * Accept a TAR of exported DecisionTreeClassifiers from sklearn and product a
     * RandomForestClassifier. This can be run in Parallel by
     * @param tar the Tar Archive input stream
     * @param factory the factory for creating the prediction class
     * @param executor An {@link ExecutorService} to run classification against the trees in parallel.
     * @return the {@link Classifier}
     * @param <T> the classifier type
     * @throws Exception when the model could no be parsed
     */
    public static <T> Classifier<T> parse(final TarArchiveInputStream tar,
                                          PredictionFactory<T> factory,
                                          ExecutorService executor) throws Exception {
        List<TreeClassifier<T>> forest = new ArrayList<>();

        try (tar) {
            TarArchiveEntry exportedTree;
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
                    TreeClassifier<T> tree = DecisionTreeClassifier.parse(reader, factory);
                    forest.add(tree);
                }
            }
        }
        
        return new RandomForestClassifier<>(forest, executor);
    }

    private final ExecutorService executorService;
    private final List<TreeClassifier<T>> forest;

    /**
     * Private Constructor
     * @param forest the random forest
     * @param executor the Executor service for parallel processing
     */
    private RandomForestClassifier(List<TreeClassifier<T>> forest, ExecutorService executor) {
        this.forest = forest;
        this.executorService = executor;
    }

    /**
     * Predict class or regression value for features.
     * @param features features of the sample
     * @return class probabilities of the input sample
     */
    @Override
    public T predict(Map<String, Double> features) {
        List<Prediction<T>> predictions = getPredictions(features);
        Map<T, Long> map = predictions.stream()
                .collect(Collectors.groupingBy(Prediction::get, Collectors.counting()));

        long max = map.values().stream().mapToLong(Long::longValue).max().orElse(0);
        for (Map.Entry<T, Long> entry : map.entrySet()) {
            if (entry.getValue() == max) {
                return entry.getKey();
            }
        }

        throw new IllegalStateException("no classification");
    }

    /**
     * Predict class probabilities of the input samples features.
     * The predicted class probability is the fraction of samples of the same class in a leaf.
     * @param features the input samples
     * @return the class probabilities of the input sample
     */
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

        if (result != null) {
            int forestSize = forest.size();
            for (int i = 0; i < result.length; i++) {
                result[i] /= forestSize;
            }
        }

        return result;
    }

    /**
     * Get all the predictions for the features of the sample
     * @param features features of the sample
     * @return a List of {@link Prediction} objects from the trees in the forest.
     */
    protected List<Prediction<T>> getPredictions(final Map<String, Double> features) {
        List<Prediction<T>> predictions;

        if (executorService != null) {
            int jobs = Runtime.getRuntime().availableProcessors();
            List<ParallelPrediction<T>> parallel = new ArrayList<>(jobs);
            for (int i = 0; i < jobs; i++) {
                ParallelPrediction<T> parallelPrediction = new ParallelPrediction<>(forest, features, i, jobs);
                parallel.add(parallelPrediction);
            }

            try {
                List<Future<List<Prediction<T>>>> futures = executorService.invokeAll(parallel);
                predictions = futures.stream()
                        .flatMap(ThrowingFunction.wrap(listFuture -> listFuture.get().stream()))
                        .collect(Collectors.toList());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

        } else {
            predictions = new ArrayList<>(forest.size());
            for (TreeClassifier<T> tree : forest) {
                Prediction<T> prediction = tree.getClassification(features);
                predictions.add(prediction);
            }
        }

        return predictions;
    }

    /**
     * Get the names of all the features in the model.
     * @return set of unique features used in the model
     */
    @Override
    public Set<String> getFeatureNames() {
        Set<String> features = new HashSet<>();
        for (Classifier<T> tree : forest) {
            features.addAll(tree.getFeatureNames());
        }
        return features;
    }


    /**
     * A job that will only provide {@link Prediction} results
     * from a subset of the trees in the forest.
     * @param <T> the classification class
     */
    private static class ParallelPrediction<T> implements Callable<List<Prediction<T>>> {
        private final int start;
        private final int offset;
        private final List<TreeClassifier<T>> forest;
        private final Map<String, Double> features;

        /**
         * Constructor
         * @param forest the random forest
         * @param features the features of a sample
         * @param start the index of the tree in the forest this job will begin with
         * @param offset the offest between the current tree and the next tree to call predict on
         */
        private ParallelPrediction(List<TreeClassifier<T>> forest,
                                   Map<String, Double> features,
                                   int start,
                                   int offset) {
            this.offset = offset;
            this.start = start;
            this.forest = forest;
            this.features = features;
        }

        /**
         * Will classify the sample against every 'offset' tree from the starting tree.
         * @return a {@link Prediction}
         * @throws Exception when the model could not product a prediction
         */
        @Override
        public List<Prediction<T>> call() throws Exception {
            List<Prediction<T>> predictions = new ArrayList<>();

            for (int i = start; i < forest.size(); i+=offset) {
                predictions.add(forest.get(i).getClassification(features));
            }

            return predictions;
        }
    }
}
