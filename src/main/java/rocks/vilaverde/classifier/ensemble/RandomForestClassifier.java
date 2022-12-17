package rocks.vilaverde.classifier.ensemble;

import org.apache.commons.compress.archivers.ArchiveEntry;
import org.apache.commons.compress.archivers.ArchiveInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import rocks.vilaverde.classifier.AbstractTreeClassifier;
import rocks.vilaverde.classifier.Classifier;
import rocks.vilaverde.classifier.FeatureVector;
import rocks.vilaverde.classifier.Prediction;
import rocks.vilaverde.classifier.dt.DecisionTreeClassifier;
import rocks.vilaverde.classifier.dt.PredictionFactory;
import rocks.vilaverde.classifier.dt.TreeClassifier;
import rocks.vilaverde.classifier.util.ThrowingFunction;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

/**
 * A forest of DecisionTreeClassifiers.
 */
public class RandomForestClassifier<T> extends AbstractTreeClassifier<T>
        implements Classifier<T> {
    private static final Logger LOG = LoggerFactory.getLogger(RandomForestClassifier.class);

    /**
     * Accept a TAR of exported DecisionTreeClassifiers from sklearn and product a
     * RandomForestClassifier. This default to running in a single (current) thread.
     * @param tar the {@link ArchiveInputStream}
     * @param factory the factory for creating the prediction class
     * @return the {@link Classifier}
     * @param <T> the classifier type
     * @throws Exception when the model could no be parsed
     */
    public static <T> Classifier<T> parse(final ArchiveInputStream tar,
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
    public static <T> Classifier<T> parse(final ArchiveInputStream tar,
                                          PredictionFactory<T> factory,
                                          ExecutorService executor) throws Exception {
        List<TreeClassifier<T>> forest = new ArrayList<>();

        try (tar) {
            ArchiveEntry exportedTree;
            while ((exportedTree = tar.getNextEntry()) != null) {
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
     * @param samples features of the sample
     * @return class probabilities of the input sample
     */
    @Override
    public List<T> predict(FeatureVector ... samples) {
        return Arrays.stream(samples)
                .map(this::predictSingle)
                .collect(Collectors.toList());
    }

    /**
     * Predict class probabilities of the input samples features.
     * The predicted class probability is the fraction of samples of the same class in a leaf.
     * @param samples the input samples
     * @return the class probabilities of the input sample
     */
    @Override
    public double[][] predict_proba(FeatureVector ... samples) {
        double[][] probabilities = null;

        for (int i = 0; i < samples.length; i ++) {
            FeatureVector fv = samples[i];
            Prediction<T> prediction = getClassification(fv);

            double[] sampleProb = prediction.getProbability();
            if (probabilities == null) {
                probabilities = new double[samples.length][sampleProb.length];
            }

            probabilities[i] = sampleProb;
        }

        return probabilities;
    }

    protected T predictSingle(FeatureVector sample) {
        return getClassification(sample).get();
    }

    /**
     * Return a prediction from the forest for the sample.
     * @param sample a samples feature vector
     * @return Prediction
     */
    @Override
    public Prediction<T> getClassification(FeatureVector sample) {
        final List<Prediction<T>> predictions = getPredictions(sample);
        return new RandomForestPrediction<>(predictions, forest.size());
    }

    /**
     * Get all the predictions for the features of the sample
     * @param sample features of the sample
     * @return a List of {@link Prediction} objects from the trees in the forest.
     */
    protected List<Prediction<T>> getPredictions(final FeatureVector sample) {
        List<Prediction<T>> predictions;

        if (executorService != null) {
            int jobs = Runtime.getRuntime().availableProcessors();
            List<ParallelPrediction<T>> parallel = new ArrayList<>(jobs);
            for (int i = 0; i < jobs; i++) {
                ParallelPrediction<T> parallelPrediction = new ParallelPrediction<>(forest, sample, i, jobs);
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
                Prediction<T> prediction = tree.getClassification(sample);
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


    static class RandomForestPrediction<T> implements Prediction<T> {
        private final List<Prediction<T>> predictions;
        private final int forestSize;

        /**
         * Constructor
         * @param predictions the list of predictions that need to be merged
         * @param forestSize the number of trees in the forest
         */
        public RandomForestPrediction(List<Prediction<T>> predictions, int forestSize) {
            this.predictions = predictions;
            this.forestSize = forestSize;
        }

        /**
         * @return The class.
         */
        @Override
        public T get() {
            Map<T, Long> map = predictions.stream()
                    .collect(Collectors.groupingBy(Prediction::get, Collectors.counting()));

            long max = map.values().stream().mapToLong(Long::longValue).max().orElse(0);
            T prediction = null;
            for (Map.Entry<T, Long> entry : map.entrySet()) {
                if (entry.getValue() == max) {
                    prediction =  entry.getKey();
                    break;
                }
            }

            if (prediction == null) {
                throw new IllegalStateException("no classification");
            }

            return prediction;
        }

        /**
         * @return the probability
         */
        @Override
        public double[] getProbability() {
            int arraySize = predictions.get(0).getProbability().length;
            final double[] result = new double[arraySize];
            for (Prediction<T> prediction : predictions) {
                double[] prob = prediction.getProbability();
                for (int j = 0; j < prob.length; j++) {
                    result[j] += prob[j];
                }
            }

            for (int j = 0; j < result.length; j++) {
                result[j] /= forestSize;
            }
            return result;
        }
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
        private final FeatureVector sample;

        /**
         * Constructor
         * @param forest the random forest
         * @param sample the features of a sample
         * @param start the index of the tree in the forest this job will begin with
         * @param offset the offest between the current tree and the next tree to call predict on
         */
        private ParallelPrediction(List<TreeClassifier<T>> forest,
                                   FeatureVector sample,
                                   int start,
                                   int offset) {
            this.offset = offset;
            this.start = start;
            this.forest = forest;
            this.sample = sample;
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
                predictions.add(forest.get(i).getClassification(sample));
            }

            return predictions;
        }
    }
}
