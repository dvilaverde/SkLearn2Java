package rocks.vilaverde.classifier.dt.visitors;

import rocks.vilaverde.classifier.FeatureVector;
import rocks.vilaverde.classifier.Prediction;
import rocks.vilaverde.classifier.dt.AbstractDecisionTreeVisitor;
import rocks.vilaverde.classifier.dt.ChoiceNode;
import rocks.vilaverde.classifier.dt.DecisionNode;
import rocks.vilaverde.classifier.dt.EndNode;
import rocks.vilaverde.classifier.dt.TreeNode;

/**
 * Visits the nodes of the {@link rocks.vilaverde.classifier.dt.TreeClassifier} looking for the
 * {@link rocks.vilaverde.classifier.dt.EndNode} for the given {@link rocks.vilaverde.classifier.FeatureVector}.
 */
public class PredictVisitor<T> extends AbstractDecisionTreeVisitor {

    private final FeatureVector sample;
    private Prediction<T> prediction;

    /**
     * Convenience method to search a tree for a prediction.
     * @param sample the sample {@link FeatureVector}
     * @param root the root {@link TreeNode} of the Decision Tree
     * @return the {@link Prediction}
     * @param <T> the classification java type
     */
    public static <T> Prediction<T> predict(FeatureVector sample, TreeNode root) {
        PredictVisitor<T> visitor = new PredictVisitor<>(sample);
        root.accept(visitor);

        if (visitor.getPrediction() == null) {
            throw new RuntimeException("expected a prediction result from the tree, but none found");
        }

        return visitor.getPrediction();
    }

    /**
     * Constructor.
     * @param sample the {@link FeatureVector}
     */
    private PredictVisitor(FeatureVector sample) {
        this.sample = sample;
    }

    /**
     * When visiting a {@link DecisionNode} we need to test the left and right
     * {@link ChoiceNode} and visit only the one that evaluates to true.
     * @param object the {@link DecisionNode} being visited
     */
    @Override
    public void visit(DecisionNode object) {

        // don't call super otherwise both choice nodes are visited.

        double featureValue = this.sample.get(object.getFeatureName());
        if (object.getLeft().eval(featureValue)) {
            object.getLeft().getChild().accept(this);
        } else  if (object.getRight().eval(featureValue)) {
            object.getRight().getChild().accept(this);
        } else {
            throw new RuntimeException(String.format("no branches evaluated to true for feature '%s'",
                    object.getFeatureName()));
        }
    }

    /**
     * When visiting an {@link EndNode} we've found the prediction
     * and no longer need to visit the tree.
     * @param object the {@link EndNode} being visited
     */
    @Override
    public void visit(EndNode object) {
        this.prediction = object;
    }

    /**
     * Get the prediction by searching the decision tree.
     * @return the {@link Prediction}
     */
    public Prediction<T> getPrediction() {
        return prediction;
    }
}
