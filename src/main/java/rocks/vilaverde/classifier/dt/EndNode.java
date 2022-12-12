package rocks.vilaverde.classifier.dt;

import rocks.vilaverde.classifier.Prediction;
import java.text.MessageFormat;
import java.util.Arrays;

/**
 * Represents the end of the tree, where no further decisions can be made. The end node contains
 * the prediction.
 */
class EndNode<T> extends TreeNode implements Prediction<T> {
  private static final MessageFormat CLASS_FORMAT = new MessageFormat("class: {0}");
  private final T prediction;

  /**
   * Factory method to create the appropriate {@link EndNode} from the
   * String in exported tree model.
   */
  public static <T> EndNode<T> create(String endNodeString,
                                  PredictionFactory<T> predictionFactory) throws Exception {

    if (endNodeString.startsWith("weights:")) {
      return WeightedEndNode.createWeightedNode(endNodeString, predictionFactory);
    } else {
      Object[] parse = CLASS_FORMAT.parse(endNodeString);
      return new EndNode<>(predictionFactory.create(parse[0].toString()));
    }
  }

  private EndNode(T prediction) {
    this.prediction = prediction;
  }

  @Override
  public T get() {
    return prediction;
  }

  @Override
  public double[] getProbability() {
    throw new IllegalStateException("model was not exported with weights, can't calculate probability");
  }

  @Override
  public void accept(AbstractDecisionTreeVisitor visitor) {
    visitor.visit(this);
  }

  @Override
  public String toString() {
    return "EndNode{" + "classification=" + prediction + '}';
  }


  /**
   * {@link EndNode} that supports calculating the probability from the
   * weights in the exported tree model.
   */
  static class WeightedEndNode<T> extends EndNode<T> {

    private static final MessageFormat WEIGHTS_FORMAT = new MessageFormat("weights: {0} class: {1}");

    private final double[] weights;

    static <T> EndNode<T> createWeightedNode(String endNodeString,
                                             PredictionFactory<T> predictionFactory) throws Exception {
      Object[] parse = WEIGHTS_FORMAT.parse(endNodeString);
      String wt = parse[0].toString();
      wt = wt.substring(1, wt.length() - 1);

      double[] weights = Arrays.stream(wt.split(","))
              .map(String::trim)
              .mapToDouble(Double::valueOf)
              .toArray();

      return new WeightedEndNode<>(weights, predictionFactory.create(parse[1].toString()));
    }

    /**
     * Constructor.
     */
    private WeightedEndNode(double[] weights, T prediction) {
      super(prediction);
      this.weights = weights;
    }

    @Override
    public double[] getProbability() {
      double totalSamples = 0;
      for (double w : weights) {
        totalSamples += w;
      }

      double[] result = new double[weights.length];
      for (int i = 0; i < weights.length; i++) {
        result[i] = weights[i] / totalSamples;
      }

      return result;
    }

    @Override
    public String toString() {
      return "WeightedEndNode{" + "weights=" + Arrays.toString(weights) + ", classification=" + get() + '}';
    }
  }
}
