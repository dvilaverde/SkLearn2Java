package rocks.vilaverde.classifier.dt;

import rocks.vilaverde.classifier.Classification;
import java.text.MessageFormat;
import java.util.Arrays;

class EndNode<T> extends TreeNode implements Classification<T> {

  private static final MessageFormat WEIGHTS_FORMAT = new MessageFormat("weights: {0} class: {1}");
  private static final MessageFormat CLASS_FORMAT = new MessageFormat("class: {0}");
  private final Prediction<T> prediction;

  public static <T> EndNode<T> create(String endNodeString,
                                  PredictionFactory<T> predictionFactory) throws Exception {

    MessageFormat format = CLASS_FORMAT;
    if (endNodeString.startsWith("weights:")) {
      format = WEIGHTS_FORMAT;

      final Object[] parse = format.parse(endNodeString);
      String wt = parse[0].toString();
      wt = wt.substring(1, wt.length() - 1);
      String[] wts = wt.split(",");
      double[] weights = Arrays.stream(wts).map(String::trim).mapToDouble(Double::valueOf).toArray();
      return new WeightedEndNode<>(weights, predictionFactory.create(parse[1].toString()));
    } else {
      final Object[] parse = format.parse(endNodeString);
      return new EndNode<>(predictionFactory.create(parse[0].toString()));
    }
  }

  private EndNode(Prediction<T> prediction) {
    this.prediction = prediction;
  }

  @Override
  public Prediction<T> getPrediction() {
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
    return "EndNode{" +
            "classification=" + prediction.get() +
            '}';
  }

  static class WeightedEndNode<T> extends EndNode<T> {

    private final double[] weights;

    private WeightedEndNode(double[] weights, Prediction<T> prediction) {
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
      return "WeightedEndNode{" +
              "weights=" + Arrays.toString(weights) +
              ", classification=" + getPrediction().get() +
              '}';
    }
  }
}
