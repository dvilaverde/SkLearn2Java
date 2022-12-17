package rocks.vilaverde.classifier.dt;

import rocks.vilaverde.classifier.AbstractTreeClassifier;
import rocks.vilaverde.classifier.FeatureVector;
import rocks.vilaverde.classifier.Operator;
import rocks.vilaverde.classifier.Prediction;
import rocks.vilaverde.classifier.dt.visitors.FeatureNameVisitor;
import rocks.vilaverde.classifier.dt.visitors.PredictVisitor;

import java.io.BufferedReader;
import java.io.Reader;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.Stack;
import java.util.stream.Collectors;

/**
 * Represents a DecisionTreeClassifier trained in scikit-learn and exported using
 * export_text.
 * @param <T> the Prediction Class
 */
public class DecisionTreeClassifier<T> extends AbstractTreeClassifier<T>
        implements TreeClassifier<T> {

  /**
   * Factory method to create the classifier from the {@link Reader}.
   * @param reader the input Reader
   * @param factory the factory used to convert the prediction class to the correct result type
   * @return the Classifier
   * @param <T> class
   * @throws Exception when the model could no be parsed
   */
  public static <T> DecisionTreeClassifier<T> parse(Reader reader, PredictionFactory<T> factory) throws Exception {

    try (reader) {
      DecisionTreeClassifier<T> classifier = new DecisionTreeClassifier<>(factory);

      classifier.load(reader);

      FeatureNameVisitor visitor = new FeatureNameVisitor();
      classifier.root.accept(visitor);
      classifier.featureNames = visitor.getFeatureNames();

      return classifier;
    }
  }

  private final PredictionFactory<T> predictionFactory;
  private DecisionNode root;
  private Set<String> featureNames;

  /**
   * Private constructor, use factory method to create.
   */
  private DecisionTreeClassifier(PredictionFactory<T> predictionFactory) {
    this.predictionFactory = predictionFactory;
  }

  /**
   * Predict class or regression value for features.
   * For the classification model, the predicted class for the features of the sample is returned.
   * @param samples Map of feature name to value
   * @return predicted class
   */
  public List<T> predict(FeatureVector ... samples) {
    return Arrays.stream(samples)
            .map(featureVector -> getClassification(featureVector).get())
            .collect(Collectors.toList());
  }

  /**
   * Predict class probabilities of the input features.
   * The predicted class probability is the fraction of samples of the same class in a leaf.
   * @param samples Map of feature name to value
   * @return the class probabilities of the input sample
   */
  @Override
  public double[][] predict_proba(FeatureVector ... samples) {
    double[][] probabilities = null;

    for (int i = 0; i < samples.length; i++) {
      double[] result = getClassification(samples[i]).getProbability();
      if (probabilities == null) {
        probabilities = new double[samples.length][result.length];
      }

      probabilities[i] = result;
    }

    return probabilities;
  }

  /**
   * Find the {@link Prediction} in the decision tree.
   */
  public Prediction<T> getClassification(FeatureVector sample) {
    validateFeatures(sample);
    return PredictVisitor.predict(sample, root);
  }

  /**
   * Validate the features provided are expected.
   */
  private void validateFeatures(FeatureVector sample) throws IllegalArgumentException {
    for (String f : featureNames) {
      if (!sample.hasFeature(f)) {
        throw new IllegalArgumentException(String.format("expected feature named '%s' but none provided", f));
      }
    }
  }

  public Set<String> getFeatureNames() {
    return featureNames;
  }

  /**
   * Parse a text representation created using tree.export_text() from sklearn into a DecisionTreeClassifier.
   */
  private void load(Reader reader) throws Exception {

    Stack<TreeNode> stack = new Stack<>();

    try (BufferedReader bufferedReader = new BufferedReader(reader)) {
      String line = bufferedReader.readLine();

      while (line != null) {
        if (line.length() != 0) {

          // remove the indentations from each line, this expects that export_text()
          // used the default indent level of 3.
          line = removeIndentations(line);
          if (!stack.isEmpty()) {
            processChildNode(stack, line);
          } else {
            processDecisionNode(stack, line);
          }
        }

        // read next line
        line = bufferedReader.readLine();
      }
    }

    root = (DecisionNode) stack.pop();
  }

  private void processChildNode(Stack<TreeNode> stack, String line) throws Exception {

    // end nodes of the decision tree will either start with weights or class depending
    // if show_weights=True was set on export.
    if (!(line.startsWith("weights: ") || line.startsWith("class: "))) {
      // we have another decision node, delegate over to that method to handle the decision node
      processDecisionNode(stack, line);
      return;
    }

    // we have a EndNode, so pop the stack should contain a ChoiceNode,
    // pop that and add the end as a child.
    EndNode<T> node = EndNode.create(line, predictionFactory);
    ((ChoiceNode)stack.pop()).addChild(node);

    // if the current choice has been popped check if the decision node
    // has 2 operations, and if so pop that one as well and any other
    // completed decision nodes.
    while (stack.size() > 1 && ((DecisionNode)stack.peek()).isComplete() ) {
      stack.pop();
    }
  }

  /**
   * Parses the given line into a {@link DecisionNode} or {@link ChoiceNode} on the stack.
   */
  private void processDecisionNode(Stack<TreeNode> stack, String line) {
    int indexOfOperator = getOperatorIndex(line);
    String feature = line.substring(0, indexOfOperator).trim();
    String[] operatorValue = line.substring(indexOfOperator).split(" ");
    Operator op = Operator.from(operatorValue[0]);

    // the 1 char operators cause the split to produce an empty array item
    int valuePosition = 1;
    if (op == Operator.EQ || op == Operator.LT || op == Operator.GT ) {
      valuePosition = 2;
    }

    Double value = Double.parseDouble(operatorValue[valuePosition]);

    DecisionNode decisionNode;
    if (stack.isEmpty()) {
      decisionNode = DecisionNode.create(feature);
      stack.push(decisionNode);
    } else {
      TreeNode peek = stack.peek();
      if (!(peek instanceof DecisionNode && ((DecisionNode)peek).getFeatureName().equals(feature))) {
        // new node needs to go under current choice
        decisionNode = DecisionNode.create(feature);

        // The choice node will now have a child so should be removed from the stack
        ((ChoiceNode)stack.pop()).addChild(decisionNode);
        stack.push(decisionNode);
      } else {
        decisionNode = (DecisionNode) peek;
      }
    }

    ChoiceNode choice = ChoiceNode.create(op, value);

    if (decisionNode.getLeft() == null) {
      decisionNode.setLeft(choice);
    } else {
      decisionNode.setRight(choice);
    }

    stack.push(choice);
  }

  private int getOperatorIndex(String line) {
    int idx = -1;
    for (Operator op : Operator.values()) {
      idx = line.indexOf(op.toString());
      if (idx >= 0) {
        break;
      }
    }

    return idx;
  }

  /**
   * Strip the indentations from the text line.
   */
  private String removeIndentations(String line) {
    int idx = line.indexOf("|--- ");
    while (idx >= 0) {
      line = line.substring(idx + 5);
      idx = line.indexOf("|--- ");
    }
    return line;
  }
}
