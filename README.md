# SkLearn2Java

This project aims to used text exported ML models generated by sci-kit learn and make them usable in Java.

[![javadoc](https://javadoc.io/badge2/rocks.vilaverde/scikit-learn-2-java/javadoc.svg)](https://javadoc.io/doc/rocks.vilaverde/scikit-learn-2-java)

## Support
* The tree.DecisionTreeClassifier is supported
  * Supports `predict()`,
  * Supports `predict_proba()` when `export_text()` configured with `show_weights=True`
* The tree.RandomForestClassifier is supported
  * Supports `predict()`,
  * Supports `predict_proba()` when `export_text()` configured with `show_weights=True`

## Installing

### Importing Maven Dependency
```xml
<dependency>
  <groupId>rocks.vilaverde</groupId>
  <artifactId>scikit-learn-2-java</artifactId>
  <version>1.1.0</version>
</dependency>
```

## DecisionTreeClassifier

As an example, a DecisionTreeClassifier model trained on the Iris dataset and exported using `sklearn.tree`
`export_text()` as shown below:

```
>>> from sklearn.datasets import load_iris
>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.tree import export_text
>>> iris = load_iris()
>>> X = iris['data']
>>> y = iris['target']
>>> decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
>>> decision_tree = decision_tree.fit(X, y)
>>> r = export_text(decision_tree, feature_names=iris['feature_names'], show_weights=True, max_depth=sys.maxsize)
>>> print(r)

|--- petal width (cm) <= 0.80
|   |--- class: 0
|--- petal width (cm) >  0.80
|   |--- petal width (cm) <= 1.75
|   |   |--- class: 1
|   |--- petal width (cm) >  1.75
|   |   |--- class: 2
```

The exported text can then be executed in Java. Note that when calling `export_text` it is 
recommended that `max_depth` be set to `sys.maxsize` so that the tree isn't truncated.

### Java Example
In this example the iris model exported using `export_text` is parsed, features are created as a Java Map
and the decision tree is asked to predict the class.

```
    Reader tree = getTrainedModel("iris.model");
    final Classifier<Integer> decisionTree = DecisionTreeClassifier.parse(tree,
                PredictionFactory.INTEGER);

    Features features = Features.of("sepal length (cm)",
                "sepal width (cm)",
                "petal length (cm)",
                "petal width (cm)");
    FeatureVector fv = features.newSample();
    fv.add(0, 3.0).add(1, 5.0).add(2, 4.0).add(3, 2.0);
    
    Integer prediction = decisionTree.predict(fv);
    System.out.println(prediction.toString());
```

## RandomForestClassifier

To use a RandomForestClassifier that has been trained on the Iris dataset, each of the `estimators`  
in the classifiers need to be and exported using `from sklearn.tree export export_text` as shown below:

```
>>> from sklearn import datasets
>>> from sklearn import tree
>>> from sklearn.ensemble import RandomForestClassifier
>>> 
>>> import os
>>> 
>>> iris = datasets.load_iris()
>>> X = iris.data
>>> y = iris.target
>>> 
>>> clf = RandomForestClassifier(n_estimators = 50, n_jobs=8)
>>> model = clf.fit(X, y)
>>> 
>>> for i, t in enumerate(clf.estimators_):
>>>     with open(os.path.join('/tmp/estimators', "iris-" + str(i) + ".txt"), "w") as file1:
>>>         text_representation = tree.export_text(t, feature_names=iris.feature_names, show_weights=True, decimals=4, max_depth=sys.maxsize)
>>>         file1.write(text_representation)
```

Once all the estimators are exported into `/tmp/estimators`, you can create a TAR archive, for example:
```bash
cd /tmp/estimators
tar -czvf /tmp/iris.tgz .
```

Then you can use the RandomForestClassifier class to parse the TAR archive.

```
    import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
    ...
    
    TarArchiveInputStream tree = getArchive("iris.tgz");
    final Classifier<Double> decisionTree = RandomForestClassifier.parse(tree,
                PredictionFactory.DOUBLE);
```

## Testing
Testing was done using models exported using sci-kit learn version 1.1.3, but should 
work with newer versions of sci-kit learn.
