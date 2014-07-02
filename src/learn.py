#!/usr/bin/env python
from sys import exit
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
import numpy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_stars(input='../data/fort.22'):
  data = numpy.column_stack((numpy.loadtxt(input), numpy.asarray(
      [0]*89    + # anomalous cepheids
      [1]*8026  + # cepheids
      [2]*2830  + # delta scuti
      [3]*43859 + # eclipsing
      [4]*8543  + # long period mira
      [5]*44262 + # rr lyrae
      [6]*603)))  # type 2 cepheids
  data = data[numpy.all(data>=0, axis=1)] # Remove negatives
  return type("", (), dict(
    data = data.T[:-1].T,
    target = data.T[-1],
    target_names = ['ACEP','CEP','DSCT','ECL','LPMIRA','RRLYR','T2CEP']))()

def main():
  options = get_options()
  stars = load_stars(options.input)
  classifiers = [
    ('Gaussian Naive Bayes',
      GaussianNB()),
    ('Logistic Regression',
      LogisticRegression()),
    ('Random Forest',
      GridSearchCV(RandomForestClassifier(n_jobs=options.n_jobs),
                   {n_estimators: [10, 50, 100, 500]}, scoring=options.scoring,
                   n_jobs=options.n_jobs, cv=options.cv)),
    ('Extremely Randomized Trees',
      GridSearchCV(ExtraTreesClassifier(n_jobs=options.n_jobs),
                   {n_estimators: [10, 50, 100, 500]}, scoring=options.scoring,
                   n_jobs=options.n_jobs, cv=options.cv)),
    ('Linear SVC',
      SVC(kernel='linear', C=1., probability=True, random_state=0))]
  for name, classifier in classifiers:
    for i in [0, 3, 7, 10]:
      if i > 0:
        pca = PCA(n_components=i).fit(stars.data).transform(stars.data)
        print(name, "using", i, "Principal Components")
        test(pca, stars.target, stars.target_names, classifier)
      else:
        print(name, "using raw lightcurves")
        test(stars.data, stars.target, stars.target_names, classifier)

def test(X, y, target_names, classifier, cv=10):
  scores = cross_val_score(classifier, X, y, cv=cv)
  print("%d-fold Cross Validation Accuracy: %0.5f (+/- %0.5f)"
         %(cv, scores.mean(), scores.std()*2))
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  y_pred = classifier.fit(X_train, y_train).predict(X_test)
  print(confusion_matrix(y_test, y_pred), "\n")

def get_options():
  from optparse import OptionParser
  parser = OptionParser()
  parser.add_option('-i', '--input', dest='input', type='string',
    default='../data/fort.22')
  parser.add_option('-n', '--n_jobs', dest='n_jobs', default=-1)
  parser.add_option('-f', '--folds', dest='cv', default=3)
  parser.add_option('-s', '--scoring', dest='scoring', default='accuracy')
  (options, args) = parser.parse_args()
  return options

if __name__ == "__main__":
  exit(main())
