#!/usr/bin/env bash

function runclassifiers() {     

	
	cd logistic_regression/

	java -cp "./../../../weka.jar" weka.classifiers.functions.Logistic -t ../../../arff_files/$1_$2.arff > results.txt

	cd ../naive_bayes

	java -cp "./../../../weka.jar"  weka.classifiers.bayes.NaiveBayes -t ../../../arff_files/$1_$2.arff > results.txt


	cd ../neuralnets

	java -cp "./../../../weka.jar"  weka.classifiers.functions.MultilayerPerceptron -t ../../../arff_files/$1_$2.arff > results.txt


	cd ../random_forest

	java -cp "./../../../weka.jar"  weka.classifiers.trees.RandomForest -t ../../../arff_files/$1_$2.arff > results.txt


	cd ../svm

	java -cp "./../../../weka.jar"  weka.classifiers.functions.SMO -t ../../../arff_files/$1_$2.arff > results.txt

	cd ../bayesnet

	java -cp "./../../../weka.jar"   weka.classifiers.bayes.BayesNet -t ../../../arff_files/$1_$2.arff > results.txt


	cd ../


}

cd experiments/

cd special_experiment


runclassifiers "all_all" "fraser_mix"