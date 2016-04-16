#!/usr/bin/env bash



function runclassifiers() {     

	
	cd logistic_regression/

	java -cp "./weka.jar" weka.classifiers.functions.Logistic -t ../arff/$1_$2.arff > results.txt

	cd ../naive_bayes

	java -cp "./weka.jar" weka.classifiers.bayes.NaiveBayes -t ../arff/$1_$2.arff > results.txt


	cd ../neuralnets

	java -cp "./weka.jar" weka.classifiers.functions.MultilayerPerceptron -t ../arff/$1_$2.arff > results.txt


	cd ../random_forest

	java -cp "./weka.jar" weka.classifiers.trees.RandomForest -t ../arff/$1_$2.arff > results.txt


	cd ../svm

	java -cp "./weka.jar" weka.classifiers.functions.SMO-t ../arff/$1_$2.arff > results.txt

	cd ../bayesnet

	java -cp "./weka.jar"  weka.classifiers.bayes.BayesNet-t ../arff/$1_$2.arff > results.txt


	cd ../


}

cd experiments/

	cd clinical/

		cd all_features/
		
		runclassifiers "clinical" "all_features"

		cd ../top50

		runclassifiers "clinical"  "top50"


		cd ../

	cd ../mix/

		cd all_features/
		 
		runclassifiers "mix" "all_features"
		

		cd ../top50

		runclassifiers "mix" "top50"


	
