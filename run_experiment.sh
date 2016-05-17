#!/usr/bin/env bash
baselines=("functions.Logistic" "bayes.NaiveBayes" "functions.MultilayerPerceptron" "trees.RandomForest" "functions.SMO" "bayes.BayesNet")
experiment=("logistic_regression" "naive_bayes" "neuralnets" "random_forest" "svm" "bayesnet")
EXPERIMENT_COUNT=$(expr ${#baselines[@]} - 1) # Size of baseline array minus 1
EXPERIMENT_DIRECTORY='experiments/summer'
DATABASES=("optima")
# DATABASES=("optima" "dbank" "all")
JARPATH='weka-3-8-0/weka.jar'

#Run experiments
for database in ${DATABASES[@]}; do
	for i in $(seq 0 $EXPERIMENT_COUNT); do
		training_set=$database"_"train.arff
		testing_set=$database"_"test.arff
		echo "------- Running -------"
		echo "Experiment: ${experiment[$i]}"
		echo "Training dataset: $training_set"
		echo "Testing dataset: $testing_set"
		# echo "java -cp $JARPATH weka.classifiers.${baselines[$i]} -t ./arff\ files/$training_set -T ./arff\ files/$testing_set > $EXPERIMENT_DIRECTORY/$database/${experiment[$i]}.txt"
		# java -cp $JARPATH weka.classifiers.${baselines[$i]} -t ./arff\ files/$training_set -T ./arff\ files/$testing_set > $EXPERIMENT_DIRECTORY/$database/${experiment[$i]}.txt
		java -cp $JARPATH weka.classifiers.${baselines[$i]} -t ./arff\ files/$training_set -d $EXPERIMENT_DIRECTORY/$database/models/${experiment[$i]}.model > $EXPERIMENT_DIRECTORY/$database/output/${experiment[$i]}_train_crossval.txt
		java -cp $JARPATH weka.classifiers.${baselines[$i]} -l $EXPERIMENT_DIRECTORY/$database/models/${experiment[$i]}.model -T ./arff\ files/$testing_set > $EXPERIMENT_DIRECTORY/$database/output/${experiment[$i]}_test.txt
		echo "-------- Done ---------"
		echo ""
	done
done
