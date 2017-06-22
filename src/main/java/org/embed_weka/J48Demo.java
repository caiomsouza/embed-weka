package org.embed_weka;


import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.*;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;


public class J48Demo {
	
    public static void main(String[] args) throws Exception {

        Instances training_data = new Instances(new BufferedReader(
                new FileReader("dataset/embed-weka-dataset/weather-train.arff")));
        training_data.setClassIndex(training_data.numAttributes() - 1);

        Instances testing_data = new Instances(new BufferedReader(
                new FileReader("dataset/embed-weka-dataset/weather-test.arff")));
        testing_data.setClassIndex(training_data.numAttributes() - 1);

        String summary = training_data.toSummaryString();
        int number_samples = training_data.numInstances();
        int number_attributes_per_sample = training_data.numAttributes();
        System.out.println("Number of attributes in model = "
                + number_attributes_per_sample);
        System.out.println("Number of samples = " + number_samples);
        System.out.println("Summary: " + summary);
        System.out.println();

        J48 j48 = new J48();
        //ADTree adt = new ADTree();

        Remove rm = new Remove();

        rm.setAttributeIndices("1");

        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(rm);
        //fc.setClassifier(adt);
        fc.setClassifier(j48);
        
        

        fc.buildClassifier(training_data);

        for (int i = 0; i < testing_data.numInstances(); i++) {
            double pred = fc.classifyInstance(testing_data.instance(i));
            System.out.print("given value: "
                    + testing_data.classAttribute().value(
                            (int) testing_data.instance(i).classValue()));
            System.out.println(". predicted value: "
                    + testing_data.classAttribute().value((int) pred));

        }
    }
}