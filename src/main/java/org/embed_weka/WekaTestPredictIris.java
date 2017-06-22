package org.embed_weka;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

public class WekaTestPredictIris {

    public static void main(String[] args) throws Exception {

        ConverterUtils.DataSource source1 = new ConverterUtils.DataSource("dataset/embed-weka-dataset/iris.arff");
        Instances train = source1.getDataSet();
        
        
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        
        if (train.classIndex() == -1)
            train.setClassIndex(train.numAttributes() - 1);

        ConverterUtils.DataSource source2 = new ConverterUtils.DataSource("dataset/embed-weka-dataset/iris-test.arff");
        
        Instances test = source2.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        
        
        if (test.classIndex() == -1)
            test.setClassIndex(train.numAttributes() - 1);

        // model

        NaiveBayes naiveBayes = new NaiveBayes();
        naiveBayes.buildClassifier(train);
        
        
        // Model 
        Classifier cls = new weka.classifiers.trees.J48();
        cls.buildClassifier(train);
        System.out.println(cls);
        
        
        // Statistics
        //Classifier cls = new J48();
        Evaluation eval = new Evaluation(train);
        Random rand = new Random(1234);  // using seed = 1
        int folds = 10;
        eval.crossValidateModel(cls, train, folds, rand);
        System.out.println(eval.toSummaryString());
        
        
        // Detailed class statistics
        
        //Classifier cls = new J48();
        //Evaluation eval = new Evaluation(data);
        //Random rand = new Random(1);  // using seed = 1
        //int folds = 10;
        eval.crossValidateModel(cls, train, folds, rand);
        System.out.println(eval.toClassDetailsString());
        
        
        // Confusion matrix
        
        // Generating classifier evaluation output manually
        // https://weka.wikispaces.com/Generating+classifier+evaluation+output+manually
        // print stats 
       
//        Classifier cls = new J48();
        //Evaluation eval = new Evaluation(train);
//        Random rand = new Random(1234);  // using seed = 1
//        int folds = 10;
        eval.crossValidateModel(cls, train, folds, rand);
        System.out.println(eval.toMatrixString());
        
        

        // this does the trick  
        double label = naiveBayes.classifyInstance(test.instance(0));
        
       
        // Manual Test
        System.out.println("---------- Manual Text Print ------");        
        test.instance(0).setClassValue(label);
        test.instance(1).setClassValue(label);
        test.instance(5).setClassValue(label);

        
        // Manual test print
        System.out.println(test.instance(0).stringValue(4));
        System.out.println(test.instance(1).stringValue(4));        
        System.out.println(test.instance(5).stringValue(4));
        System.out.println("---------- End Manual Text Print ------");        
        

        System.out.println("---------- Loop Text Print ------");        
        System.out.println("Train Num Instances: " + train.numInstances());
        System.out.println("Test Num Instances: " + test.numInstances());

        
        // Loop 
        for (int i = 0; i < test.numInstances(); i++) {
        	
            test.instance(i).setClassValue(label);
            
//            System.out.println("-----------------------------");
            System.out.println(test.instance(i).stringValue(4));
//            System.out.println(test.instance(i).value(2));
//            System.out.println(test.instance(i).value(3));
//            System.out.println(test.instance(i).stringValue(4));
//            System.out.println("-----------------------------");
//            System.out.println("|");

         }
        
        
        
        
        
    }
}