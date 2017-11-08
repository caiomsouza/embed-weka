package org.embed_weka;

import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

public class CsvToArff {
    public static void main(String[] args) {
        try {
        	
        	// Before you execute this code, please download the dataset at https://raw.githubusercontent.com/minervarobots/minerva-auto-mining/master/minerva-auto-mining/dataset/csv-dataset/kaggle-santander-train.csv
        	
        	// Place the file at the folder dataset/embed-weka-dataset
                    	
        	//String f1 = "dataset/embed-weka-dataset/kaggle-santander-train.csv";
        	String f1 = "dataset/telemarketing.csv";        
        	
            //String f2 = "dataset/embed-weka-dataset/kaggle-santander-train.arff";
            String f2 = "dataset/telemarketing.arff";

            
            
            System.out.println("Starting ....");
            
            // load the CSV file (input file)
            CSVLoader loader = new CSVLoader();
            loader.setSource(new File(f1));
            String [] options = new String[1];
            options[0]="-H";
            loader.setOptions(options);

            Instances data = loader.getDataSet();
            System.out.println(data);
            
            System.out.println("Dataset converted to ARFF");
            

            // save as an  ARFF (output file)
            ArffSaver saver = new ArffSaver();
            saver.setInstances(data);
            saver.setFile(new File(f2));
            //saver.setDestination(new File(f2));
            saver.writeBatch();
        } catch(Exception e) {
        }
    }
}