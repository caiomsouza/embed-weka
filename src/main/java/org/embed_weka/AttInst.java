/**
 * Original Source code from: https://stackoverflow.com/questions/33392547/weka-summary-statisticss-in-cli
 * 
 */


package org.embed_weka;


import weka.experiment.Stats;
import weka.core.AttributeStats;
//import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class AttInst {
    public static void main(String args[]) throws Exception{
        //load dataset
    	DataSource source = new DataSource("dataset/embed-weka-dataset/credit-g.arff");
        //DataSource source = new DataSource("D:/y.arff");
        //get instances object 
        Instances data = source.getDataSet();
        //set class index .. as the last attribute
        if (data.classIndex() == -1) {
           data.setClassIndex(data.numAttributes() - 1);
        }
        //get number of attributes (notice class is not counted)
        int numAttr = data.numAttributes() - 1;
        for (int i = 0; i < numAttr; i++) {
            //check if current attr is of type nominal
            if (data.attribute(i).isNominal()) {
                System.out.println("The "+i+"th Attribute is Nominal"); 
                //get number of values
                int n = data.attribute(i).numValues();
                System.out.println("The "+i+"th Attribute has: "+n+" values");
            }           

            //get an AttributeStats object
            AttributeStats as = data.attributeStats(i);
            int dC = as.distinctCount;
            System.out.println("The "+i+"th Attribute has: "+dC+" distinct values");

            //get a Stats object from the AttributeStats
            if (data.attribute(i).isNumeric()){
                System.out.println("The "+i+"th Attribute is Numeric"); 
                Stats s = as.numericStats;
                System.out.println("The "+i+"th Attribute has min value: "+s.min+" and max value: "+s.max+" and mean value: "+s.mean+" and stdDev value: "+s.stdDev );
            }

    }


    }
}