

package org.embed_weka;

import java.io.*;
import weka.core.*;
import weka.attributeSelection.*;

public class TestPCA {
    
    public static void main(String[] args) {
	try {
	    // make sure all plugin packages are loaded
	    WekaPackageManager.loadPackages(false, true, false);

	    // Added for test
    	String trainData = "dataset/embed-weka-dataset/titanic/train.arff";

	    
	    // path to an arff file in the first argument
//	    Instances insts = new Instances(new BufferedReader(new FileReader(args[0])));
	    Instances insts = new Instances(new BufferedReader(new FileReader(trainData)));

	    
	    // Assume the data has a class attribute, and it is the last attribute in
	    // the ARFF file. When a class attribute is set, it is not considered in the PCA
	    insts.setClassIndex(insts.numAttributes() - 1);
	    
	    // First run PCA and get results like you'd see in the Explorer's "Select Attributes" panel
	    AttributeSelection attSel = new AttributeSelection();
	    PrincipalComponents pca = new PrincipalComponents();
	    pca.setVarianceCovered(1.0);
	    Ranker ranker = new Ranker();
	    attSel.setEvaluator(pca);
	    attSel.setSearch(ranker);
	    attSel.SelectAttributes(insts);
	    String summaryString = attSel.toResultsString();
	    // textual results like you'd see in the Explorer
	    
	    // 
	    System.out.println("------------- PCA Summary String -------------------------");
	    System.out.println(summaryString);
	    System.out.println("------------- PCA Summary String -------------------------");
	    
	    System.out.println("Summary String Length" + summaryString.length());

	    System.out.println("selectedAttributes: " + attSel.selectedAttributes());	    
	    System.out.println("numberAttributesSelected: " + attSel.numberAttributesSelected());	    
	    System.out.println("rankedAttributes: " + attSel.rankedAttributes().length);	    
	    

	    
	    // Now run PCA as a data preprocessing filter
	    weka.filters.unsupervised.attribute.PrincipalComponents pcaFilter = 
		new weka.filters.unsupervised.attribute.PrincipalComponents();
	    pcaFilter.setVarianceCovered(1.0);
	    pcaFilter.setInputFormat(insts);
	    Instances transformedIntoPCASpace = weka.filters.Filter.useFilter(insts, pcaFilter);
	    // Print the transformed instances

	    System.out.println("------------- PCA transformedIntoPCASpace -------------------------");

	    System.out.println(transformedIntoPCASpace);
	    
	    System.out.println("------------- PCA transformedIntoPCASpace -------------------------");
	    
	    
	} catch (Exception ex) {
	    ex.printStackTrace();
	}
    }
}