package com.fatec.iris.servico;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Debug;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
/*
 * 
 */

public class Test {

	
	public static final String DATASETPATH = "E:/data_set/iris_dataset.arff";
	public static final String MODElPATH = "E:/data_set/model.bin";

	public static void main(String[] args) throws Exception {

		GeradorDoModelo mg = new GeradorDoModelo();
        //Le as instancias de um arquivo ARFF, CSV, XRFF ...
		Instances dataset = mg.loadDataset(DATASETPATH);
		//Normaliza todos os valores numéricos no conjunto de dados fornecido (exceto o atributo de classe, se definido)
		//A normalização de dados é o processo de redimensionar um ou mais atributos para o intervalo de 0 a 1. 
		//Isso significa que o maior valor para cada atributo é 1 e o menor valor é 0.
		Filter filter = new Normalize();

		// divide dataset to train dataset 80% and test dataset 20%
		int trainSize = (int) Math.round(dataset.numInstances() * 0.8);
		int testSize = dataset.numInstances() - trainSize;
        //Embaralha as instâncias no conjunto para que sejam ordenadas aleatoriamente
		dataset.randomize(new Debug.Random(1));// if you comment this line the accuracy of the model will be droped from
												// 96.6% to 80%

		// Normalize dataset
		filter.setInputFormat(dataset);
		Instances datasetnor = Filter.useFilter(dataset, filter);
        //carrega as instancias de treino e de teste normalizadas
		Instances traindataset = new Instances(datasetnor, 0, trainSize);
		Instances testdataset = new Instances(datasetnor, trainSize, testSize);

		// build classifier with train dataset
		MultilayerPerceptron ann = (MultilayerPerceptron) mg.buildClassifier(traindataset);

		// Evaluate classifier with test dataset
		String evalsummary = mg.evaluateModel(ann, traindataset, testdataset);
		System.out.println("Evaluation: " + evalsummary);

		// Save model
		mg.saveModel(ann, MODElPATH);

		// classifiy a single instance - passa a instancia a ser avaliada e o modelo ja
		// treinado como parametro
		Classificacao cls = new Classificacao();
		String classname = cls.classifiy(Filter.useFilter(cls.createInstance(1.6, 0.2, 0), filter), MODElPATH);
		System.out.println(
				"\n The class name for the instance with petallength = 1.6 and petalwidth =0.2 is  " + classname);

	}

}
