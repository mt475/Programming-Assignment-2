package com.PA2_part1;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.stat.Statistics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.SparkSession;

import scala.Tuple2;

public class TrainingAndPrediction {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		SparkConf conf = new SparkConf()
				.setAppName("Main");		
		JavaSparkContext sc = JavaSparkContext.fromSparkContext(SparkSession.builder().config(conf)
				.getOrCreate().sparkContext());
		//String dataFile = "TrainingDataset.csv";
		String trainingFile=args[0];
		String validationFile=args[1];
		System.out.println("----------------- Reading training file : "+ trainingFile+" ---------------- ");
		JavaRDD<String> data = sc.textFile(trainingFile);
		
		SQLContext sqlContext = new SQLContext(sc);

	Dataset<Row> df =sqlContext.read().format("csv").option("header", "true").load(trainingFile);

	
		System.out.println("-------------------- TRAINING DATA STATS ---------------");
		JavaRDD<Vector> inputData = df.javaRDD().map(new Function<Row, Vector>() {
			public Vector call(Row row) throws Exception {
				String[] parts = row.get(0).toString().split(";");
				double[] v = new double[parts.length - 1];
				for (int i = 0; i < parts.length - 1; i++) {
					v[i] = Double.parseDouble(parts[i]);
				}

				return Vectors.dense(v);
			}
		});
		;

		// 3.2. Performing Statistical Analysis
		MultivariateStatisticalSummary summary = Statistics.colStats(inputData.rdd());
		System.out.println("Summary Mean:");
		System.out.println(summary.mean());
		System.out.println("Summary Variance:");
		System.out.println(summary.variance());
		System.out.println("Summary Non-zero:");
		System.out.println(summary.numNonzeros());

		Map<String, Integer> map = new HashMap<>();
		map.put("1", 1);
		map.put("2", 2);
		map.put("3", 3);
		map.put("4", 4);
		map.put("5", 5);
		map.put("6", 6);
		map.put("7", 7);
		map.put("8", 8);
		map.put("9", 9);
		map.put("10", 10);

		JavaRDD<LabeledPoint> labeledData = df.javaRDD().map(new Function<Row, LabeledPoint>() {
			public LabeledPoint call(Row row) throws Exception {
				String[] parts = row.get(0).toString().split(";");
				double[] v = new double[parts.length];
				for (int i = 0; i < parts.length - 1; i++) {
					v[i] = Double.parseDouble(parts[i]);
				}
				return new LabeledPoint(map.get(parts[parts.length - 1]), Vectors.dense(v));
			}
		});
		System.out.println("------------------ Reading validation file : "+ validationFile+"  --------------------------");
		
		
		Dataset<Row> validationDF =sqlContext.read().format("csv").option("header", "true").load(validationFile);

		JavaRDD<LabeledPoint> validationLabeledData = validationDF.javaRDD().map(new Function<Row, LabeledPoint>() {
			public LabeledPoint call(Row row) throws Exception {
				String[] parts = row.get(0).toString().split(";");
				double[] v = new double[parts.length];
				for (int i = 0; i < parts.length - 1; i++) {
					v[i] = Double.parseDouble(parts[i]);
				}
				return new LabeledPoint(map.get(parts[parts.length - 1]), Vectors.dense(v));
			}
		});
		// 6. Modeling
		// 6.1. Model Training
		LogisticRegressionModel model = new LogisticRegressionWithLBFGS().setNumClasses(10).run(labeledData.rdd());
		// 6.2. Model Evaluatio

		// Compute raw scores on the test set.
		JavaPairRDD<Object, Object> predictionAndLabels = validationLabeledData.mapToPair(p ->
		  new Tuple2<>(model.predict(p.features()), p.label()));
		// Get evaluation metrics.
		MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());

		// Confusion matrix
		Matrix confusion = metrics.confusionMatrix();
		System.out.println("----------------- Confusion matrix: \n" + confusion);

		// Overall statistics
		System.out.println("-------------------- Accuracy = " + metrics.accuracy());

		//Weighted stats
		System.out.format("-------------------- Weighted precision = %f\n", metrics.weightedPrecision());
		System.out.format("-------------------- Weighted recall = %f\n", metrics.weightedRecall());
		System.out.format("-------------------- Weighted F1 score = %f\n", metrics.weightedFMeasure());
		System.out.format("-------------------- Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());

 
	
	}

}
