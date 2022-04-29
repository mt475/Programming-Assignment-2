package com.PA2_part2;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;

import scala.Tuple2;

/**
 * Hello world!
 *
 */
public class PredictOnly 
{
    public static void main( String[] args )
    {
        //loading spark context 
        SparkConf conf = new SparkConf().setAppName("Main");
		JavaSparkContext sc = new JavaSparkContext(conf);
		String dataFile = args[0];
		String path=args[1];
		JavaRDD<String> validationDataset = sc.textFile(dataFile);
		
		SQLContext sqlContext = new SQLContext(sc);

		HashMap<String, String> options = new HashMap<String, String>();
		options.put("header", "true");
		options.put("path", dataFile);

		Dataset<Row> df = sqlContext.load("com.databricks.spark.csv", options);
		// Dataset<Row> df = sqlContext.load(dataFile, options);

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
		

		Map<String, Integer> map = new HashMap<>();
		//map.put("0", 0);
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
		// map.put("11", 11);

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
		
		LogisticRegressionModel sameModel = LogisticRegressionModel
				  .load(sc.sc(),path );
		
	
		// Compute raw scores on the test set.
		JavaPairRDD<Object, Object> predictionAndLabels = labeledData.mapToPair(p ->
		  new Tuple2<>(sameModel.predict(p.features()), p.label()));
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
