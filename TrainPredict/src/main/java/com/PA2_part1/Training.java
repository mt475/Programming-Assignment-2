package com.PA2_part1;

import java.util.HashMap;
import java.util.Map;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.SparkFiles;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
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

/**
 * Hello world!
 *
 */
public class Training {
	public static void main(String[] args) {
		

		SparkConf conf = new SparkConf()
				.setAppName("Main");		
		JavaSparkContext sc = JavaSparkContext.fromSparkContext(SparkSession.builder().config(conf)
				.getOrCreate().sparkContext());
		//String dataFile = "TrainingDataset.csv";
		String dataFile=args[0];
		String path=args[1];
		System.out.println("Reading file : "+ dataFile);
		JavaRDD<String> data = sc.textFile(dataFile);
		
		SQLContext sqlContext = new SQLContext(sc);

		HashMap<String, String> options = new HashMap<String, String>();
		options.put("header", "true");
		options.put("path", dataFile);
		Dataset<Row> df =sqlContext.read().format("csv").option("header", "true").load(dataFile);

		System.out.println("Reading without data without label.");
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
 
		System.out.println("-------------------- TRAINING DATA STATS ---------------");
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
		// map.put("11", 11);


		System.out.println("Reading without data with labels.");
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
		// 6. Modeling
		// 6.1. Model Training
		LogisticRegressionModel model = new LogisticRegressionWithLBFGS().setNumClasses(10).run(labeledData.rdd());
		// 6.2. Model Evaluatio
	//	sc.sc().hadoopConfiguration().set("fs.defaultFS","hdfs://localhost:9000/");
		System.out.println("Saving the model.");
		model.save(sc.sc(), path);
	



	}

}
