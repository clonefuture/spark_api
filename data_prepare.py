import argparse
import os
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, count, avg, split, regexp_replace, percentile_approx, row_number, collect_list, concat_ws

def main(input_dir, output_dir):
    #Создание сессии
    spark = SparkSession.builder.appName("Boston_crimes").getOrCreate()

    #Чтение файлов из директории в датафреймы
    crime_file = os.path.join(input_dir, "crime.csv")
    offense_file = os.path.join(input_dir, "offense_codes.csv")

    crime_df = spark.read.format("csv") \
        .option("header", "true") \
        .option("InferSchema", "true") \
        .load(crime_file)
    
    offense_df = spark.read.format("csv") \
        .option("header", "true") \
        .option("InferSchema", "true") \
        .load(offense_file)

    #Удаляем дубли и пустые строки, добавляем поле crime_type
    crime_df = crime_df.dropDuplicates().dropna(subset=["DISTRICT", "Lat", "Long"]).withColumn("offense_code", col("offense_code").cast("int"))
    offense_trans_df = offense_df.dropDuplicates(["code"]).dropna().withColumn("code", col("code").cast("int"))\
        .withColumn("crime_type", split(offense_df["NAME"], " - ")[0])
    offense_df = offense_trans_df.withColumn("crime_type", regexp_replace(offense_trans_df["crime_type"], ",", ""))

    #Объединяем датафреймы
    joined_df = crime_df.join(
        offense_df, crime_df["offense_code"] == offense_df["code"],
        "left"
    )

    #Агрегации
    #Общее количество преступлений в районе и среднее значение по долготе и широте
    crimes_total_df = joined_df.groupBy("district").agg(
        count("INCIDENT_NUMBER").alias("crimes_total"),
        avg("Lat").alias("lat"),
        avg("Long").alias("lng")
        )
    
    #медиана числа преступлений в месяц в этом районе
    crimes_by_month_df = joined_df.groupBy("district", "YEAR", "MONTH").agg(
        count("INCIDENT_NUMBER").alias("crimes_by_month")
    )
    crimes_monthly_df = crimes_by_month_df.groupBy("district").agg(
        percentile_approx("crimes_by_month", 0.5).alias("crimes_monthly")
    )

    #Три самых частых crime_type за всю историю наблюдений в этом районе
    #подсчет частоты типов преступлений в каждом районе
    crime_counts_df = joined_df.groupBy("district", "crime_type").agg(
        count("*").alias("frequency")
    )

    #добавление ранга по частоте внутри каждого района
    window_spec = Window.partitionBy("district").orderBy(col("frequency").desc())
    ranked_df = crime_counts_df.withColumn("rank", row_number().over(window_spec))

    #выбор топ 3 типа преступления по частоте
    top_crimes_df = ranked_df.filter(col("rank") <= 3)

    #Объединение типов преступлений в строку
    frequent_crime_types_df = top_crimes_df.groupBy("district").agg(
        concat_ws(", ", collect_list("crime_type")).alias("frequent_crime_types")
    )

    #Объединение датафреймов и выбор столбцов для витрины
    union_df = crimes_total_df.join(crimes_monthly_df, "district", "inner").join(frequent_crime_types_df, "district", "inner")
    total_df = union_df.select("district","crimes_total","crimes_monthly","frequent_crime_types","lat","lng")

    #Сохраняем результат в Parquet
    output_path = os.path.join(output_dir, "boston_crimes_stat")
    total_df.write.mode("overwrite").parquet(output_path)

    spark.stop()


if __name__ == "__main__":
    #Создаем парсер аргументов
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="path/to/output_folder")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)